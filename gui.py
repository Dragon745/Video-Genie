import sys
import os
import time
import shutil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QLabel, QPushButton, QLineEdit, QGroupBox, 
    QFormLayout, QMessageBox, QComboBox, QTextEdit, QProgressBar,
    QDialog, QSlider, QStyle, QFileDialog, QFrame, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette

# Import our modular scripts
from scripts.whisper_script import generate_srt
from scripts.translation_script import translate_srt
from scripts.transliteration_script import run_transliteration
from scripts.trim_video import trim_video, get_video_duration
from scripts.utils import parse_srt

import subprocess

class SubprocessWorker(QThread):
    """
    Runs an isolated python script in a distinct child process to prevent 
    deep C-level memory access violations (like ctranslate2/VAD filtering bugs)
    from crashing the main PyQt UI thread. Streams stdout continuously.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, script_path, args_list):
        super().__init__()
        self.script_path = script_path
        self.args_list = args_list

    def run(self):
        import re
        start_time = time.time()
        
        cmd = [sys.executable, "-u", self.script_path] + self.args_list
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self._is_running = True
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Stream output in real-time
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    clean_line = line.strip()
                    self.log_signal.emit(clean_line)
                    
                    # Parse numerical progress if formatted like "[45.2%]"
                    match = re.search(r'\[([\d\.]+)%\]', clean_line)
                    if match:
                        try:
                            self.progress_signal.emit(int(float(match.group(1))))
                        except ValueError:
                            pass
            
            self.process.stdout.close()
            self.process.wait()
            
            if not self._is_running:
                return # Thread was manually forcefully terminated
            
            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            time_str = f"Execution Time: {int(mins)}m {secs:.1f}s"
            
            if self.process.returncode == 0:
                self.finished_signal.emit(True, f"Task completed successfully. ({time_str})")
            else:
                self.finished_signal.emit(False, f"Sub-process exited with code {self.process.returncode}. ({time_str})")
        except Exception as e:
            if self._is_running:
                self.finished_signal.emit(False, f"Thread failed with Exception:\\n{e}")

    def stop(self):
        self._is_running = False
        if hasattr(self, 'process') and self.process.poll() is None:
            import psutil
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except Exception:
                self.process.kill()
        self.terminate()
        self.wait()

class VisualTrimmerDialog(QDialog):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visual Video Trimmer")
        self.setGeometry(150, 150, 800, 600)
        self.video_path = video_path
        
        self.in_point_ms = 0
        self.out_point_ms = 0
        self.duration_ms = 0
        
        self.init_ui()
        self.load_video()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Video Player Area
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget, stretch=1)

        # Controls Area
        control_layout = QHBoxLayout()
        
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        
        self.lbl_time = QLabel("00:00:00 / 00:00:00")
        
        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.lbl_time)
        layout.addLayout(control_layout)

        # Trimming Points
        trim_layout = QHBoxLayout()
        self.btn_set_in = QPushButton("Set In Point [ { ]")
        self.btn_set_out = QPushButton("Set Out Point [ } ]")
        self.lbl_in = QLabel("In: 0.0s")
        self.lbl_out = QLabel("Out: End")
        
        self.btn_set_in.clicked.connect(self.set_in_point)
        self.btn_set_out.clicked.connect(self.set_out_point)
        
        trim_layout.addWidget(self.btn_set_in)
        trim_layout.addWidget(self.lbl_in)
        trim_layout.addWidget(self.btn_set_out)
        trim_layout.addWidget(self.lbl_out)
        layout.addLayout(trim_layout)

        # Action Buttons
        action_layout = QHBoxLayout()
        self.btn_apply = QPushButton("Trim & Replace Video")
        self.btn_apply.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.btn_cancel = QPushButton("Cancel")
        
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        
        action_layout.addStretch()
        action_layout.addWidget(self.btn_cancel)
        action_layout.addWidget(self.btn_apply)
        layout.addLayout(action_layout)

        # MediaPlayer Setup
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

    def load_video(self):
        self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
        self.media_player.play()
        self.media_player.pause()

    def toggle_play(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self.media_player.play()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def position_changed(self, position):
        self.slider.setValue(position)
        self.update_time_label()

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.duration_ms = duration
        self.out_point_ms = duration
        self.update_time_label()
        self.lbl_out.setText(f"Out: {duration/1000:.1f}s")

    def set_position(self, position):
        self.media_player.setPosition(position)

    def update_time_label(self):
        pos_s = self.media_player.position() // 1000
        dur_s = self.duration_ms // 1000
        self.lbl_time.setText(f"{pos_s//3600:02d}:{(pos_s%3600)//60:02d}:{pos_s%60:02d} / {dur_s//3600:02d}:{(dur_s%3600)//60:02d}:{dur_s%60:02d}")

    def set_in_point(self):
        self.in_point_ms = self.media_player.position()
        self.lbl_in.setText(f"In: {self.in_point_ms/1000:.1f}s")

    def set_out_point(self):
        self.out_point_ms = self.media_player.position()
        self.lbl_out.setText(f"Out: {self.out_point_ms/1000:.1f}s")

    def get_trim_points_sec(self):
        """Returns the start and end trim points in seconds."""
        return self.in_point_ms / 1000.0, self.out_point_ms / 1000.0

    def closeEvent(self, event):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        self.media_player.setSource(QUrl()) # Free the video file handle
        super().closeEvent(event)


class VideoGenieApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Genie Pipeline")
        self.setGeometry(100, 100, 900, 700)
        
        # Set Window Icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.cwd = os.getcwd()
        self.selected_video = None
        
        self.init_ui()
        self.refresh_video_list()

    def init_ui(self):
        # Apply standard application font globally
        app_font = QFont("Inter", 10)
        app_font.setStyleHint(QFont.StyleHint.SansSerif)
        QApplication.setFont(app_font)

        # Main Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Super clean background wrapper
        central_widget.setStyleSheet("QWidget#CentralBackground { background-color: #F9FAFB; }")
        central_widget.setObjectName("CentralBackground")
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(24)
        
        top_layout = QHBoxLayout()
        top_layout.setSpacing(24)

        # Helper to create styled white cards
        def create_card(title=""):
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #FFFFFF;
                    border: 1px solid #E5E7EB;
                    border-radius: 8px;
                }
            """)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            if title:
                lbl_title = QLabel(title)
                lbl_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #111827; border: none;")
                layout.addWidget(lbl_title)
            return card, layout

        # =============================================
        # 1. Left Panel (Navigation / Explorer)
        # =============================================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(16)
        
        lbl_videos = QLabel("📁 Workspace Videos")
        lbl_videos.setStyleSheet("font-size: 14px; font-weight: 600; color: #111827; margin-bottom: 4px;")
        
        self.video_list = QListWidget()
        self.video_list.setStyleSheet("""
            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
                outline: 0;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 6px;
                color: #374151;
            }
            QListWidget::item:selected {
                background-color: #EEF2FF;
                color: #4F46E5;
                font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #F3F4F6;
            }
        """)
        self.video_list.itemSelectionChanged.connect(self.on_video_selected)
        self.video_list.itemDoubleClicked.connect(self.open_visual_trimmer)
        
        left_panel.addWidget(lbl_videos)
        left_panel.addWidget(self.video_list)
        
        # File Action Buttons at bottom
        file_actions_layout = QHBoxLayout()
        self.btn_import = QPushButton("➕ Import Video")
        self.btn_import.setStyleSheet("background-color: #FFFFFF; color: #374151; border: 1px dashed #D1D5DB; border-radius: 6px; padding: 10px; font-weight: 600;")
        self.btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import.clicked.connect(self.import_video)
        
        self.btn_delete = QPushButton("🗑️ Delete")
        self.btn_delete.setStyleSheet("background-color: #FEF2F2; color: #EF4444; border: 1px solid #FCA5A5; border-radius: 6px; padding: 10px; font-weight: 600;")
        self.btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete.clicked.connect(self.delete_video)
        self.btn_delete.setEnabled(False) 
        
        file_actions_layout.addWidget(self.btn_import)
        file_actions_layout.addWidget(self.btn_delete)
        left_panel.addLayout(file_actions_layout)

        # =============================================
        # 2. Middle Panel (Actions & Data)
        # =============================================
        middle_panel = QVBoxLayout()
        middle_panel.setSpacing(16)
        
        # Document Header
        self.lbl_selected_video = QLabel("🎬 Select a video to begin")
        self.lbl_selected_video.setStyleSheet("font-size: 18px; font-weight: 700; color: #111827; margin-bottom: 4px;")
        middle_panel.addWidget(self.lbl_selected_video)

        # Card: File Statistics
        stats_card, stats_layout = create_card("File Status")
        
        self.lbl_stat_duration = QLabel("Duration: \t\tN/A")
        self.lbl_stat_duration.setStyleSheet("font-size: 13px; color: #4B5563; border: none; padding: 2px;")
        stats_layout.addWidget(self.lbl_stat_duration)
        
        lbl_srt_header = QLabel("Generated SRT Files:")
        lbl_srt_header.setStyleSheet("font-size: 13px; color: #4B5563; border: none; padding: 2px; font-weight: bold; margin-top: 8px;")
        stats_layout.addWidget(lbl_srt_header)

        self.srt_list_layout = QVBoxLayout()
        self.srt_list_layout.setSpacing(4)
        stats_layout.addLayout(self.srt_list_layout)
        
        middle_panel.addWidget(stats_card)

        # Card: Video Tools
        tools_card, tools_layout = create_card("Video Tools")
        self.btn_trim_visual = QPushButton("✂️ Open Visual Trimmer")
        self.btn_trim_visual.setStyleSheet("background-color: #E5E7EB; color: #1F2937; border-radius: 6px; padding: 12px; font-weight: 600;")
        self.btn_trim_visual.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_trim_visual.clicked.connect(self.open_visual_trimmer)
        tools_layout.addWidget(self.btn_trim_visual)
        middle_panel.addWidget(tools_card)

        # Card: AI Pipeline
        pipeline_card, pipe_layout = create_card("AI Subtitle Pipeline")
        
        btn_style_primary = "background-color: #4F46E5; color: white; border-radius: 6px; padding: 12px; font-weight: 600;"
        btn_style_secondary = "background-color: #E5E7EB; color: #1F2937; border-radius: 6px; padding: 12px; font-weight: 600;"
        combo_style = """
            QComboBox { border: 1px solid #D1D5DB; border-radius: 4px; padding: 6px; background: white; color: #111827; }
            QComboBox QAbstractItemView { background: white; color: #111827; selection-background-color: #EEF2FF; selection-color: #4F46E5; }
        """
        
        # Whisper Row
        w_layout = QHBoxLayout()
        w_layout.setSpacing(12)
        self.btn_generate_srt = QPushButton("🎙️ Generate Subtitles")
        self.btn_generate_srt.setStyleSheet(btn_style_primary)
        self.btn_generate_srt.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_generate_srt.clicked.connect(self.start_whisper)
        
        self.combo_model = QComboBox()
        self.combo_model.setStyleSheet(combo_style)
        self.combo_model.setToolTip("Select Whisper Model Size")
        self.combo_model.addItems(["base", "small", "medium", "large-v2", "large-v3"])
        
        self.combo_device = QComboBox()
        self.combo_device.setStyleSheet(combo_style)
        self.combo_device.setToolTip("Select Processing Device")
        self.combo_device.addItems(["cpu", "cuda"])

        self.combo_whisper_lang = QComboBox()
        self.combo_whisper_lang.setStyleSheet(combo_style)
        self.combo_whisper_lang.setToolTip("Select Source Language")
        self.combo_whisper_lang.addItems(["Auto-Detect", "Arabic", "English", "Hindi", "Urdu", "German"])
        
        self.combo_compute = QComboBox()
        self.combo_compute.setStyleSheet(combo_style)
        self.combo_compute.setToolTip("Select Compute Precision")
        self.combo_compute.addItems(["float32", "float16", "int8"])
        
        w_layout.addWidget(self.btn_generate_srt, stretch=3)
        w_layout.addWidget(self.combo_whisper_lang, stretch=2)
        w_layout.addWidget(self.combo_model, stretch=1)
        w_layout.addWidget(self.combo_device, stretch=1)
        w_layout.addWidget(self.combo_compute, stretch=1)
        pipe_layout.addLayout(w_layout)

        # Advanced Whisper settings row
        w_adv_layout = QHBoxLayout()
        w_adv_layout.setSpacing(12)
        
        self.combo_timestamps = QComboBox()
        self.combo_timestamps.setStyleSheet(combo_style)
        self.combo_timestamps.setToolTip("Timestamp Granularity")
        self.combo_timestamps.addItems(["Sentence-Level", "Word-Level"])
        
        self.chk_vad = QCheckBox(" VAD Filter")
        self.chk_vad.setChecked(True)
        self.chk_vad.setStyleSheet("QCheckBox { color: #111827; font-weight: bold; }")
        self.chk_vad.setToolTip("Remove silent sections organically")
        
        self.chk_condition = QCheckBox(" Cond. on Prev")
        self.chk_condition.setChecked(True)
        self.chk_condition.setStyleSheet("QCheckBox { color: #111827; font-weight: bold; }")
        self.chk_condition.setToolTip("Helpful for context, turn off if it loops.")
        
        self.chk_no_punct = QCheckBox(" No Punct.")
        self.chk_no_punct.setChecked(False)
        self.chk_no_punct.setStyleSheet("QCheckBox { color: #111827; font-weight: bold; }")
        self.chk_no_punct.setToolTip("Force-strip punctuation (.,?!) from final subtitles.")
        
        # Beam Size
        lbl_beam = QLabel("Beam:")
        lbl_beam.setStyleSheet("color: #111827; font-weight: bold;")
        self.spin_beam = QSpinBox()
        self.spin_beam.setRange(1, 15)
        self.spin_beam.setValue(5)
        self.spin_beam.setStyleSheet("QSpinBox { border: 1px solid #D1D5DB; border-radius: 4px; padding: 4px; background: white; color: #111827; }")

        w_adv_layout.addWidget(self.combo_timestamps)
        w_adv_layout.addWidget(self.chk_vad)
        w_adv_layout.addWidget(self.chk_condition)
        w_adv_layout.addWidget(self.chk_no_punct)
        w_adv_layout.addWidget(lbl_beam)
        w_adv_layout.addWidget(self.spin_beam)
        pipe_layout.addLayout(w_adv_layout)

        # Prompt Row
        prompt_layout = QHBoxLayout()
        prompt_layout.setSpacing(12)
        self.line_prompt = QLineEdit()
        self.line_prompt.setPlaceholderText("Optional: Initial Prompt (e.g., proper nouns, slang...)")
        self.line_prompt.setText("بسم اللہ الرحمن الرحیم الحمد للہ آج ہم قرآن حدیث اور سنت رسول ﷺ کے بارے میں بات کریں گے اللہ ﷻ کے فضل سے تمام صحابہ کرام ؓ اور اجمعین کا ذکر ہوگا")
        self.line_prompt.setStyleSheet("QLineEdit { border: 1px solid #D1D5DB; border-radius: 4px; padding: 6px; background: white; color: #111827; }")
        
        prompt_layout.addWidget(self.line_prompt)
        pipe_layout.addLayout(prompt_layout)

        # VAD Control Row
        vad_layout = QHBoxLayout()
        vad_layout.setSpacing(12)
        
        self.lbl_vad_slider = QLabel("VAD Silence Split (2000 ms)")
        self.lbl_vad_slider.setStyleSheet("color: #111827; font-weight: bold;")
        self.lbl_vad_slider.setToolTip("Lower to split long sentences on tiny breaths. Default is 2000ms.")
        
        self.slider_vad = QSlider(Qt.Orientation.Horizontal)
        self.slider_vad.setRange(500, 3000)
        self.slider_vad.setSingleStep(500)
        self.slider_vad.setPageStep(500)
        self.slider_vad.setValue(2000)
        self.slider_vad.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_vad.setTickInterval(500)
        self.slider_vad.setStyleSheet("QSlider::handle:horizontal { background: #4F46E5; border-radius: 6px; width: 14px; margin: -4px 0; }")
        self.slider_vad.valueChanged.connect(lambda val: self.lbl_vad_slider.setText(f"VAD Silence Split ({val} ms)"))
        
        vad_layout.addWidget(self.lbl_vad_slider)
        vad_layout.addWidget(self.slider_vad)
        pipe_layout.addLayout(vad_layout)

        # Subtle divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #E5E7EB; margin-top: 8px; margin-bottom: 8px;")
        pipe_layout.addWidget(divider)

        # Translation Row
        trans_layout = QHBoxLayout()
        trans_layout.setSpacing(12)
        
        self.btn_translate = QPushButton("🌍 Translate")
        self.btn_translate.setStyleSheet(btn_style_secondary)
        self.btn_translate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_translate.clicked.connect(self.start_translation)
        
        self.combo_trans_model = QComboBox()
        self.combo_trans_model.setStyleSheet(combo_style)
        self.combo_trans_model.addItems(["facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B", "facebook/nllb-200-3.3B"])
        self.combo_trans_model.setCurrentText("facebook/nllb-200-3.3B")
        
        self.combo_lang = QComboBox()
        self.combo_lang.setStyleSheet(combo_style)
        self.combo_lang.addItems(["en", "hi", "ur"])
        
        trans_layout.addWidget(self.btn_translate, stretch=2)
        trans_layout.addWidget(self.combo_trans_model, stretch=3)
        trans_layout.addWidget(self.combo_lang, stretch=1)
        pipe_layout.addLayout(trans_layout)

        # Transliteration Row
        translit_layout = QHBoxLayout()
        translit_layout.setSpacing(12)
        
        self.btn_transliterate = QPushButton("🔤 Romanize")
        self.btn_transliterate.setStyleSheet(btn_style_secondary)
        self.btn_transliterate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_transliterate.clicked.connect(self.start_transliteration)
        
        self.combo_translit_srt = QComboBox()
        self.combo_translit_srt.setStyleSheet(combo_style)
        self.combo_translit_srt.setToolTip("Select SRT file to Romanize")
        
        self.combo_translit_lang = QComboBox()
        self.combo_translit_lang.setStyleSheet(combo_style)
        self.combo_translit_lang.setToolTip("Select language of the original SRT")
        self.combo_translit_lang.addItems(["hi", "ur"])
        
        translit_layout.addWidget(self.btn_transliterate, stretch=2)
        translit_layout.addWidget(self.combo_translit_srt, stretch=4)
        translit_layout.addWidget(self.combo_translit_lang, stretch=1)
        
        pipe_layout.addLayout(translit_layout)
        middle_panel.addWidget(pipeline_card)
        
        middle_panel.addStretch(1) # Push everything up cleanly

        # =============================================
        # 3. Right Panel (Console Layout)
        # =============================================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)
        
        lbl_console = QLabel("📜 Console Output")
        lbl_console.setStyleSheet("font-size: 13px; font-weight: 600; color: #111827; margin-bottom: 4px;")
        right_panel.addWidget(lbl_console)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #0F172A; 
                color: #E2E8F0; 
                font-family: Consolas, monospace; 
                font-size: 12px;
                border-radius: 8px; 
                padding: 12px;
            }
        """)
        right_panel.addWidget(self.console_output)
        
        # Progress Bar for tasks
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                text-align: center;
                background-color: #F3F4F6;
                color: #1F2937;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4F46E5;
                border-radius: 5px;
            }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.hide() # Hidden until active
        right_panel.addWidget(self.progress_bar)

        # Assemble Top layout (2-Panel Width Proportions)
        top_layout.addLayout(left_panel, stretch=2)
        top_layout.addLayout(middle_panel, stretch=5)
        
        # Assemble main layout (Top/Bottom)
        main_layout.addLayout(top_layout, stretch=3)
        main_layout.addLayout(right_panel, stretch=1)
        
        # Add basic status bar
        self.statusBar().showMessage("Ready")

    def log(self, text):
        if text.strip():
            self.console_output.append(text.strip())
            scrollbar = self.console_output.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def set_buttons_enabled(self, enabled: bool):
        self.btn_import.setEnabled(enabled) if hasattr(self, 'btn_import') else None
        
        if self.selected_video:
            self.btn_delete.setEnabled(enabled)
            
        self.btn_trim_visual.setEnabled(enabled)
        self.btn_generate_srt.setEnabled(enabled)
        self.btn_translate.setEnabled(enabled)
        self.btn_transliterate.setEnabled(enabled)
        
        if enabled:
            self.progress_bar.hide()
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.show()
            self.progress_bar.setRange(0, 0) # Indeterminate spinning

    def import_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video to Import", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if file_path:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.cwd, filename)
            
            if os.path.exists(dest_path):
                QMessageBox.warning(self, "Conflict", f"A video named '{filename}' already exists in the folder.")
                return
                
            try:
                shutil.copy2(file_path, dest_path)
                self.log(f"✅ Successfully imported: {filename}")
                self.refresh_video_list()
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import video:\n{e}")

    def delete_video(self):
        if not self.selected_video:
            return
            
        reply = QMessageBox.question(self, 'Confirm Deletion', 
                     f"Are you sure you want to completely delete '{self.selected_video}' and all of its generated SRT files from the working directory?",
                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Delete video
                video_path = os.path.join(self.cwd, self.selected_video)
                if os.path.exists(video_path):
                    os.remove(video_path)
                
                # Delete associated working files
                base_name = os.path.splitext(self.selected_video)[0]
                work_dir = os.path.join(self.cwd, "working")
                if os.path.exists(work_dir):
                    for file in os.listdir(work_dir):
                        if file.startswith(base_name) and file.endswith(".srt"):
                            os.remove(os.path.join(work_dir, file))
                            
                self.log(f"🗑️ Deleted video and associated cache for: {self.selected_video}")
                self.selected_video = None
                self.lbl_selected_video.setText("No video selected")
                self.btn_delete.setEnabled(False)
                self.update_stats() # reset UI
                self.refresh_video_list()
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Failed to delete files:\n{e}")

    def delete_specific_srt(self, filename):
        reply = QMessageBox.question(self, 'Confirm Delete', 
                     f"Are you sure you want to delete '{filename}'?",
                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                work_dir = os.path.join(self.cwd, "working")
                file_path = os.path.join(work_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                self.log(f"🗑️ Deleted SRT: {filename}")
                self.update_stats()
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Failed to delete SRT:\n{e}")

    def refresh_video_list(self):
        self.video_list.clear()
        video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
        for file in os.listdir(self.cwd):
            if file.lower().endswith(video_extensions):
                self.video_list.addItem(f"▶ {file}")
                
        if self.video_list.count() == 0:
            self.lbl_selected_video.setText("🎬 Select a video to begin")

    def on_video_selected(self):
        items = self.video_list.selectedItems()
        if not items:
            return
            
        # Clean the play icon when saving name
        self.selected_video = items[0].text().replace("▶ ", "")
        self.lbl_selected_video.setText(f"🎬 {self.selected_video}")
        self.btn_delete.setEnabled(True)
        self.update_stats()

    def update_stats(self):
        if not self.selected_video:
            return
            
        video_path = os.path.join(self.cwd, self.selected_video)
        base_name = os.path.splitext(self.selected_video)[0]
        work_dir = os.path.join(self.cwd, "working")
        
        # Set Duration
        dur = get_video_duration(video_path)
        if dur:
            self.lbl_stat_duration.setText(f"Duration: \t\t{dur/60:.2f} mins ({dur:.0f}s)")
        else:
            self.lbl_stat_duration.setText("Duration: \t\tUnknown")

        # Helper to clear old items safely
        def clear_layout(layout):
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                    else:
                        sub_layout = item.layout()
                        clear_layout(sub_layout)
                        
        clear_layout(self.srt_list_layout)

        if not os.path.exists(work_dir):
            lbl_none = QLabel("⚪ None")
            lbl_none.setStyleSheet("font-size: 13px; color: #6B7280; border: none; padding: 2px; margin-left: 8px;")
            self.srt_list_layout.addWidget(lbl_none)
            return

        available_srt_files = []
        found_srts = False
        
        for file in os.listdir(work_dir):
            if file.startswith(base_name) and file.endswith(".srt"):
                found_srts = True
                if not file.endswith("roman.srt"):
                     available_srt_files.append(file)
                     
                file_row = QHBoxLayout()
                lbl_file = QLabel(file)
                lbl_file.setStyleSheet("font-size: 13px; color: #059669; font-weight: bold; border: none; padding: 0px;")
                
                btn_del = QPushButton("❌")
                btn_del.setFixedSize(22, 22)
                btn_del.setStyleSheet("""
                    QPushButton { background-color: transparent; border: none; font-size: 12px; }
                    QPushButton:hover { background-color: #FEE2E2; border-radius: 4px; }
                """)
                btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
                btn_del.clicked.connect(lambda checked, f=file: self.delete_specific_srt(f))
                
                file_row.addWidget(lbl_file)
                file_row.addStretch()
                file_row.addWidget(btn_del)
                
                row_widget = QWidget()
                row_widget.setLayout(file_row)
                row_widget.setStyleSheet("background-color: transparent;")
                self.srt_list_layout.addWidget(row_widget)
                
        if not found_srts:
            lbl_none = QLabel("⚪ None")
            lbl_none.setStyleSheet("font-size: 13px; color: #6B7280; border: none; padding: 2px; margin-left: 8px;")
            self.srt_list_layout.addWidget(lbl_none)
                    
        if hasattr(self, 'combo_translit_srt'):
            self.combo_translit_srt.clear()
            self.combo_translit_srt.addItems(available_srt_files)

    # ----------------------------------------------------
    # Background Task Handlers
    # ----------------------------------------------------
    def _run_script_subprocess(self, task_name, script_name, *args):
        if not self.selected_video:
            QMessageBox.warning(self, "Error", "Please select a video first.")
            return

        self.console_output.clear()
        self.log(f"--- Starting {task_name} ---")
        self.set_buttons_enabled(False)
        self.statusBar().showMessage(f"Running: {task_name}...")
        
        self.progress_bar.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        script_path = os.path.join(self.cwd, "scripts", script_name)
        args_list = [str(a) for a in args]

        self.worker = SubprocessWorker(script_path, args_list)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self._on_task_finished)
        self.worker.start()

    def _on_task_finished(self, success, message):
        self.set_buttons_enabled(True)
        self.update_stats()
        self.statusBar().showMessage("Ready")
        
        # Hide progress bar when done
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        
        self.log(f"\n--- Output: {message} ---")
        if not success:
            QMessageBox.critical(self, "Task Error", message)

    def open_visual_trimmer(self):
        if not self.selected_video:
            QMessageBox.warning(self, "Error", "Please select a video first.")
            return

        video_path = os.path.join(self.cwd, self.selected_video)
        dialog = VisualTrimmerDialog(video_path, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            start_sec, end_sec = dialog.get_trim_points_sec()
            
            # Use the new trim_video_exact function
            from scripts.trim_video import trim_video_exact
            
            reply = QMessageBox.question(self, 'Confirm Overwrite', 
                         f"This will overwrite the original video file.\nTrimming to {start_sec:.1f}s -> {end_sec:.1f}s. Are you sure?",
                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self._run_script_subprocess("Visual Video Trimming", "trim_video.py", "exact", video_path, start_sec, end_sec)

    def start_whisper(self):
        if not self.selected_video:
            QMessageBox.warning(self, "Error", "Please select a video first.")
            return
            
        video_path = os.path.join(self.cwd, self.selected_video)
        model = self.combo_model.currentText()
        device = self.combo_device.currentText()
        compute = self.combo_compute.currentText()
        
        lang_selection = self.combo_whisper_lang.currentText()
        lang_map = {
            "Auto-Detect": "auto",
            "Arabic": "ar",
            "English": "en",
            "Hindi": "hi",
            "Urdu": "ur",
            "German": "de"
        }
        iso_lang = lang_map.get(lang_selection, "auto")
        
        args = [video_path, "--model", model, "--device", device, "--language", iso_lang, "--compute_type", compute]
        
        if self.combo_timestamps.currentText() == "Word-Level":
            args.append("--word_timestamps")
            
        if not self.chk_vad.isChecked():
            args.append("--no_vad")
            
        if not self.chk_condition.isChecked():
            args.append("--no_condition")
            
        if self.chk_no_punct.isChecked():
            args.append("--no_punctuation")
            
        args.extend(["--beam_size", str(self.spin_beam.value())])
        args.extend(["--vad_min_silence_ms", str(self.slider_vad.value())])
            
        opt_prompt = self.line_prompt.text().strip()
        if opt_prompt:
            args.extend(["--initial_prompt", opt_prompt])
            
        self._run_script_subprocess("Whisper Generation", "whisper_script.py", *args)

    def start_translation(self):
        base_name = os.path.splitext(self.selected_video)[0]
        srt_file = os.path.join(self.cwd, "working", f"{base_name}.srt")
        lang = self.combo_lang.currentText()
        model_name = self.combo_trans_model.currentText()
        
        if not os.path.exists(srt_file):
            QMessageBox.warning(self, "Error", "Original SRT not found. Generate it first.")
            return
            
        self._run_script_subprocess("Translation", "translation_script.py", srt_file, "--lang", lang, "--model", model_name)

    def start_transliteration(self):
        srt_filename = self.combo_translit_srt.currentText()
        if not srt_filename:
            QMessageBox.warning(self, "Error", "No SRT file selected for transliteration.")
            return
            
        srt_file_path = os.path.join(self.cwd, "working", srt_filename)
        lang = self.combo_translit_lang.currentText()
            
        self._run_script_subprocess("Transliteration", "transliteration_script.py", srt_file_path, "--lang", lang)

    def closeEvent(self, event):
        # Gracefully handle running threads when the main app closes
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    
    # Fallback to standard Fusion and inject global background explicitly
    app.setStyle("Fusion")
    
    # Custom light theme palette to replicate the modern look without external dependencies
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#F9FAFB"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#F3F4F6"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4F46E5"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    
    window = VideoGenieApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
