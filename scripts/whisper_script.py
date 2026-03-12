import os
import sys
import argparse

# Inject CUDA 12 PyPI package paths for CTranslate2 to locate cublas64_12.dll
try:
    import site
    packages_dir = site.getusersitepackages()
    cublas_bin = os.path.join(packages_dir, "nvidia", "cublas", "bin")
    cudnn_bin = os.path.join(packages_dir, "nvidia", "cudnn", "bin")
    
    for bin_dir in [cublas_bin, cudnn_bin]:
        if os.path.exists(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(bin_dir)
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils import setup_threading, format_timestamp

def generate_srt(video_file: str, model_size: str = "base", device: str = "cpu", language: str = None, 
                 compute_type: str = "float32", word_timestamps: bool = False, vad_filter: bool = True, 
                 initial_prompt: str = "", beam_size: int = 5, condition_on_previous_text: bool = True, 
                 vad_min_silence_ms: int = 2000, work_dir: str = "working", no_punctuation: bool = False) -> str:
    """
    Generates an SRT file from a video using faster-whisper.
    Returns the path to the expected SRT file, and a boolean indicating if it was newly created.
    """
    setup_threading()

    if not os.path.exists(video_file):
        print(f"Error: File '{video_file}' not found.")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(video_file))[0]
    os.makedirs(work_dir, exist_ok=True)
    srt_file = os.path.join(work_dir, f"{base_name}.srt")

    if not os.path.exists(srt_file):
        print(f"SRT does not exist. Creating original SRT from video...")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Please install faster-whisper.")
            sys.exit(1)

        print(f"Loading Whisper model '{model_size}' on '{device}' with precision '{compute_type}'...")
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=7)
        except Exception as e:
            print(f"Failed to load Whisper model... ({e})")
            sys.exit(1)

        try:
            # Prepare kwargs for transcribe
            transcribe_kwargs = {
                "beam_size": beam_size,
                "word_timestamps": word_timestamps,
                "vad_filter": vad_filter,
                "language": language,
                "condition_on_previous_text": condition_on_previous_text
            }
            if vad_filter:
                transcribe_kwargs["vad_parameters"] = dict(min_silence_duration_ms=vad_min_silence_ms)
                
            if initial_prompt.strip():
                transcribe_kwargs["initial_prompt"] = initial_prompt.strip()

            segments, info = model.transcribe(video_file, **transcribe_kwargs)
        except RuntimeError as e:
            if "cublas64_12.dll" in str(e):
                print("\n[ERROR] Missing NVIDIA CUDA 12 Toolkit!")
                print("Your GPU does not have the CUDA 12 'cublas64_12.dll' installed.")
                print("Please either:")
                print("  1. Select 'cpu' from the Device dropdown instead.")
                print("  2. Install the CUDA Toolkit 12.x from NVIDIA's website.")
                sys.exit(1)
            else:
                raise e
        total_duration = info.duration

        def strip_punct(text):
            if not no_punctuation:
                return text
            # Define common and language specific punctuation to remove
            punct_chars = '.,!?;:()[]{}"\'-_۔،؟'
            for char in punct_chars:
                text = text.replace(char, '')
            return text.strip()

        with open(srt_file, "w", encoding="utf-8") as f:
            item_index = 1
            for segment in segments:
                if word_timestamps and hasattr(segment, 'words') and getattr(segment, 'words', None):
                    for word in segment.words:
                        start_time = format_timestamp(word.start)
                        end_time = format_timestamp(word.end)
                        text = word.word.strip()
                        if text:
                            f.write(f"{item_index}\n{start_time} --> {end_time}\n{text}\n\n")
                            item_index += 1
                else:
                    start_time = format_timestamp(segment.start)
                    end_time = format_timestamp(segment.end)
                    text = strip_punct(segment.text)
                    if text:
                        f.write(f"{item_index}\n{start_time} --> {end_time}\n{text}\n\n")
                        item_index += 1

                percent_complete = (segment.end / total_duration) * 100 if total_duration > 0 else 0
                print(f"Progress: [{percent_complete:.1f}%] - {format_timestamp(segment.start)} --> {format_timestamp(segment.end)} | {segment.text.strip()[:50]}...", end="\r", flush=True)

        print(f"\n✅ Successfully created '{srt_file}'. Exiting pipeline.")
        return srt_file, True
    else:
        print(f"✅ Found existing SRT: '{srt_file}'")
        return srt_file, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SRT from a video using Whisper.")
    parser.add_argument("video_file", help="Input video or audio file path")
    parser.add_argument("--model", default="base", help="Whisper model size.")
    parser.add_argument("--device", default="cpu", help="Compute device: cpu or cuda.")
    parser.add_argument("--language", default="auto", help="ISO language code to force transcription to.")
    parser.add_argument("--compute_type", default="float32", help="Compute type for model (float16, int8, float32).")
    parser.add_argument("--word_timestamps", action="store_true", help="Generate word-level timestamps instead of sentence-level.")
    parser.add_argument("--no_vad", action="store_true", help="Disable VAD filter.")
    parser.add_argument("--no_condition", action="store_true", help="Disable condition on previous text.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for transcription.")
    parser.add_argument("--vad_min_silence_ms", type=int, default=2000, help="VAD split silence threshold in ms.")
    parser.add_argument("--initial_prompt", default="", help="Initial prompt to help whisper recognize terms.")
    parser.add_argument("--no_punctuation", action="store_true", help="Force-strip punctuation from output.")
    
    args = parser.parse_args()
    iso_lang = None if args.language == "auto" else args.language
    vad_active = not args.no_vad
    condition_active = not args.no_condition
    
    created_srt, is_new = generate_srt(
        args.video_file, 
        model_size=args.model, 
        device=args.device, 
        language=iso_lang,
        compute_type=args.compute_type,
        word_timestamps=args.word_timestamps,
        vad_filter=vad_active,
        initial_prompt=args.initial_prompt,
        beam_size=args.beam_size,
        condition_on_previous_text=condition_active,
        vad_min_silence_ms=args.vad_min_silence_ms,
        no_punctuation=args.no_punctuation
    )
    if is_new:
        sys.exit(0)
