# 🧞 Video Genie

![Video Genie Banner](C:\Users\conta\.gemini\antigravity\brain\da1fb44e-4eb2-45d5-8128-1f36e0df01b0\video_genie_banner_1773337495163.png)

**Video Genie** is an advanced, AI-powered video localization pipeline designed to take raw video files and turn them into fully subtitled, translated, and transliterated content. Built with a focus on high-accuracy transcription (Whisper) and near-human translation (NLLB), it provides a professional-grade tool for creators targeting multilingual audiences.

---

## 🚀 Key Features

### 🎙️ AI Subtitle Generation
- **Faster-Whisper Backend**: High-performance transcription using CTranslate2.
- **Advanced Control**: Choose model sizes from `base` to `large-v3`.
- **VAD Filtering**: Integrated Voice Activity Detection to remove background noise and eliminate hallucinations during silence.
- **Compute Precision**: Support for `float32`, `float16`, and `int8` for optimal performance on CPU or NVIDIA GPUs (CUDA).

### 🌍 Intelligent Translation
- **NLLB-200 (No Language Left Behind)**: Uses Facebook's state-of-the-art 3.3B model for high-fidelity grammar mapping.
- **Multilingual Support**: Optimized for English, Hindi, and Urdu (extendable to 200+ languages).
- **Auto-Detection**: Scans existing SRT files to detect source language before translation.

### 🔤 Smart Transliteration (Romanization)
- **Offline-First Caching**: Utilizes a pre-built dictionary of ~150,000 words for instant offline Romanization.
- **OpenAI Hybrid Logic**: Intelligently detects "cache misses" and performs bulk, context-aware API calls using GPT-4o-mini to fill gaps in the local dictionary.

### ✂️ Visual Video Tools
- **Precision Trimmer**: Frame-accurate visual scrubbing to set In/Out points.
- **Fast Processing**: Uses FFmpeg stream copying (`-c copy`) for near-instant trimming without re-encoding quality loss.

---

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg**: Must be installed and added to your system `PATH`.
- **NVIDIA GPU (Optional)**: For CUDA acceleration, ensures you have the NVIDIA Toolkit 12.x installed.

### 2. Dependencies
```bash
pip install PyQt6 faster-whisper transformers torch sentencepiece langdetect openai psutil
```

### 3. Setup Secret Keys (Optional)
For the Romanization feature to use AI fallback, place your OpenAI API key in:
`secret/openai_api_key.json`
```json
{
  "OPENAI_API_KEY": "your-key-here"
}
```

---

## 💻 Technical Architecture

Video Genie uses an **Isolated Subprocess Model**. 

AI tasks (Whisper/NLLB) are executed in distinct OS-level child processes rather than pure threads. This architecture completely prevents the main PyQt GUI from hanging or crashing due to deep C-level memory access violations (common in ML libraries like CTranslate2 or Torch), ensuring a smooth, reliable user experience.

---

## 🏗️ Tech Stack

- **GUI**: PyQt6 (Python)
- **ASR**: Faster-Whisper (OpenAI Whisper via CTranslate2)
- **Translation**: Meta NLLB-200
- **Transliteration**: uroman / OpenAI GPT-4o-mini
- **Video Engine**: FFmpeg / ffprobe
- **Thread Management**: Psutil / Subprocess Pipes

---

## 📝 License
This project is for private research and development in AI video localization. 

---

*Wishes granted, one subtitle at a time.*
