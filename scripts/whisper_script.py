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
from scripts.utils import (
    setup_threading, format_timestamp, estimate_model_gb, select_device,
    get_gpu_compute_capability,
)

# Minimum CUDA compute capability CTranslate2 requires per compute type.
CT2_MIN_COMPUTE_CAPABILITY = {
    "int8": (6, 1),
    "int8_float32": (6, 1),
    "int8_float16": (7, 0),
    "float16": (7, 0),
    "bfloat16": (8, 0),
    "float32": (3, 5),
}

# The prebuilt CTranslate2 wheels only ship CUDA kernels for Volta (7.0) and newer.
# Older cards load a model on the GPU without error, then die at inference with
# "cudaErrorInvalidDevice: invalid device ordinal", so they must be excluded up front.
CT2_MIN_CUDA_COMPUTE_CAPABILITY = (7, 0)

# Approximate parameter counts (in millions) for Whisper model sizes.
WHISPER_PARAMS_M = {
    "tiny": 39, "tiny.en": 39,
    "base": 74, "base.en": 74,
    "small": 244, "small.en": 244,
    "medium": 769, "medium.en": 769,
    "large-v1": 1550, "large-v2": 1550, "large-v3": 1550, "large": 1550,
}
# Bytes per parameter for each faster-whisper compute type.
WHISPER_BYTES_PER_PARAM = {
    "int8": 1, "int8_float16": 1, "int8_float32": 1,
    "float16": 2, "int16": 2, "bfloat16": 2,
    "float32": 4,
}

def _gpu_can_run(model_size, compute_type):
    """True if this GPU supports the compute type AND the model fits the VRAM budget."""
    cc = get_gpu_compute_capability()
    if cc is None or cc < CT2_MIN_CUDA_COMPUTE_CAPABILITY:
        return False
    needed = CT2_MIN_COMPUTE_CAPABILITY.get(compute_type, (3, 5))
    if cc < needed:
        return False
    params_m = WHISPER_PARAMS_M.get(model_size, 1550)
    est_gb = estimate_model_gb(params_m, WHISPER_BYTES_PER_PARAM.get(compute_type, 4))
    return select_device(est_gb) == "cuda"

def resolve_device_and_compute(device, model_size, compute_type):
    """Resolve device/compute_type 'auto' into concrete values for this machine.

    Prefers the GPU when the card supports a compute type *and* the model fits the
    VRAM budget; otherwise falls back to CPU with int8, which is by far the fastest
    CPU path and costs almost no accuracy compared to float32.
    """
    if compute_type == "auto":
        if device in ("auto", "cuda"):
            # Fastest supported GPU compute type first.
            for ct in ("float16", "int8", "float32"):
                if _gpu_can_run(model_size, ct):
                    print(f"[Auto] Whisper '{model_size}' fits the GPU using {ct}.")
                    return "cuda", ct
            if device == "cuda":
                print("[Auto] CUDA forced, but no supported compute type fits this GPU. Using CPU.")
        cc = get_gpu_compute_capability()
        if cc is None:
            reason = "no CUDA GPU available"
        elif cc < CT2_MIN_CUDA_COMPUTE_CAPABILITY:
            reason = (f"GPU compute capability {cc[0]}.{cc[1]} is below CTranslate2's minimum of "
                      f"{CT2_MIN_CUDA_COMPUTE_CAPABILITY[0]}.{CT2_MIN_CUDA_COMPUTE_CAPABILITY[1]}")
        else:
            reason = f"'{model_size}' does not fit the GPU VRAM budget"
        print(f"[Auto] Whisper '{model_size}' -> CPU / int8 ({reason}).")
        return "cpu", "int8"

    # An explicit compute type was requested.
    if device == "auto":
        device = "cuda" if _gpu_can_run(model_size, compute_type) else "cpu"
    if device == "cpu" and compute_type == "float16":
        print("[Note] float16 is not supported on CPU; using int8 instead.")
        compute_type = "int8"
    print(f"[Auto] Whisper '{model_size}' @ {compute_type} -> {device.upper()}")
    return device, compute_type

def _remove_quietly(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def _write_segments_to_srt(model, video_file, out_path, transcribe_kwargs,
                           word_timestamps, no_punctuation):
    """Transcribe video_file and write SRT blocks to out_path.

    faster-whisper returns a lazy generator, so transcription errors surface here
    during iteration rather than at model.transcribe(). Exceptions propagate to the
    caller, which is responsible for cleaning up out_path.
    """
    def strip_punct(text):
        if not no_punctuation:
            return text
        # Common and language specific punctuation to remove
        punct_chars = '.,!?;:()[]{}"\'-_۔،؟'
        for char in punct_chars:
            text = text.replace(char, '')
        return text.strip()

    segments, info = model.transcribe(video_file, **transcribe_kwargs)
    total_duration = info.duration

    with open(out_path, "w", encoding="utf-8") as f:
        item_index = 1
        for segment in segments:
            if word_timestamps and getattr(segment, 'words', None):
                for word in segment.words:
                    text = word.word.strip()
                    if text:
                        f.write(f"{item_index}\n{format_timestamp(word.start)} --> {format_timestamp(word.end)}\n{text}\n\n")
                        item_index += 1
            else:
                text = strip_punct(segment.text)
                if text:
                    f.write(f"{item_index}\n{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n{text}\n\n")
                    item_index += 1

            percent_complete = (segment.end / total_duration) * 100 if total_duration > 0 else 0
            print(f"Progress: [{percent_complete:.1f}%] - {format_timestamp(segment.start)} --> {format_timestamp(segment.end)} | {segment.text.strip()[:50]}...", end="\r", flush=True)

def generate_srt(video_file: str, model_size: str = "base", device: str = "auto", language: str = None,
                 compute_type: str = "auto", word_timestamps: bool = False, vad_filter: bool = True,
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

    # An existing but empty SRT means a previous run crashed part-way through.
    # Treat it as missing, otherwise it silently poisons every later run.
    if os.path.exists(srt_file) and os.path.getsize(srt_file) == 0:
        print(f"Found an empty SRT from a failed run; regenerating: '{srt_file}'")
        os.remove(srt_file)

    if not os.path.exists(srt_file):
        print(f"SRT does not exist. Creating original SRT from video...")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Please install faster-whisper.")
            sys.exit(1)

        device, compute_type = resolve_device_and_compute(device, model_size, compute_type)

        print(f"Loading Whisper model '{model_size}' on '{device}' with precision '{compute_type}'...")
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=7)
        except Exception as e:
            # If the GPU path fails (missing CUDA libs, unsupported card, OOM), fall back to CPU.
            if device == "cuda":
                cpu_compute = "int8" if compute_type == "float16" else compute_type
                print(f"[Warning] GPU load failed ({e}).")
                print(f"Falling back to CPU with compute type '{cpu_compute}'...")
                device = "cpu"
                compute_type = cpu_compute
                try:
                    model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=7)
                except Exception as e2:
                    print(f"Failed to load Whisper model on CPU... ({e2})")
                    sys.exit(1)
            else:
                print(f"Failed to load Whisper model... ({e})")
                sys.exit(1)

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

        # Transcribe into a temporary file and only publish it on success, so a crash
        # can never leave a half-written or empty SRT behind.
        tmp_file = srt_file + ".partial"
        try:
            _write_segments_to_srt(model, video_file, tmp_file, transcribe_kwargs,
                                   word_timestamps, no_punctuation)
        except Exception as e:
            _remove_quietly(tmp_file)
            # Some GPUs load a model fine but fail at inference (e.g. cards older than
            # CTranslate2's kernels support), so the retry has to live here, not at load.
            if device == "cuda":
                print(f"\n[Warning] GPU inference failed ({e}).")
                print("Reloading the model on CPU with int8 and retrying...")
                try:
                    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=7)
                    _write_segments_to_srt(model, video_file, tmp_file, transcribe_kwargs,
                                           word_timestamps, no_punctuation)
                except Exception as e2:
                    _remove_quietly(tmp_file)
                    print(f"\n❌ Transcription failed on CPU as well: {e2}")
                    sys.exit(1)
            else:
                print(f"\n❌ Transcription failed: {e}")
                sys.exit(1)

        os.replace(tmp_file, srt_file)
        print(f"\n✅ Successfully created '{srt_file}'. Exiting pipeline.")
        return srt_file, True
    else:
        print(f"✅ Found existing SRT: '{srt_file}'")
        return srt_file, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SRT from a video using Whisper.")
    parser.add_argument("video_file", help="Input video or audio file path")
    parser.add_argument("--model", default="base", help="Whisper model size.")
    parser.add_argument("--device", default="auto", help="Compute device: auto, cpu or cuda. 'auto' uses GPU for models under 2GB, else CPU.")
    parser.add_argument("--language", default="auto", help="ISO language code to force transcription to.")
    parser.add_argument("--compute_type", default="auto", help="Compute type: auto, float16, int8, float32. 'auto' picks the best for this machine.")
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
