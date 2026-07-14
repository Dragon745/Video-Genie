import os
import torch

def setup_threading():
    cpu_threads = os.cpu_count()
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

def get_gpu_free_memory_gb():
    """Return free VRAM (GB) on the default CUDA GPU, or 0.0 if none is usable.

    Returns 0.0 for CPU-only torch builds (torch.cuda.is_available() is False),
    so callers can treat 0.0 as "no GPU available".
    """
    try:
        if not torch.cuda.is_available():
            return 0.0
        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0

def get_gpu_compute_capability():
    """Return the CUDA compute capability as (major, minor), or None if no usable GPU."""
    try:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None

def estimate_model_gb(params_millions, bytes_per_param):
    """Rough in-memory size of a model from its parameter count and dtype width."""
    return (params_millions * 1_000_000 * bytes_per_param) / (1024 ** 3)

def select_device(estimated_model_gb, max_gpu_model_gb=2.0, overhead_gb=0.3):
    """Choose 'cuda' or 'cpu' for a model of the given estimated size.

    Rule (per user request): run on the GPU only when the model is small enough
    (<= max_gpu_model_gb, default 2 GB) AND there is currently enough free VRAM to
    hold it plus a little overhead. Otherwise use the CPU. Always returns 'cpu'
    when no usable CUDA GPU is present.
    """
    free = get_gpu_free_memory_gb()
    if free <= 0:
        return "cpu"
    if estimated_model_gb > max_gpu_model_gb:
        return "cpu"
    if estimated_model_gb + overhead_gb > free:
        return "cpu"
    return "cuda"

def format_timestamp(seconds: float) -> str:
    """Convert a timestamp in seconds to SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def parse_srt(srt_file):
    with open(srt_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    blocks = content.strip().split("\n\n")
    parsed_blocks = []
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            idx = lines[0]
            timestamp = lines[1]
            text = "\n".join(lines[2:])
            parsed_blocks.append((idx, timestamp, text))
    return parsed_blocks

def save_srt(parsed_blocks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, timestamp, text in parsed_blocks:
            f.write(f"{idx}\n{timestamp}\n{text}\n\n")
