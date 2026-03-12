import os
import torch

def setup_threading():
    cpu_threads = os.cpu_count()
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

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
