import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils import (
    setup_threading, parse_srt, save_srt, estimate_model_gb, select_device,
    get_gpu_compute_capability,
)

# Approximate parameter counts (in millions) for the NLLB model variants.
NLLB_PARAMS_M = {
    "facebook/nllb-200-distilled-600M": 615,
    "facebook/nllb-200-1.3B": 1370,
    "facebook/nllb-200-distilled-1.3B": 1370,
    "facebook/nllb-200-3.3B": 3300,
}

# Languages this pipeline supports, mapped to their NLLB codes.
NLLB_CODE_MAP = {
    "ur": "urd_Arab",   # Urdu
    "hi": "hin_Deva",   # Hindi
    "en": "eng_Latn",   # English
    "ar": "arb_Arab",   # Arabic
}

def detect_language(blocks):
    """Detect the dominant language of an SRT by majority vote across all lines.

    The old version sampled only the first few blocks, which in a religious lecture
    are the Arabic opening (bismillah etc.) - so an otherwise Urdu file was labelled
    Arabic. Voting over the whole file lets the true majority language win, and any
    language outside NLLB_CODE_MAP is folded into the closest supported one.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic results across runs
    except Exception:
        return "ur"

    from collections import Counter
    votes = Counter()
    for _idx, _ts, text in blocks:
        text = text.strip()
        if len(text) < 4:
            continue  # too short to classify reliably
        try:
            lang = detect(text)
        except Exception:
            continue
        # Fold unsupported detections onto a supported neighbour rather than dropping
        # to English: Persian/Arabic-script variants are, for our purposes, Urdu/Arabic.
        if lang not in NLLB_CODE_MAP:
            lang = {"fa": "ur", "ps": "ur", "ckb": "ar"}.get(lang, lang)
        if lang in NLLB_CODE_MAP:
            votes[lang] += 1

    if not votes:
        return "ur"
    winner, _count = votes.most_common(1)[0]
    print(f"  Language vote across {sum(votes.values())} lines: {dict(votes)}")
    return winner

def translate_srt(srt_file: str, tgt_lang: str, model_name: str = "facebook/nllb-200-3.3B") -> tuple[str, str, list]:
    """
    Translates an SRT file to the specified language (en, hi, ur).
    Returns a tuple of:
    - the path to the translated (or original) SRT file
    - the auto-detected language of the input SRT
    - the blocks of the translated (or original) SRT file
    """
    setup_threading()

    if not os.path.exists(srt_file):
        print(f"Error: File '{srt_file}' not found.")
        sys.exit(1)

    blocks = parse_srt(srt_file)
    if not blocks:
        print("SRT file is empty or corrupted.")
        sys.exit(1)

    detected_lang = detect_language(blocks)
    print(f"Auto-detected language of existing SRT: {detected_lang}")

    if not tgt_lang or tgt_lang == detected_lang:
        print("No translation needed or skipping translation.")
        return srt_file, detected_lang, blocks

    print(f"\nLoading NLLB-200 Translation Model for '{tgt_lang}'...")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        print("Please perform: pip install transformers torch sentencepiece")
        sys.exit(1)

    nllb_src = NLLB_CODE_MAP.get(detected_lang, "urd_Arab")
    nllb_tgt = NLLB_CODE_MAP.get(tgt_lang, "eng_Latn")
    print(f"Translating {nllb_src} -> {nllb_tgt}")

    # Decide device: use the GPU only if the model (in float16) fits the 2GB rule.
    # On the GPU we load in float16 to halve the footprint; on CPU we keep float32.
    params_m = NLLB_PARAMS_M.get(model_name, 3300)
    est_fp16_gb = estimate_model_gb(params_m, 2)
    device = select_device(est_fp16_gb)

    # Cards below compute capability 5.3 have no native fp16 math, so half precision
    # there is emulated and ends up slower than just running on the CPU.
    if device == "cuda":
        cc = get_gpu_compute_capability()
        if cc is None or cc < (5, 3):
            print(f"[Auto-Device] GPU compute capability {cc} has no native fp16 support; using CPU.")
            device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[Auto-Device] NLLB '{model_name}' ~ {est_fp16_gb:.2f} GB (fp16) -> {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=nllb_src)
    # Force full model ingestion into RAM rather than streaming from disk cache
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=dtype,
        # Force the safetensors weights. Newer transformers refuses the .bin path
        # (torch.load) unless torch >= 2.6 (CVE-2025-32434); safetensors is both safe
        # and the file every NLLB variant on the Hub already ships.
        use_safetensors=True,
    )
    try:
        model = model.to(device)
    except Exception as e:
        print(f"[Warning] Could not place model on {device} ({e}). Using CPU float32.")
        device = "cpu"
        model = model.float().to("cpu")

    tgt_lang_id = tokenizer.convert_tokens_to_ids(nllb_tgt)

    # Warm-up probe: catch unsupported-GPU errors (e.g. old cards) before the full run.
    if device == "cuda":
        try:
            probe = tokenizer("test", return_tensors="pt").to(device)
            with torch.no_grad():
                model.generate(**probe, forced_bos_token_id=tgt_lang_id, max_length=8)
        except Exception as e:
            print(f"[Warning] GPU inference failed ({e}). Falling back to CPU float32.")
            device = "cpu"
            model = model.float().to("cpu")

    # Attempt to compile the model for optimized speeds (Experimental)
    try:
        print("Attempting to compile the model with Torch.compile (this may take a minute or fail)...")
        model = torch.compile(model)
        print("Model compiled successfully for faster translations!")
    except Exception as e:
        print(f"Skipping torch.compile (Unsupported/Failed on this OS or hardware): {e}")

    translated_blocks = []
    print(f"Translating to {tgt_lang}...")
    total_blocks = len(blocks)
    for i, (idx, ts, text) in enumerate(blocks):
        pct = (i / total_blocks) * 100
        print(f"Progress: [{pct:.1f}%] - Translating line {i+1}/{total_blocks}...", end='\r')
        if text.strip():
            flat_text = text.replace('\n', ' ')
            inputs = tokenizer(flat_text, return_tensors="pt").to(device)
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs, forced_bos_token_id=tgt_lang_id, max_length=400
                )
            result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_blocks.append((idx, ts, result))
        else:
            translated_blocks.append((idx, ts, text))
            
    base_name = os.path.splitext(os.path.basename(srt_file))[0]
    work_dir = os.path.dirname(srt_file)
    
    translated_srt_file = os.path.join(work_dir, f"{base_name}_{tgt_lang}.srt")
    save_srt(translated_blocks, translated_srt_file)
    
    print(f"\n✅ Saved translated SRT to: {translated_srt_file}")
    return translated_srt_file, detected_lang, translated_blocks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate an SRT file.")
    parser.add_argument("srt_file", help="Input SRT file path")
    parser.add_argument("--lang", choices=["en", "hi", "ur"], help="Target language code (en, hi, ur). If not provided, it detects the language and exits.")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B", help="The NLLB huggingface model string to use for translation.")
    
    args = parser.parse_args()
    translate_srt(args.srt_file, args.lang, args.model)
