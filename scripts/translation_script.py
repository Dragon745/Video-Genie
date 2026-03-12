import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils import setup_threading, parse_srt, save_srt

def detect_language(blocks):
    sample_text = " ".join([b[2] for b in blocks[:5]])
    try:
        from langdetect import detect
        detected_lang = detect(sample_text)
    except Exception as e:
        detected_lang = "ur"
    return detected_lang

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

    code_map = {"ur": "urd_Arab", "hi": "hin_Deva", "en": "eng_Latn"}
    nllb_src = code_map.get(detected_lang, "eng_Latn")
    nllb_tgt = code_map.get(tgt_lang, "eng_Latn")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=nllb_src)
    # Force full model ingestion into System RAM rather than streaming from disk cache
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float32
    )
    
    # Attempt to compile the model for optimized speeds (Experimental on Windows)
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
            inputs = tokenizer(flat_text, return_tensors="pt")
            tgt_lang_id = tokenizer.convert_tokens_to_ids(nllb_tgt)
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
