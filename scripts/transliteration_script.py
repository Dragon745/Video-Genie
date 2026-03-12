import os
import json
import re
from openai import OpenAI

# ---------------- CONFIGURATION ----------------
DATA_DIR = r"d:\GitHub\Video Genie\data"
SECRET_FILE = r"d:\GitHub\Video Genie\secret\openai_api_key.json"
DICT_FILE = os.path.join(DATA_DIR, "urdu_roman_dict.json")
# -----------------------------------------------

def load_api_key():
    if not os.path.exists(SECRET_FILE):
        return None
    with open(SECRET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("OPENAI_API_KEY")

def load_dict():
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_dict(dictionary):
    with open(DICT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

def clean_word(word):
    # Remove punctuation for dictionary lookup, but keep it for output
    # IMPORTANT: Do not strip specific Islamic unicode ligatures so they can be explicitly translated
    cleaned = re.sub(r'[^\w\sﷺﷻؓ]', '', word).strip()
    return cleaned

def get_openai_bulk_translation(client, missing_words):
    """
    Sends all missing words from the SRT to OpenAI in ONE single API call.
    missing_words is a dictionary: {"urdu_word": "The full context sentence"}
    Returns a dictionary of {"urdu_word": "roman_translation"}
    """
    if not client or not missing_words:
        return {}
        
    print(f"  -> Bulk Requesting {len(missing_words)} cache misses from OpenAI in one trip...")
    
    # We build a JSON string of what we want decoded to keep the prompt super structured.
    prompt_payload = json.dumps(missing_words, ensure_ascii=False, indent=2)
    
    prompt = (
        f"You are an expert in colloquial Urdu to Roman Urdu transliteration.\n"
        f"Below is a JSON object where keys are missing Urdu words, and values are the sentences they appeared in for context.\n"
        f"Transliterate each Urdu key into Roman Urdu. Respond ONLY with a valid raw JSON object where keys are the original Urdu words, and values are your Roman Urdu transliterations.\n"
        f"Do NOT include markdown blocks like ```json. ONLY output the raw JSON brackets.\n\n"
        f"{prompt_payload}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a transliterator. Output only valid raw JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up in case the AI accidentally returned markdown despite instructions
        content = content.replace('```json', '').replace('```', '').strip()
        return json.loads(content)
        
    except Exception as e:
        print(f"  [API Error translating bulk definitions]: {e}")
        return {}

def process_srt(input_path, output_path):
    print(f"Loading Offline Dictionary ({DICT_FILE})...")
    urdu_dict = load_dict()
    print(f"Dictionary Size: {len(urdu_dict)} words.")
    
    api_key = load_api_key()
    client = OpenAI(api_key=api_key) if api_key else None
    if not client:
        print("Warning: No OpenAI API key found. Missing words will remain in Urdu.")

    print(f"\nReading from {input_path}")
    if not os.path.exists(input_path):
         print(f"Error: Could not find {input_path}")
         return
         
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # --- PASS 1: SCAN FOR MISSING WORDS ---
    missing_words = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.isdigit() or '-->' in stripped:
            continue
            
        words = stripped.split()
        for word in words:
            cleaned = clean_word(word)
            if cleaned and cleaned not in urdu_dict:
                # Store the word and the sentence it lived in for context!
                if cleaned not in missing_words:
                    missing_words[cleaned] = stripped
                    
    # --- BULK TRANSLATE MISSING WORDS ---
    if missing_words:
        new_translations = get_openai_bulk_translation(client, missing_words)
        
        # Merge new findings into dictionary RAM
        dict_updates = 0
        for k, v in new_translations.items():
            if k not in urdu_dict:
                urdu_dict[k] = v
                dict_updates += 1
                
        if dict_updates > 0:
            print(f"Saving {dict_updates} newly bulk-discovered words to the dictionary...")
            save_dict(urdu_dict)
            
    # --- PASS 2: RECONSTRUCT THE SRT ---
    print(f"Transliterating and saving...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            stripped = line.strip()
            # Skip empty lines, sequence numbers, and timestamp lines
            if len(stripped) == 0:
                f.write('\n')
            elif stripped.isdigit():
                f.write(line)
            elif '-->' in stripped:
                f.write(line)
            else:
                # Text line: transliterate it
                words = stripped.split()
                roman_words = []
                
                for word in words:
                    cleaned = clean_word(word)
                    
                    if not cleaned:
                        roman_words.append(word)
                        continue
                        
                    if cleaned in urdu_dict:
                        # Re-attach any punctuation that was originally attached to the word
                        roman_word = str(urdu_dict[cleaned])
                        final_word = word.replace(cleaned, roman_word)
                        roman_words.append(final_word)
                    else:
                        # Even after OpenAI bulk retry, it failed to translate this specific word.
                        # Leave it as original Urdu rather than crashing the sentence.
                        roman_words.append(word)
                        
                # Reconstruct the line
                roman_line = " ".join(roman_words)
                f.write(roman_line + '\n')
                
    print(f"\nDone! Entire SRT saved flawlessly to {output_path}")

def run_transliteration():
    # Dummy function to maintain compatibility with legacy gui.py imports.
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transliterate an SRT file.")
    parser.add_argument("input_file", type=str, help="Path to the input SRT file")
    parser.add_argument("--lang", type=str, default="ur", help="Language of input (hi/ur)")
    args = parser.parse_args()

    input_file = args.input_file
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_roman{ext}"
    
    if os.path.exists(input_file):
        process_srt(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file}")