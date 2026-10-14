import os
import json
import re
from openai import OpenAI

# ---------------- CONFIGURATION ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SECRET_FILE = os.path.join(PROJECT_ROOT, "secret", "openai_api_key.json")
DICT_FILE = os.path.join(DATA_DIR, "urdu_roman_dict.json")

# gpt-4o is ~16x the price of gpt-4o-mini but still only cents per video, and it is
# markedly better at Urdu/Arabic religious vocabulary. Accuracy is worth far more
# here than the fraction of a cent saved, since results are cached permanently.
OPENAI_MODEL = "gpt-4o"

# USD per 1M tokens (input, output).
PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

# Words per request. Smaller chunks keep the model's attention tight and avoid the
# output truncation that silently drops words from one huge request.
WORDS_PER_REQUEST = 150

# How many times to re-request words the model failed to return.
MAX_RETRIES = 3

# Fixed transliterations for Islamic honorific ligatures. These are unambiguous, so
# they are seeded directly rather than spending an API call and risking a variant.
HONORIFICS = {
    "ﷺ": "sallallahu alaihi wasallam",
    "ﷻ": "subhanahu wa ta'ala",
    "ؓ": "radiyallahu anhu",
}

SYSTEM_PROMPT = (
    "You are an expert transliterator of Urdu and Arabic into Roman script, specialising in "
    "Islamic religious lectures (dars, khutbah, tafsir). You transliterate sound, you never "
    "translate meaning. You always output valid JSON."
)

TRANSLITERATION_RULES = """Rules:
1. TRANSLITERATE the sound. Never translate into English. "کتاب" -> "kitaab" (never "book").
2. The text mixes Urdu with Arabic (Quran ayahs, hadith, and set phrases). Detect which one each
   word is and use the convention that fits:
   - Urdu -> common Roman Urdu as written by Pakistani/Indian speakers ("hai", "kya", "nahi").
   - Arabic -> standard Roman Arabic ("bismillah", "alhamdulillah", "rasoolihil kareem").
3. Be internally consistent: the same word must always map to the same spelling.
4. Use plain ASCII. No diacritics, no macrons, no special characters. Apostrophes only for ain/hamza
   where it is customary ("ta'ala", "qur'an").
5. Lowercase everything except proper nouns (Allah, Muhammad, Makkah, Quran).
6. Preserve well-known conventional spellings rather than inventing phonetic ones:
   "صلوۃ" -> "salaat", "نماز" -> "namaz", "توحید" -> "tawheed", "حدیث" -> "hadith".
7. Use the context sentence to disambiguate, but transliterate ONLY the requested word.
8. Never leave a word in Arabic/Urdu script and never return an empty string. Always give your
   best attempt."""
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

def group_by_context(missing_words):
    """Invert {word: sentence} into [{"context": sentence, "words": [...]}, ...].

    The caller's map repeats the same sentence once per word taken from it. Grouping
    sends each sentence a single time, which cuts input tokens several-fold.
    """
    by_sentence = {}
    for word, sentence in missing_words.items():
        by_sentence.setdefault(sentence, []).append(word)
    return [{"context": s, "words": w} for s, w in by_sentence.items()]

def looks_untransliterated(value):
    """True if the model echoed Arabic/Urdu script back instead of romanising it."""
    return bool(re.search(r'[؀-ۿﭐ-﷿ﹰ-﻿]', value))

def validate_translations(requested, received):
    """Split the model's reply into usable results and words that need another attempt."""
    good, bad = {}, []
    for word in requested:
        value = received.get(word)
        if not isinstance(value, str) or not value.strip() or looks_untransliterated(value):
            bad.append(word)
        else:
            good[word] = value.strip()
    return good, bad

def split_honorifics(word):
    """Split a word into (base, [honorific romanisations]).

    Models reliably romanise the base but leave the ligature attached ("اللہﷺ" ->
    "Allahﷺ"), so the ligatures are removed here and re-attached afterwards instead.
    """
    base = word
    found = []
    for ligature, roman in HONORIFICS.items():
        if ligature in base:
            base = base.replace(ligature, "")
            found.append(roman)
    return base.strip(), found

def request_batch(client, words, contexts, model):
    """Ask the model for one chunk of words. Returns ({word: roman}, response).

    Words are sent with numeric ids and the reply is keyed by id rather than by the
    word itself. Models sometimes silently "correct" a word containing transcription
    noise and answer under the corrected spelling, which can never be matched back
    to what was asked.
    """
    ids = {str(i): w for i, w in enumerate(words, 1)}
    by_sentence = {}
    for i, w in enumerate(words, 1):
        by_sentence.setdefault(contexts[w], []).append({"id": str(i), "word": w})
    payload = {"items": [{"context": s, "words": ws} for s, ws in by_sentence.items()]}

    user_msg = (
        f"{TRANSLITERATION_RULES}\n\n"
        f"Transliterate every word in the JSON below. The 'context' field is only there to help you "
        f"disambiguate; do not transliterate the context itself.\n"
        f"Answer for every id you are given, keyed by that same id. If a word looks misspelled, "
        f"transliterate it as best you can, but still answer under its original id.\n\n"
        f'Return JSON of exactly this shape:\n{{"transliterations": {{"<id>": "<roman>"}}}}\n\n'
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            # Structurally guarantees parseable JSON, so no markdown stripping is needed.
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        raw = data.get("transliterations", {})
        return {ids[k]: v for k, v in raw.items() if k in ids and isinstance(v, str)}, response
    except Exception as e:
        print(f"  [API error on batch of {len(words)}]: {e}")
        return {}, None

def get_openai_bulk_translation(client, missing_words, model=OPENAI_MODEL):
    """Transliterate every missing word, retrying any the model drops or fumbles.

    missing_words: {"urdu_word": "context sentence it appeared in"}
    Returns {"urdu_word": "roman"} for everything successfully transliterated.
    """
    if not client or not missing_words:
        return {}

    results = {}
    composition = {}   # original word -> (base word, [honorific romanisations])
    contexts = {}      # base word -> a context sentence it appeared in

    for word, sentence in missing_words.items():
        base, honorifics = split_honorifics(word)
        if not base:
            # The "word" was nothing but honorific ligatures; no API call needed.
            results[word] = " ".join(honorifics)
            continue
        composition[word] = (base, honorifics)
        contexts.setdefault(base, sentence)

    pending = list(contexts)
    resolved = {}
    total_in = total_out = 0
    print(f"  -> Transliterating {len(pending)} cache misses via {model}...")

    for attempt in range(1, MAX_RETRIES + 1):
        if not pending:
            break
        if attempt > 1:
            print(f"  -> Retry {attempt - 1}/{MAX_RETRIES - 1} for {len(pending)} unresolved word(s)...")

        still_pending = []
        chunks = [pending[i:i + WORDS_PER_REQUEST] for i in range(0, len(pending), WORDS_PER_REQUEST)]
        for n, chunk in enumerate(chunks, 1):
            received, response = request_batch(client, chunk, contexts, model)
            good, bad = validate_translations(chunk, received)
            resolved.update(good)
            still_pending.extend(bad)

            if response is not None:
                usage = response.usage
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens
                print(f"     [{n}/{len(chunks)}] id={response.id} model={response.model} "
                      f"ok={len(good)} missing={len(bad)} "
                      f"tokens={usage.prompt_tokens}+{usage.completion_tokens}")
        pending = still_pending

    # Re-attach honorifics to their base word: "اللہﷺ" -> "Allah sallallahu alaihi wasallam".
    for original, (base, honorifics) in composition.items():
        roman = resolved.get(base)
        if roman:
            results[original] = " ".join([roman] + honorifics)

    rate_in, rate_out = PRICING.get(model, PRICING["gpt-4o"])
    cost = (total_in / 1e6) * rate_in + (total_out / 1e6) * rate_out
    print(f"  -> Done: {len(results)}/{len(missing_words)} transliterated | "
          f"{total_in:,} in + {total_out:,} out tokens | ~${cost:.4f}")
    failed = [w for w in missing_words if w not in results]
    if failed:
        print(f"  [!] {len(failed)} word(s) failed after {MAX_RETRIES} attempts "
              f"and stay in original script: {', '.join(failed[:8])}"
              f"{' ...' if len(failed) > 8 else ''}")
    return results

def process_srt(input_path, output_path):
    print(f"Loading Offline Dictionary ({DICT_FILE})...")
    urdu_dict = load_dict()
    print(f"Dictionary Size: {len(urdu_dict)} words.")

    # Honorific ligatures are unambiguous, and some are cached mapped to themselves
    # (e.g. "ﷺ" -> "ﷺ"), which would never be treated as a miss and so would stay
    # untransliterated forever. Force the correct values regardless of the cache.
    urdu_dict.update(HONORIFICS)
    
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
            # Use the raw token as the dictionary key. Nothing is stripped, so words
            # that differ only by diacritics (e.g. "billah" vs "billahi") stay distinct.
            if not word:
                continue
            cached = urdu_dict.get(word)
            # Re-request words that are absent, and also ones a previous run cached
            # while still in the original script (e.g. "خربان" -> "khurbaان"), so the
            # dictionary heals itself instead of serving bad entries forever.
            if cached is None or looks_untransliterated(cached):
                # Store the word and the sentence it lived in for context!
                if word not in missing_words:
                    missing_words[word] = stripped
                    
    # --- BULK TRANSLATE MISSING WORDS ---
    if missing_words:
        new_translations = get_openai_bulk_translation(client, missing_words)
        
        # Merge new findings into dictionary RAM. Bad cached entries are overwritten,
        # not skipped, otherwise the self-healing pass above could never take effect.
        dict_updates = 0
        for k, v in new_translations.items():
            if k not in urdu_dict or looks_untransliterated(urdu_dict[k]):
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
                    # Look up the raw token directly and output its roman value.
                    roman = urdu_dict.get(word)
                    if roman and not looks_untransliterated(str(roman)):
                        roman_words.append(str(roman))
                    else:
                        # Not in the dictionary (or a bad entry): keep the original token.
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