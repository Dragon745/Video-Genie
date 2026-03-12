import os
import json
import time
from openai import OpenAI

# ---------------- CONFIGURATION ----------------
DATA_DIR = r"d:\GitHub\Video Genie\data"
SECRET_FILE = r"d:\GitHub\Video Genie\secret\openai_api_key.json"
WORDS_FILE = os.path.join(DATA_DIR, "words.txt")
DICT_FILE = os.path.join(DATA_DIR, "urdu_roman_dict.json")
STATE_FILE = os.path.join(DATA_DIR, "batch_state.json")
BATCH_FILE = os.path.join(DATA_DIR, "current_batch.jsonl")

# How many words to explicitly ask the AI to translate in one single prompt
# We keep this at 500 so the AI output token limit isn't reached per prompt.
WORDS_PER_PROMPT = 500
# -----------------------------------------------

def load_api_key():
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

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

def get_unprocessed_words(existing_dict):
    if not os.path.exists(WORDS_FILE):
        print(f"Error: Could not find {WORDS_FILE}")
        return []
        
    with open(WORDS_FILE, 'r', encoding='utf-8') as f:
        all_words = [line.strip() for line in f if line.strip()]
    
    # Intelligently filter out words we ALREADY paid to translate
    unprocessed = [w for w in all_words if w not in existing_dict]
    return unprocessed

def create_jsonl_batch(unprocessed_words, file_path):
    # Group words into clean chunks of 500 words each.
    # For 154,000 words, this will create ~310 chunks.
    chunks = [unprocessed_words[i:i + WORDS_PER_PROMPT] for i in range(0, len(unprocessed_words), WORDS_PER_PROMPT)]
    
    if not chunks:
        return False
        
    print(f"Preparing {len(chunks)} AI prompts (Total words this batch: {len(unprocessed_words)})...")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            custom_id = f"req_{int(time.time())}_{i}"
            
            prompt_text = (
                "You are an expert Urdu linguist. Transliterate the following list of Urdu words into highly colloquial "
                "Roman Urdu (the way people text in Pakistan/India). "
                "Return ONLY a valid JSON object where keys are the original Urdu words and values are the Roman Urdu strings. "
                "Do not include markdown blocks like ```json or any conversational text. ONLY output the raw JSON brackets.\n\n"
            )
            prompt_text += "\n".join(chunk)
            
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini", # Extremely cheap, highly accurate for basic translation
                    "messages": [
                        {"role": "system", "content": "You are a transliteration API. You output raw valid JSON only."},
                        {"role": "user", "content": prompt_text}
                    ],
                    "temperature": 0.1 # Low temperature so AI doesn't get creative with spellings
                }
            }
            # The Batch API requires perfectly formatted JSONL specifically
            f.write(json.dumps(request_obj) + "\n")
            
    return True

def process_completed_batch(client, batch_id, output_file_id):
    print(f"Great! Downloading results for completed batch {batch_id}...")
    
    # Download the `.jsonl` results file OpenAI created
    file_response = client.files.content(output_file_id)
    
    if not file_response:
        print("Error: Could not download the result file content from OpenAI.")
        return
        
    # Split the multi-line JSON responses
    responses = file_response.text.strip().split('\n')
    
    existing_dict = load_dict()
    words_added = 0
    words_failed = 0
    
    for resp_line in responses:
        if not resp_line: continue
        try:
            resp_data = json.loads(resp_line)
            # Drill down into the JSON to extract the AI's actual text message
            content = resp_data['response']['body']['choices'][0]['message']['content']
            
            # Clean up OpenAI sometimes trying to be helpful and returning markdown formatting
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Convert that text into a Python Dictionary
            new_words = json.loads(content)
            
            # Append completely new words to our Master List Dictionary
            for k, v in new_words.items():
                if k not in existing_dict:
                    existing_dict[k] = v
                    words_added += 1
                    
        except Exception as e:
            # If AI hallucinated or broke the JSON format for one prompt, we just skip it.
            # Next time the script runs, it will realize these words are still missing and try them again!
            words_failed += 1
            continue
            
    save_dict(existing_dict)
    print(f"\n=> Successfully added {words_added} new words to your local offline Dictionary!")
    if words_failed > 0:
         print(f"=> {words_failed} blocks failed parsing. They will be retried automatically if you run the script again!")
         
    # Extremely important: Clear out our memory so the script knows we are ready for a brand new Batch!
    clear_state()

def main():
    api_key = load_api_key()
    if not api_key:
        print(f"Error: Could not load OPENAI_API_KEY from {SECRET_FILE}.")
        return
        
    client = OpenAI(api_key=api_key)
    
    state = load_state()
    
    # ========================================================
    # SCENARIO B & C: Check if we left a job running!
    # ========================================================
    if "batch_id" in state:
        batch_id = state["batch_id"]
        print(f"Found active batch job: {batch_id}. Pinging OpenAI for status...")
        
        try:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            print(f"Current OpenAI Status: {status.upper()}")
            
            if status == "completed":
                if batch_status.output_file_id:
                    process_completed_batch(client, batch_id, batch_status.output_file_id)
                    print("\n[✔] ALL DONE! Dictionary fully updated.")
                else:
                    print("Error: Batch completed but OpenAI failed to generate an output file. Soft resetting.")
                    clear_state()
                    
            elif status in ["failed", "expired", "cancelled"]:
                print(f"[!] Batch mysteriously {status}! Clearing our active status so it safely RETRIES the exact same work block next run.")
                if batch_status.errors:
                    print(f"OpenAI Errors: {batch_status.errors}")
                clear_state()
                print("\nRun me again to retry!")
                
            else:
                # "validating", "in_progress", "finalizing", etc.
                print(f"[⏳] Batch is still processing on OpenAI servers. Close this terminal and check back later!")
                return
                
        except Exception as e:
            print(f"Error checking batch status with OpenAI API: {e}")
            return
            
        # We successfully processed an outcome (Downloaded or Failed). We EXIT here to let the user re-run manually.
        return

    # ========================================================
    # SCENARIO A: No active job! Build the massive payload.
    # ========================================================
    print("No active jobs running. Looking for new work...")
    existing_dict = load_dict()
    print(f"Current Local Dictionary Size: {len(existing_dict)} words.")
    
    unprocessed_words = get_unprocessed_words(existing_dict)
    
    if not unprocessed_words:
        print(f"\n[🏆] COMPLETE! All words in words.txt have been successfully transliterated!")
        return
        
    print(f"Found {len(unprocessed_words)} words remaining in the queue.")
    
    # Build our massive payload
    if not create_jsonl_batch(unprocessed_words, BATCH_FILE):
        print("Failed to create .jsonl batch file payload.")
        return
        
    print("\nUploading giant payload securely to OpenAI...")
    try:
        with open(BATCH_FILE, "rb") as f:
            batch_file_obj = client.files.create(
                file=f,
                purpose="batch"
            )
        
        print(f"File uploaded successfully! (File ID: {batch_file_obj.id})")
        print("Submitting the massive job to the OpenAI Batch Queue...")
        
        batch_job = client.batches.create(
            input_file_id=batch_file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h" # The only available Batch queue option. OpenAI promises it will finish within 24 hours. Usually takes 30 mins to 2 hours.
        )
        
        print(f"Job Active! (Batch ID: {batch_job.id})")
        
        # Keep track of this so our script remembers it when we run it again later!
        save_state({
            "batch_id": batch_job.id,
            "file_id": batch_file_obj.id
        })
        
        print("\n[✔] SUCCESS! OpenAI is now crunching all of your words in the background.")
        print("You can completely close your computer now. Run this script again in a few hours to download the resulting dictionary!")
        
    except Exception as e:
        print(f"Critical error communicating with OpenAI API: {e}")

if __name__ == "__main__":
    main()
