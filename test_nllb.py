import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M" # Use the small one for fast testing
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="urd_Arab")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

texts = [
    "وہ",
    "خدا",
    "یہ بہت سے لوگوں کو دیا جائے گا.",
    "The longer string is used to force a much longer padding sequence for the shorter strings because NLLB can get confused." * 5
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
tgt_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")

outputs = model.generate(**inputs, forced_bos_token_id=tgt_lang_id, max_length=400)
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for r in results:
    print(r)
