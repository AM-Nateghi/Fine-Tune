import json
from transformers import AutoTokenizer
from tqdm import tqdm

file_path = "assets/dataset_output.jsonl"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

batch_size = 512  # می‌تونی بسته به GPU بیشتر یا کمتر کنی
texts = []
total_tokens = 0
total_samples = 0

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in tqdm(range(0, len(lines), batch_size), desc="Batch tokenizing", unit="batch"):
    batch = []
    for line in lines[i : i + batch_size]:
        data = json.loads(line)
        q, r = data.get("question", ""), data.get("response", "")
        batch.append(q.strip() + "\n" + r.strip())
    # توکنایز کل batch یکجا
    encodings = tokenizer(batch, padding=False, truncation=False, return_tensors=None)
    total_tokens += sum(len(ids) for ids in encodings["input_ids"])
    total_samples += len(batch)

print("\n📊 Dataset Tokenization Summary")
print(f"Total tokens: {total_tokens}")
print(f"Total samples: {total_samples}")
print(f"Average tokens per sample: {total_tokens / total_samples:.2f}")
