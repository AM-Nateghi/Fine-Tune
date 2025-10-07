import json

input_path = "assets/dataset_output.jsonl"
output_path = "assets/dataset_output.filtered.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if (
                isinstance(obj, dict)
                and "question" in obj
                and "response" in obj
                and "score_ratio" in obj
            ):
                # فقط نمونه‌هایی که هر سه کلید را دارند بنویس
                outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                print(f"Skipped (missing keys) in line {i}: {line}")
        except Exception as e:
            print(f"Skipped (invalid JSON) in line {i}: {line}")