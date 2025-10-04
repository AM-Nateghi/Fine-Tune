import re
import json
from tqdm import tqdm
import random


def clean_text(text):
    text = re.sub(r"[\n\r\t]", " ", text)  # حذف کاراکترهای کنترلی
    text = re.sub(r"&\w+;", "", text)  # حذف HTML entities مثل &zwj;
    text = re.sub(r"\u200c", "", text)  # حذف ZWNJ
    text = re.sub(r"\s+", " ", text).strip()  # نرمال‌سازی فاصله‌ها
    return text


def clean_html_tags(text):
    text = re.sub(r'<[^>]*style="[^"]*"[^>]*>', "", text)  # حذف تگ‌های با style
    text = re.sub(r"<[^>]+>", "", text)  # حذف باقی تگ‌ها
    return text


# بارگذاری دیتاست
with open("Note.Content.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# پاک‌سازی با tqdm
for item in tqdm(data, desc="Cleaning JSON"):
    for key, value in item.items():
        if isinstance(value, str):
            value = clean_html_tags(value)
            value = clean_text(value)
            item[key] = value

# اگر داده‌ها به صورت لیست نیستن، تبدیلش کن
if isinstance(data, dict):
    data = [data]

# تبدیل به فرمت استاندارد
converted_data = []
for item in tqdm(data, desc="Converting to QA format"):
    qa_pair = {
        "input": item.get("Title", "").strip(),
        "output": item.get("ShortDescription", "").strip(),
    }
    converted_data.append(qa_pair)

# شافل کردن داده‌ها برای جلوگیری از بایاس
random.shuffle(converted_data)

train_test_split = int(0.95 * len(converted_data))

# تقسیم به train و test
train_data = converted_data[:train_test_split]
test_data = converted_data[train_test_split:]

# ذخیره فایل‌ها
with open("qa_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("qa_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# # ذخیره خروجی
# with open("cleaned_data.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)
