"""
Data Flattening Script - JSONL Input Version
=============================================
تبدیل ساختار nested به flat dataset برای SFT و DPO

Input Structure (JSONL - هر خط یک JSON):
{
  "question_id": "...",
  "positive_responses": [{"text": "...", "score_ratio": 1.0}, ...],
  "negative_responses": [{"text": "...", "score_ratio": 0.0}, ...],
  "questions": ["سوال اصلی", "variant 1", ...]
}

Output:
- sft_dataset.jsonl: برای Supervised Fine-Tuning
- dpo_dataset.jsonl: برای Direct Preference Optimization
- dataset_stats.json: آمار کامل
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import numpy as np


# ================================================================
# Configuration
# ================================================================
class Config:
    INPUT_FILE = "assets/merged_dataset.jsonl"  # 🔥 حالا JSONL
    OUTPUT_DIR = "assets/flattened"

    # SFT Settings
    SFT_OUTPUT = "sft_dataset.jsonl"
    SFT_MIN_SCORE = 0.8  # حداقل score برای پاسخ‌های مثبت

    # DPO Settings
    DPO_OUTPUT = "dpo_dataset.jsonl"
    DPO_MIN_POSITIVE_SCORE = 0.9  # حداقل score برای chosen
    DPO_MAX_NEGATIVE_SCORE = 0.3  # حداکثر score برای rejected
    DPO_PAIRING_STRATEGY = "random"  # "random", "best_worst", "score_based"

    # Balancing
    ENABLE_BALANCING = True
    MAX_SAMPLES_PER_QUESTION = 1000  # برای جلوگیری از غلبه سوالات پرپاسخ ولی خب تعداد سوالات خیلی بیشتر از این عدده
    MIN_SAMPLES_PER_QUESTION = 5  # حداقل برای DPO pairing

    # Quality Control
    MIN_TEXT_LENGTH = 10  # حداقل طول متن (کاراکتر)
    MAX_TEXT_LENGTH = 2000  # حداکثر طول متن

    SEED = 42


config = Config()
random.seed(config.SEED)
np.random.seed(config.SEED)


# ================================================================
# Text Normalization
# ================================================================
ARABIC_TO_PERSIAN = {"\u064a": "ی", "\u0643": "ک"}
PERSIAN_DIGITS_TO_EN = {f"۰۱۲۳۴۵۶۷۸۹"[i]: str(i) for i in range(10)}


def normalize_text(text: str) -> str:
    """نرمال‌سازی متن فارسی"""
    if not isinstance(text, str):
        return ""

    # تبدیل عربی به فارسی
    for ar, fa in ARABIC_TO_PERSIAN.items():
        text = text.replace(ar, fa)

    # تبدیل اعداد فارسی به انگلیسی
    for fa, en in PERSIAN_DIGITS_TO_EN.items():
        text = text.replace(fa, en)

    # حذف فضاهای اضافی
    text = " ".join(text.split())

    # اصلاح نشانه‌گذاری
    text = text.replace(" ؟", "؟").replace("،", "، ")

    return text.strip()


def is_valid_text(text: str) -> bool:
    """چک کردن validity متن"""
    if not text or not isinstance(text, str):
        return False

    text_len = len(text)
    if text_len < config.MIN_TEXT_LENGTH or text_len > config.MAX_TEXT_LENGTH:
        return False

    # چک کردن اینکه متن فقط کاراکترهای عجیب نباشه
    if len(text.strip()) < 5:
        return False

    return True


# ================================================================
# Load JSONL Data
# ================================================================
def load_jsonl_data() -> List[Dict]:
    """
    خواندن فایل JSONL (هر خط یک JSON object)
    """
    print(f"\n📂 Loading data from: {config.INPUT_FILE}")

    data = []
    with open(config.INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Warning: Invalid JSON at line {line_num}: {e}")
                continue

    print(f"   ✅ Loaded {len(data)} questions")
    return data


# ================================================================
# SFT Dataset Generation
# ================================================================
def generate_sft_dataset(data: List[Dict]) -> List[Dict]:
    """
    تولید dataset برای SFT
    Format: {"question": "...", "response": "...", "score": 1.0, "weight": 0.5}
    """
    print("\n" + "=" * 70)
    print("📚 GENERATING SFT DATASET")
    print("=" * 70)

    sft_samples = []
    question_response_counts = Counter()
    skipped = {"invalid_text": 0, "low_score": 0, "no_questions": 0}

    for item_idx, item in enumerate(data):
        question_variants = item.get("questions", [])
        if not question_variants:
            skipped["no_questions"] += 1
            continue

        positive_responses = item.get("positive_responses", [])
        question_id = item.get("question_id", f"q_{item_idx}")

        # محدود کردن تعداد پاسخ‌ها
        if len(positive_responses) > config.MAX_SAMPLES_PER_QUESTION:
            # انتخاب بهترین‌ها براساس score
            positive_responses = sorted(
                positive_responses, key=lambda x: x.get("score_ratio", 0), reverse=True
            )[: config.MAX_SAMPLES_PER_QUESTION]

        valid_samples_for_question = 0

        for resp in positive_responses:
            score = resp.get("score_ratio", 0)

            # فیلتر score
            if score < config.SFT_MIN_SCORE:
                skipped["low_score"] += 1
                continue

            response_text = normalize_text(resp.get("text", ""))

            # Validation
            if not is_valid_text(response_text):
                skipped["invalid_text"] += 1
                continue

            # انتخاب یک variant رندوم از سوال
            question_text = normalize_text(random.choice(question_variants))

            if not is_valid_text(question_text):
                skipped["invalid_text"] += 1
                continue

            sft_samples.append(
                {
                    "question": question_text,
                    "response": response_text,
                    "score": float(score),
                    "question_id": question_id,
                    "weight": 1.0,  # بعدا محاسبه می‌شه
                }
            )

            valid_samples_for_question += 1

        question_response_counts[question_id] = valid_samples_for_question

    # ================================================================
    # Weight Balancing
    # ================================================================
    if config.ENABLE_BALANCING and sft_samples:
        print("\n⚖️  Calculating sample weights for balancing...")

        # محاسبه میانگین تعداد پاسخ‌ها
        response_counts = list(question_response_counts.values())
        avg_responses = np.mean(response_counts)

        print(f"   Average responses per question: {avg_responses:.1f}")
        print(f"   Min: {min(response_counts)}, Max: {max(response_counts)}")

        # محاسبه weight برای هر sample
        for sample in sft_samples:
            q_id = sample["question_id"]
            count = question_response_counts[q_id]

            # weight معکوس تعداد (سوالات کم‌پاسخ وزن بیشتر می‌گیرن)
            sample["weight"] = avg_responses / count if count > 0 else 1.0

        # نرمال‌سازی weights
        total_weight = sum(s["weight"] for s in sft_samples)
        for sample in sft_samples:
            sample["weight"] = sample["weight"] / total_weight * len(sft_samples)

        weight_stats = [s["weight"] for s in sft_samples]
        print(f"   Weight range: {min(weight_stats):.3f} - {max(weight_stats):.3f}")

    print(f"\n✅ SFT Samples: {len(sft_samples):,}")
    print(f"   Skipped - Invalid text: {skipped['invalid_text']:,}")
    print(f"   Skipped - Low score: {skipped['low_score']:,}")
    print(f"   Skipped - No questions: {skipped['no_questions']:,}")

    return sft_samples


# ================================================================
# DPO Dataset Generation
# ================================================================
def generate_dpo_dataset(data: List[Dict]) -> List[Dict]:
    """
    تولید dataset برای DPO
    Format: {"question": "...", "chosen": "...", "rejected": "...", "weight": 0.5}

    skipped دیکشنری: ذخیره تعداد samples که skip شدن به دلایل مختلف:
    - no_positives: سوالاتی که پاسخ مثبت معتبر ندارن
    - no_negatives: سوالاتی که پاسخ منفی معتبر ندارن
    - invalid_text: پاسخ‌هایی که validation رو pass نکردن
    - insufficient_samples: سوالاتی که کمتر از MIN_SAMPLES_PER_QUESTION pair دارن
    """
    print("\n" + "=" * 70)
    print("🔀 GENERATING DPO DATASET")
    print("=" * 70)

    dpo_samples = []
    question_pair_counts = Counter()

    # 🔥 این دیکشنری تعداد samples که به دلایل مختلف skip شدن رو نگه میداره
    skipped = {
        "no_positives": 0,  # سوالاتی که پاسخ مثبت معتبر ندارن
        "no_negatives": 0,  # سوالاتی که پاسخ منفی معتبر ندارن
        "invalid_text": 0,  # پاسخ‌هایی که خیلی کوتاه/بلند یا نامعتبر بودن
        "insufficient_samples": 0,  # سوالاتی که بعد از pairing کمتر از حد مجاز pair داشتن
    }

    for item_idx, item in enumerate(data):
        question_variants = item.get("questions", [])
        if not question_variants:
            continue

        question_id = item.get("question_id", f"q_{item_idx}")

        # فیلتر کردن پاسخ‌های مثبت (chosen)
        # باید score بالا داشته باشن و متن معتبر باشه
        positive_responses = [
            r
            for r in item.get("positive_responses", [])
            if r.get("score_ratio", 0) >= config.DPO_MIN_POSITIVE_SCORE
            and is_valid_text(normalize_text(r.get("text", "")))
        ]

        # فیلتر کردن پاسخ‌های منفی (rejected)
        # باید score پایین داشته باشن و متن معتبر باشه
        negative_responses = [
            r
            for r in item.get("negative_responses", [])
            if r.get("score_ratio", 1) <= config.DPO_MAX_NEGATIVE_SCORE
            and is_valid_text(normalize_text(r.get("text", "")))
        ]

        # اگه پاسخ مثبت نداریم، این سوال رو skip می‌کنیم
        if not positive_responses:
            skipped["no_positives"] += 1
            continue

        # اگه پاسخ منفی نداریم، این سوال رو skip می‌کنیم
        if not negative_responses:
            skipped["no_negatives"] += 1
            continue

        # Pairing: ساخت جفت‌های (مثبت، منفی)
        pairs = create_pairs(
            positive_responses, negative_responses, config.DPO_PAIRING_STRATEGY
        )

        # محدود کردن تعداد pairs (برای بالانس)
        if len(pairs) > config.MAX_SAMPLES_PER_QUESTION:
            pairs = random.sample(pairs, config.MAX_SAMPLES_PER_QUESTION)

        # اگه خیلی کم pair داریم، این سوال رو skip می‌کنیم
        if len(pairs) < config.MIN_SAMPLES_PER_QUESTION:
            skipped["insufficient_samples"] += 1
            continue

        # ساخت samples نهایی
        for pos_resp, neg_resp in pairs:
            question_text = normalize_text(random.choice(question_variants))
            chosen_text = normalize_text(pos_resp.get("text", ""))
            rejected_text = normalize_text(neg_resp.get("text", ""))

            dpo_samples.append(
                {
                    "question": question_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "chosen_score": float(pos_resp.get("score_ratio", 1.0)),
                    "rejected_score": float(neg_resp.get("score_ratio", 0.0)),
                    "question_id": question_id,
                    "weight": 1.0,
                }
            )

        question_pair_counts[question_id] = len(pairs)

    # ================================================================
    # Weight Balancing
    # ================================================================
    if config.ENABLE_BALANCING and dpo_samples:
        print("\n⚖️  Calculating sample weights for balancing...")

        pair_counts = list(question_pair_counts.values())
        avg_pairs = np.mean(pair_counts)

        print(f"   Average pairs per question: {avg_pairs:.1f}")
        print(f"   Min: {min(pair_counts)}, Max: {max(pair_counts)}")

        for sample in dpo_samples:
            q_id = sample["question_id"]
            count = question_pair_counts[q_id]
            sample["weight"] = avg_pairs / count if count > 0 else 1.0

        total_weight = sum(s["weight"] for s in dpo_samples)
        for sample in dpo_samples:
            sample["weight"] = sample["weight"] / total_weight * len(dpo_samples)

        weight_stats = [s["weight"] for s in dpo_samples]
        print(f"   Weight range: {min(weight_stats):.3f} - {max(weight_stats):.3f}")

    print(f"\n✅ DPO Pairs: {len(dpo_samples):,}")
    print(f"   Skipped - No positives: {skipped['no_positives']:,}")
    print(f"   Skipped - No negatives: {skipped['no_negatives']:,}")
    print(f"   Skipped - Insufficient samples: {skipped['insufficient_samples']:,}")

    return dpo_samples


def create_pairs(
    positive_responses: List[Dict], negative_responses: List[Dict], strategy: str
) -> List[tuple]:
    """ایجاد pairs از chosen و rejected"""

    if strategy == "random":
        # رندوم pairing
        n_pairs = min(len(positive_responses), len(negative_responses))
        pos_shuffled = random.sample(positive_responses, n_pairs)
        neg_shuffled = random.sample(negative_responses, n_pairs)
        return list(zip(pos_shuffled, neg_shuffled))

    elif strategy == "best_worst":
        # بهترین positive با بدترین negative
        pos_sorted = sorted(
            positive_responses, key=lambda x: x.get("score_ratio", 0), reverse=True
        )
        neg_sorted = sorted(negative_responses, key=lambda x: x.get("score_ratio", 1))
        n_pairs = min(len(pos_sorted), len(neg_sorted))
        return list(zip(pos_sorted[:n_pairs], neg_sorted[:n_pairs]))

    elif strategy == "score_based":
        # Pairing براساس اختلاف score
        pairs = []
        for pos in positive_responses:
            for neg in negative_responses:
                score_diff = pos.get("score_ratio", 1) - neg.get("score_ratio", 0)
                if score_diff > 0.5:  # حداقل 0.5 اختلاف
                    pairs.append((pos, neg))

        # اگر خیلی زیاد شد، sample کن
        if len(pairs) > config.MAX_SAMPLES_PER_QUESTION:
            pairs = random.sample(pairs, config.MAX_SAMPLES_PER_QUESTION)

        return pairs

    else:
        raise ValueError(f"Unknown pairing strategy: {strategy}")


# ================================================================
# Statistics
# ================================================================
def calculate_statistics(sft_samples: List[Dict], dpo_samples: List[Dict]) -> Dict:
    """محاسبه آمار کامل"""

    stats = {
        "sft": {
            "total_samples": len(sft_samples),
            "unique_questions": len(set(s["question_id"] for s in sft_samples)),
            "avg_response_length": (
                np.mean([len(s["response"]) for s in sft_samples]) if sft_samples else 0
            ),
            "score_distribution": {},
            "weight_distribution": {},
        },
        "dpo": {
            "total_pairs": len(dpo_samples),
            "unique_questions": len(set(s["question_id"] for s in dpo_samples)),
            "avg_chosen_length": (
                np.mean([len(s["chosen"]) for s in dpo_samples]) if dpo_samples else 0
            ),
            "avg_rejected_length": (
                np.mean([len(s["rejected"]) for s in dpo_samples]) if dpo_samples else 0
            ),
            "weight_distribution": {},
        },
    }

    # Score distribution for SFT
    if sft_samples:
        scores = [s["score"] for s in sft_samples]
        stats["sft"]["score_distribution"] = {
            "min": float(min(scores)),
            "max": float(max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
        }

        weights = [s["weight"] for s in sft_samples]
        stats["sft"]["weight_distribution"] = {
            "min": float(min(weights)),
            "max": float(max(weights)),
            "mean": float(np.mean(weights)),
        }

    # Weight distribution for DPO
    if dpo_samples:
        weights = [s["weight"] for s in dpo_samples]
        stats["dpo"]["weight_distribution"] = {
            "min": float(min(weights)),
            "max": float(max(weights)),
            "mean": float(np.mean(weights)),
        }

    return stats


# ================================================================
# Main Execution
# ================================================================
def main():
    print("\n" + "=" * 70)
    print("🚀 DATA FLATTENING SCRIPT - JSONL VERSION")
    print("=" * 70)

    # ایجاد output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 خواندن دیتای JSONL
    data = load_jsonl_data()

    # تولید SFT dataset
    sft_samples = generate_sft_dataset(data)

    # تولید DPO dataset
    dpo_samples = generate_dpo_dataset(data)

    # محاسبه آمار
    print("\n" + "=" * 70)
    print("📊 CALCULATING STATISTICS")
    print("=" * 70)
    stats = calculate_statistics(sft_samples, dpo_samples)

    print(f"\n✅ SFT Statistics:")
    print(f"   Total samples: {stats['sft']['total_samples']:,}")
    print(f"   Unique questions: {stats['sft']['unique_questions']}")
    print(f"   Avg response length: {stats['sft']['avg_response_length']:.0f} chars")
    print(
        f"   Score range: {stats['sft']['score_distribution'].get('min', 0):.2f} - {stats['sft']['score_distribution'].get('max', 0):.2f}"
    )

    print(f"\n✅ DPO Statistics:")
    print(f"   Total pairs: {stats['dpo']['total_pairs']:,}")
    print(f"   Unique questions: {stats['dpo']['unique_questions']}")
    print(f"   Avg chosen length: {stats['dpo']['avg_chosen_length']:.0f} chars")
    print(f"   Avg rejected length: {stats['dpo']['avg_rejected_length']:.0f} chars")

    # Shuffle
    print("\n🔀 Shuffling datasets...")
    random.shuffle(sft_samples)
    random.shuffle(dpo_samples)

    # ذخیره SFT
    sft_path = output_dir / config.SFT_OUTPUT
    print(f"\n💾 Saving SFT dataset to: {sft_path}")
    with open(sft_path, "w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"   ✅ Saved {len(sft_samples):,} samples")

    # ذخیره DPO
    dpo_path = output_dir / config.DPO_OUTPUT
    print(f"\n💾 Saving DPO dataset to: {dpo_path}")
    with open(dpo_path, "w", encoding="utf-8") as f:
        for sample in dpo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"   ✅ Saved {len(dpo_samples):,} pairs")

    # ذخیره آمار
    stats_path = output_dir / "dataset_stats.json"
    print(f"\n💾 Saving statistics to: {stats_path}")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("✅ FLATTENING COMPLETE!")
    print(f"   SFT: {sft_path}")
    print(f"   DPO: {dpo_path}")
    print(f"   Stats: {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
