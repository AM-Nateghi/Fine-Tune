import os
import hashlib
import torch
from typing import Dict, Any, List
from functools import partial
import json
from pathlib import Path

import huggingface_hub
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ================================================================
# Configuration
# ================================================================
class Config:
    # Model
    MODEL_NAME = "google/gemma-3-270m"
    USE_QLORA = True

    # 🔥 DATA SAMPLING - برای تست
    SAMPLE_RATIO = 0.01  # 1% از 303 سوال = ~3 سوال
    # برای production بزن 1.0

    # Data - ساختار جدید
    DATA_FILE = "assets/dataset_output.json"  # فایل JSON با ساختار جدید
    MAX_LENGTH = 512

    # 🔥 استراتژی استفاده از پاسخ‌ها
    USE_NEGATIVE_SAMPLES = True  # آیا پاسخ‌های بد هم استفاده بشه؟
    MAX_RESPONSES_PER_QUESTION = 50  # حداکثر پاسخ برای هر سوال (برای بالانس)

    # QLoRA
    LORA_R = 256
    LORA_ALPHA = 512
    LORA_DROPOUT = 0.03
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
    ]

    # Quantization
    LOAD_IN_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT = True

    # Curriculum
    CURRICULUM_PHASES = [
        {"name": "test_phase", "min_score": 0.8, "epochs": 2, "lr": 2e-4},
    ]

    # Training
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Checkpointing
    SAVE_STEPS = 100
    EVAL_STEPS = 100
    LOGGING_STEPS = 20
    SAVE_TOTAL_LIMIT = 3
    RESUME_FROM_CHECKPOINT = True

    # Output
    OUTPUT_DIR = "./outputs/gemma3_new"
    BACKUP_DIR = "./outputs/gemma3_new/backups"


config = Config()

# ================================================================
# Environment
# ================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
hf_token = ""
if hf_token:
    huggingface_hub.login(token=hf_token)


# ================================================================
# Safety Checkpoint Manager
# ================================================================
class SafeCheckpointCallback(TrainerCallback):
    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.log_file = self.backup_dir / "training_log.jsonl"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            loss = metrics["eval_loss"]

            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_loss": loss,
                "train_loss": (
                    state.log_history[-1].get("loss") if state.log_history else None
                ),
                "learning_rate": (
                    state.log_history[-1].get("learning_rate")
                    if state.log_history
                    else None
                ),
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            if loss < self.best_loss:
                self.best_loss = loss
                backup_path = self.backup_dir / f"best_model_step{state.global_step}"
                print(f"\n💾 NEW BEST! Backing up to {backup_path}")

                for old_backup in self.backup_dir.glob("best_model_step*"):
                    if old_backup != backup_path:
                        import shutil

                        shutil.rmtree(old_backup, ignore_errors=True)


# ================================================================
# Quantization
# ================================================================
def get_bnb_config():
    if not config.USE_QLORA:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )


# ================================================================
# Model Setup
# ================================================================
def setup_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model (QLoRA: {config.USE_QLORA})")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if config.USE_QLORA:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
            inference_mode=False,
            modules_to_save=["lm_head"],
        )

        model = get_peft_model(model, lora_config)
        print("\n📊 Trainable Parameters:")
        model.print_trainable_parameters()
    else:
        model.gradient_checkpointing_enable()

    return model, tokenizer


model, tokenizer = setup_model_and_tokenizer()


# ================================================================
# Normalization
# ================================================================
ARABIC_TO_PERSIAN = {"\u064a": "ی", "\u0643": "ک"}
PERSIAN_DIGITS_TO_EN = {f"۰۱۲۳۴۵۶۷۸۹"[i]: str(i) for i in range(10)}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for ar, fa in ARABIC_TO_PERSIAN.items():
        text = text.replace(ar, fa)
    for fa, en in PERSIAN_DIGITS_TO_EN.items():
        text = text.replace(fa, en)
    text = " ".join(text.split())
    text = text.replace(" ؟", "؟").replace("،", "، ")
    return text.strip()


# ================================================================
# 🔥 NEW: Dataset Processing با ساختار جدید
# ================================================================
def split_key(text: str) -> int:
    """برای split کردن train/val/test"""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:6], 16) % 100


def load_new_structure_data(min_score: float) -> List[Dict[str, Any]]:
    """
    خواندن دیتا با ساختار جدید:
    [
        {
            "question_id": "...",
            "best_response": "...",
            "positive_responses": [{"text": "...", "score_ratio": 1.0}, ...],
            "negative_responses": [{"text": "...", "score_ratio": 0.0}, ...],
            "questions": ["سوال اصلی", "variant 1", ...]
        },
        ...
    ]
    """
    print(f"\n📂 Loading data from {config.DATA_FILE}")
    with open(config.DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sampling برای تست
    if config.SAMPLE_RATIO < 1.0:
        n_samples = int(len(data) * config.SAMPLE_RATIO)
        import random

        random.seed(42)
        data = random.sample(data, n_samples)
        print(f"   📊 Sampled: {len(data)} questions")

    # تبدیل به فرمت flat برای training
    examples = []
    stats = {"total_pairs": 0, "positive_pairs": 0, "negative_pairs": 0, "skipped": 0}

    for item in data:
        question_variants = item.get("questions", [])
        if not question_variants:
            stats["skipped"] += 1
            continue

        # انتخاب سوال اصلی (اولین variant)
        main_question = question_variants[0]

        # پاسخ‌های مثبت
        positive_responses = item.get("positive_responses", [])
        # محدود کردن تعداد پاسخ‌ها
        if len(positive_responses) > config.MAX_RESPONSES_PER_QUESTION:
            import random

            positive_responses = random.sample(
                positive_responses, config.MAX_RESPONSES_PER_QUESTION
            )

        for resp in positive_responses:
            if resp.get("score_ratio", 0) >= min_score:
                examples.append(
                    {
                        "question": main_question,
                        "response": resp["text"],
                        "score_ratio": resp["score_ratio"],
                        "label": "positive",
                    }
                )
                stats["positive_pairs"] += 1

        # پاسخ‌های منفی (اختیاری)
        if config.USE_NEGATIVE_SAMPLES:
            negative_responses = item.get("negative_responses", [])
            # تعداد کمتری از negative samples
            max_neg = min(len(negative_responses), len(positive_responses) // 2)
            if len(negative_responses) > max_neg:
                import random

                negative_responses = random.sample(negative_responses, max_neg)

            for resp in negative_responses[:max_neg]:
                examples.append(
                    {
                        "question": main_question,
                        "response": resp["text"],
                        "score_ratio": resp.get("score_ratio", 0.0),
                        "label": "negative",
                    }
                )
                stats["negative_pairs"] += 1

        stats["total_pairs"] = stats["positive_pairs"] + stats["negative_pairs"]

    print(f"\n📊 Data Statistics:")
    print(f"   Total Q-A pairs: {stats['total_pairs']:,}")
    print(f"   Positive pairs: {stats['positive_pairs']:,}")
    print(f"   Negative pairs: {stats['negative_pairs']:,}")
    print(f"   Skipped questions: {stats['skipped']}")

    return examples


def preprocess_function(examples: Dict[str, list]):
    """Tokenization برای ساختار جدید"""
    texts = []
    for q, a, label in zip(
        examples["question"], examples["response"], examples["label"]
    ):
        if not (isinstance(q, str) and isinstance(a, str)):
            texts.append("")
            continue

        q_norm = normalize_text(q)
        a_norm = normalize_text(a)

        # 🔥 برای negative samples می‌تونی prefix اضافه کنی (اختیاری)
        if label == "negative":
            # اگر می‌خوای مدل یاد بگیره چی بد هست:
            # text = f"سوال: {q_norm}\nپاسخ نادرست: {a_norm}"
            # ولی معمولا negative samples رو نمی‌زاریم توی causal LM
            texts.append("")  # Skip negative samples in training
            continue

        text = f"سوال: {q_norm}\nپاسخ: {a_norm}"
        texts.append(text)

    return tokenizer(
        texts,
        truncation=True,
        max_length=config.MAX_LENGTH,
        padding=False,
    )


SPLIT_RANGES = {
    "train": range(0, 95),
    "validation": range(95, 97),
    "test": range(97, 100),
}


def create_curriculum_dataset(min_score: float):
    """ساخت dataset با split های train/val/test"""
    print(f"\n📚 Creating curriculum dataset (min_score={min_score})")

    # Load data
    raw_examples = load_new_structure_data(min_score)

    # Split به train/val/test
    datasets = {}
    for split_name, split_range in SPLIT_RANGES.items():
        split_examples = [
            ex for ex in raw_examples if split_key(ex["question"]) in split_range
        ]

        # تبدیل به HuggingFace Dataset
        ds = Dataset.from_list(split_examples)

        # Tokenization
        ds = ds.map(
            preprocess_function,
            batched=True,
            remove_columns=ds.column_names,
            num_proc=4,
        )

        # فیلتر کردن examples خالی
        ds = ds.filter(lambda ex: len(ex.get("input_ids", [])) > 10, num_proc=4)

        if split_name == "train":
            ds = ds.shuffle(seed=42)

        datasets[split_name] = ds
        print(f"  ✅ {split_name}: {len(ds):,} examples")

    return datasets


# ================================================================
# Training Args
# ================================================================
def get_training_args(phase_name: str, num_epochs: int, lr: float):
    return TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR}/{phase_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=lr,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        warmup_ratio=config.WARMUP_RATIO,
        bf16=True,
        bf16_full_eval=True,
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        logging_steps=config.LOGGING_STEPS,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_safetensors=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        gradient_checkpointing=config.USE_QLORA,
        optim="paged_adamw_8bit" if config.USE_QLORA else "adamw_torch",
        report_to=["tensorboard"],
        run_name=f"gemma3_{phase_name}",
        resume_from_checkpoint=config.RESUME_FROM_CHECKPOINT,
    )


def get_last_checkpoint(output_dir: str):
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    last_checkpoint = str(checkpoints[-1])
    print(f"\n🔄 Found checkpoint: {last_checkpoint}")
    return last_checkpoint


# ================================================================
# Training Loop
# ================================================================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n" + "=" * 70)
print("🚀 TRAINING START - NEW DATA STRUCTURE")
print(f"   Sample: {config.SAMPLE_RATIO*100:.0f}% | QLoRA: {config.USE_QLORA}")
print(f"   Use Negatives: {config.USE_NEGATIVE_SAMPLES}")
print("=" * 70)

for i, phase in enumerate(config.CURRICULUM_PHASES, 1):
    print(f"\n{'='*70}")
    print(f"📖 PHASE {i}/{len(config.CURRICULUM_PHASES)}: {phase['name'].upper()}")
    print(f"{'='*70}")

    datasets = create_curriculum_dataset(phase["min_score"])
    args = get_training_args(phase["name"], phase["epochs"], phase["lr"])

    checkpoint_callback = SafeCheckpointCallback(config.BACKUP_DIR)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        callbacks=[checkpoint_callback],
    )

    last_checkpoint = None
    if config.RESUME_FROM_CHECKPOINT:
        last_checkpoint = get_last_checkpoint(args.output_dir)

    print(f"\n🏋️ Training...")
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("💾 Model state saved. You can resume later.")
        raise

    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    print(f"\n✅ Results:")
    print(f"   Loss: {eval_results['eval_loss']:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")

    trainer.save_model()


# ================================================================
# Test Evaluation
# ================================================================
print("\n" + "=" * 70)
print("🧪 TEST EVALUATION")
print("=" * 70)

test_ds = create_curriculum_dataset(0.7)["test"]
test_results = trainer.evaluate(eval_dataset=test_ds)
test_ppl = torch.exp(torch.tensor(test_results["eval_loss"]))
print(f"Test Loss: {test_results['eval_loss']:.4f}")
print(f"Test Perplexity: {test_ppl:.2f}")


# ================================================================
# Save Final
# ================================================================
final_path = f"{config.OUTPUT_DIR}/final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"\n💾 Final model: {final_path}")


# ================================================================
# 🔥 Persian Quality Test
# ================================================================
print("\n" + "=" * 70)
print("🇮🇷 PERSIAN QUALITY TEST")
print("=" * 70)

model.eval()
test_cases = [
    "معنی زندگی چیست؟",
    "چرا آسمان آبی است؟",
    "چگونه می‌توانم خوشبخت باشم؟",
    "تفاوت عشق و علاقه چیست؟",
]

for q in test_cases:
    prompt = f"سوال: {q}\nپاسخ:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.5,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("پاسخ:")[-1].strip()
    print(f"\n❓ {q}")
    print(f"💬 {answer[:200]}...")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print(f"📊 TensorBoard: tensorboard --logdir={config.OUTPUT_DIR}/logs")
print(f"💾 Backups: {config.BACKUP_DIR}")
print("=" * 70)
