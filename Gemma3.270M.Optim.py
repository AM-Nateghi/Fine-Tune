import os
import hashlib
import torch
from typing import Dict, Any
from functools import partial
import json
from pathlib import Path

import huggingface_hub
from datasets import load_dataset
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

    # 🔥 DATA - فایل flatten شده
    SFT_DATA_FILE = "assets/flattened/sft_dataset.jsonl"
    MAX_LENGTH = 512

    # 🔥 Sampling
    SAMPLE_RATIO = 0.01  # برای تست - 1% دیتا
    # برای production: 1.0

    # 🔥 Weighted Training
    USE_SAMPLE_WEIGHTS = True  # استفاده از weight های محاسبه شده

    # QLoRA
    LORA_R = 64
    LORA_ALPHA = 128
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
    # Production:
    # {"name": "warmup", "min_score": 0.95, "epochs": 1, "lr": 5e-5},
    # {"name": "main", "min_score": 0.85, "epochs": 2, "lr": 2e-4},
    # {"name": "finetune", "min_score": 0.8, "epochs": 2, "lr": 1e-4},

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
    OUTPUT_DIR = "./outputs/gemma3_weighted"
    BACKUP_DIR = "./outputs/gemma3_weighted/backups"


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
    """
    Callback برای:
    1. Backup کردن بهترین checkpoint
    2. لاگ کردن دقیق metrics
    """

    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.log_file = self.backup_dir / "training_log.jsonl"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            loss = metrics["eval_loss"]

            # لاگ کردن metrics
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

            # Backup بهترین مدل
            if loss < self.best_loss:
                self.best_loss = loss
                backup_path = self.backup_dir / f"best_model_step{state.global_step}"
                print(f"\n💾 NEW BEST! Backing up to {backup_path}")

                # حذف backup های قدیمی
                for old_backup in self.backup_dir.glob("best_model_step*"):
                    if old_backup != backup_path:
                        import shutil

                        shutil.rmtree(old_backup, ignore_errors=True)


# ================================================================
# 🔥 Custom Weighted Trainer
# ================================================================
class WeightedTrainer(Trainer):
    """
    Trainer با پشتیبانی از sample weights

    این Trainer می‌تونه به هر sample یک وزن اختصاص بده.
    مثلا: سوالاتی که کمتر پاسخ دارن، وزن بیشتری می‌گیرن
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override loss برای اعمال weights

        نحوه کار:
        1. weights رو از inputs استخراج می‌کنیم
        2. Forward pass معمولی
        3. Loss رو با weights ضرب می‌کنیم
        """

        # استخراج weights از inputs
        weights = inputs.pop("weights", None)

        # Forward pass
        outputs = model(**inputs)

        # محاسبه loss
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # اعمال weights
        if weights is not None and config.USE_SAMPLE_WEIGHTS:
            # weights shape: (batch_size,)
            # میانگین weights batch رو می‌گیریم و به loss اعمال می‌کنیم
            weight_avg = weights.mean()
            loss = loss * weight_avg

        return (loss, outputs) if return_outputs else loss


# ================================================================
# Quantization
# ================================================================
def get_bnb_config():
    """
    تنظیمات 4-bit quantization برای QLoRA

    چرا quantization؟
    - کاهش حافظه VRAM (از 24GB به 12GB)
    - سرعت بیشتر
    - دقت تقریباً یکسان

    نوع: NF4 (NormalFloat4) - بهترین برای LLM ها
    """
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
    """نرمال‌سازی متن فارسی"""
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
# Dataset Processing
# ================================================================
def split_key(text: str) -> int:
    """
    برای split کردن train/val/test به صورت deterministic

    از hash استفاده می‌کنیم تا همیشه یک سوال توی همون split بمونه
    """
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:6], 16) % 100


def preprocess_function(examples: Dict[str, list]):
    """
    Tokenization برای flattened data

    Input format:
    {
        "question": "...",
        "response": "...",
        "score": 1.0,
        "weight": 0.5
    }

    Output: tokenized text + weights
    """
    texts = []
    weights = []

    for q, r, score, weight in zip(
        examples["question"],
        examples["response"],
        examples["score"],
        examples["weight"],
    ):
        if not (isinstance(q, str) and isinstance(r, str)):
            texts.append("")
            weights.append(1.0)
            continue

        q_norm = normalize_text(q)
        r_norm = normalize_text(r)

        text = f"سوال: {q_norm}\nپاسخ: {r_norm}"
        texts.append(text)
        weights.append(float(weight))

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=config.MAX_LENGTH,
        padding=False,
    )

    # اضافه کردن weights به output
    tokenized["weights"] = weights

    return tokenized


SPLIT_RANGES = {
    "train": range(0, 95),  # 95% train
    "validation": range(95, 97),  # 2% validation
    "test": range(97, 100),  # 3% test
}


def create_curriculum_dataset(min_score: float):
    """
    Load و split کردن flattened dataset

    مراحل:
    1. Load کردن JSONL
    2. Sampling (اگه لازم باشه)
    3. فیلتر براساس score
    4. Split به train/val/test
    5. Tokenization
    """
    print(f"\n📚 Loading flattened dataset (min_score={min_score})")
    print(f"   File: {config.SFT_DATA_FILE}")

    # Load dataset
    raw = load_dataset(
        "json", data_files={"train": config.SFT_DATA_FILE}, split="train"
    )

    print(f"   Total samples: {len(raw):,}")

    # 🔥 Sampling برای تست
    if config.SAMPLE_RATIO < 1.0:
        n_samples = int(len(raw) * config.SAMPLE_RATIO)
        raw = raw.shuffle(seed=42).select(range(n_samples))
        print(f"   Sampled: {len(raw):,} examples")

    # فیلتر براساس score
    raw = raw.filter(lambda ex: ex.get("score", 0) >= min_score, num_proc=4)
    print(f"   After score filter (>={min_score}): {len(raw):,}")

    # Split به train/val/test
    datasets = {}
    for split_name, split_range in SPLIT_RANGES.items():
        ds = raw.filter(
            lambda ex: split_key(ex.get("question", "")) in split_range,
            num_proc=4,
        )

        # Tokenization
        ds = ds.map(
            preprocess_function,
            batched=True,
            remove_columns=raw.column_names,
            num_proc=4,
        )

        # فیلتر examples خالی
        ds = ds.filter(lambda ex: len(ex.get("input_ids", [])) > 10, num_proc=4)

        if split_name == "train":
            ds = ds.shuffle(seed=42)

        datasets[split_name] = ds
        print(f"  ✅ {split_name}: {len(ds):,} examples")

    return datasets


# ================================================================
# 🔥 Custom Data Collator با Weight Support
# ================================================================
class WeightedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator که weights رو هم handle می‌کنه

    نحوه کار:
    1. از هر feature، weight رو جدا می‌کنیم
    2. باقی کارها رو collator معمولی انجام می‌ده
    3. weights رو به batch اضافه می‌کنیم
    """

    def __call__(self, features):
        # استخراج weights
        weights = [f.pop("weights", 1.0) for f in features]

        # Collate معمولی
        batch = super().__call__(features)

        # اضافه کردن weights به batch
        batch["weights"] = torch.tensor(weights, dtype=torch.float32)

        return batch


# ================================================================
# Training Args
# ================================================================
def get_training_args(phase_name: str, num_epochs: int, lr: float):
    """تنظیمات training برای هر phase"""
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
        # Precision
        bf16=True,
        bf16_full_eval=True,
        # Logging
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        logging_steps=config.LOGGING_STEPS,
        logging_first_step=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        # Checkpointing
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_safetensors=True,
        # Efficiency
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        gradient_checkpointing=config.USE_QLORA,
        optim="paged_adamw_8bit" if config.USE_QLORA else "adamw_torch",
        # Reporting
        report_to=["tensorboard"],
        run_name=f"gemma3_{phase_name}",
        # Resume support
        resume_from_checkpoint=config.RESUME_FROM_CHECKPOINT,
    )


def get_last_checkpoint(output_dir: str):
    """پیدا کردن آخرین checkpoint برای resume"""
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
data_collator = WeightedDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n" + "=" * 70)
print("🚀 TRAINING START - WEIGHTED & BALANCED")
print(f"   Sample: {config.SAMPLE_RATIO*100:.0f}% | QLoRA: {config.USE_QLORA}")
print(f"   Use Weights: {config.USE_SAMPLE_WEIGHTS}")
print("=" * 70)

for i, phase in enumerate(config.CURRICULUM_PHASES, 1):
    print(f"\n{'='*70}")
    print(f"📖 PHASE {i}/{len(config.CURRICULUM_PHASES)}: {phase['name'].upper()}")
    print(f"{'='*70}")

    datasets = create_curriculum_dataset(phase["min_score"])
    args = get_training_args(phase["name"], phase["epochs"], phase["lr"])

    checkpoint_callback = SafeCheckpointCallback(config.BACKUP_DIR)

    # 🔥 استفاده از WeightedTrainer
    trainer = WeightedTrainer(
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
# Persian Quality Test
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
