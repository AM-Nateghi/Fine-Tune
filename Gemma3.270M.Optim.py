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
    USE_QLORA = True  # QLoRA برای سرعت

    # 🔥 DATA SAMPLING - برای تست 30 دقیقه‌ای
    SAMPLE_RATIO = 0.005  # 5% داده = ~2.95M توکن
    # برای production بزن 1.0

    # Data
    DATA_FILE = "assets/dataset_output.jsonl"
    MAX_LENGTH = 512

    # QLoRA - متعادل برای کیفیت فارسی
    LORA_R = 256  # 🔥 خیلی بالا برای حفظ زبان فارسی
    LORA_ALPHA = 512  # 2×R
    LORA_DROPOUT = 0.03  # پایین‌تر = stability بیشتر
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # Attention
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP
        "embed_tokens",  # 🔥 مهم برای فارسی!
    ]

    # Quantization
    LOAD_IN_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT = True

    # Curriculum - کوتاه برای تست
    CURRICULUM_PHASES = [
        {"name": "test_phase", "min_score": 0.8, "epochs": 2, "lr": 2e-4},
    ]
    # برای production:
    # {"name": "warmup", "min_score": 0.9, "epochs": 1, "lr": 5e-5},
    # {"name": "main", "min_score": 0.8, "epochs": 2, "lr": 2e-4},
    # {"name": "finetune", "min_score": 0.7, "epochs": 2, "lr": 1e-4},

    # Training - بهینه برای RTX 4070
    BATCH_SIZE = 4  # 12GB VRAM سیف
    GRADIENT_ACCUMULATION = 4  # Effective = 16
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # 🔥 Checkpointing - قوی برای production
    SAVE_STEPS = 100
    EVAL_STEPS = 100
    LOGGING_STEPS = 20
    SAVE_TOTAL_LIMIT = 3  # آخرین 3 checkpoint
    RESUME_FROM_CHECKPOINT = True  # خودکار resume

    # Output
    OUTPUT_DIR = "./outputs/gemma3_safe"
    BACKUP_DIR = "./outputs/gemma3_safe/backups"


config = Config()

# ================================================================
# Environment
# ================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # دیباگ بهتر
hf_token = ""
if hf_token:
    huggingface_hub.login(token=hf_token)


# ================================================================
# Safety Checkpoint Manager
# ================================================================
class SafeCheckpointCallback(TrainerCallback):
    """Backup بهترین checkpoint + لاگ دقیق"""

    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.log_file = self.backup_dir / "training_log.jsonl"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            loss = metrics["eval_loss"]

            # لاگ کردن
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
            modules_to_save=["lm_head"],  # 🔥 مهم برای فارسی
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
# Dataset
# ================================================================
def split_key(question: str) -> int:
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return int(h[:6], 16) % 100


def filter_by_split(example: Dict[str, Any], split_range: range) -> bool:
    q = example.get("question", "")
    return isinstance(q, str) and split_key(normalize_text(q)) in split_range


def preprocess_function(examples: Dict[str, list], min_score: float):
    texts = []
    for q, a, s in zip(
        examples["question"], examples["response"], examples["score_ratio"]
    ):
        if not (isinstance(q, str) and isinstance(a, str)):
            texts.append("")
            continue
        try:
            if not (min_score <= float(s) <= 1.0):
                texts.append("")
                continue
        except:
            texts.append("")
            continue

        q_norm = normalize_text(q)
        a_norm = normalize_text(a)
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
    print(f"\n📚 Loading dataset (sample={config.SAMPLE_RATIO*100:.1f}%)")
    raw = load_dataset("json", data_files={"train": config.DATA_FILE}, split="train")

    # 🔥 Sampling برای تست
    if config.SAMPLE_RATIO < 1.0:
        n_samples = int(len(raw) * config.SAMPLE_RATIO)
        raw = raw.shuffle(seed=42).select(range(n_samples))
        print(f"   Sampled: {len(raw):,} examples")

    datasets = {}
    for split_name, split_range in SPLIT_RANGES.items():
        ds = raw.filter(
            lambda ex: filter_by_split(ex, split_range),
            num_proc=4,
        )
        ds = ds.map(
            partial(preprocess_function, min_score=min_score),
            batched=True,
            remove_columns=raw.column_names,
            num_proc=4,
        )
        ds = ds.filter(lambda ex: len(ex.get("input_ids", [])) > 10, num_proc=4)

        if split_name == "train":
            ds = ds.shuffle(seed=42)

        datasets[split_name] = ds
        print(f"  {split_name}: {len(ds):,} examples")

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
        # Precision
        bf16=True,
        bf16_full_eval=True,
        # Logging - دقیق برای TensorBoard
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        logging_steps=config.LOGGING_STEPS,
        logging_first_step=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        # 🔥 Checkpointing - قوی
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_safetensors=True,  # امن‌تر
        # Efficiency
        dataloader_num_workers=2,  # کمتر برای 4070
        dataloader_pin_memory=True,
        gradient_checkpointing=config.USE_QLORA,
        optim="paged_adamw_8bit" if config.USE_QLORA else "adamw_torch",
        # Reporting
        report_to=["tensorboard"],
        run_name=f"gemma3_{phase_name}",
        # 🔥 Resume support
        resume_from_checkpoint=config.RESUME_FROM_CHECKPOINT,
    )


# ================================================================
# پیدا کردن آخرین Checkpoint
# ================================================================
def get_last_checkpoint(output_dir: str):
    """پیدا کردن آخرین checkpoint برای resume"""
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if not checkpoints:
        return None

    # Sort به step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    last_checkpoint = str(checkpoints[-1])
    print(f"\n🔄 Found checkpoint: {last_checkpoint}")
    return last_checkpoint


# ================================================================
# Training Loop با Safety
# ================================================================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n" + "=" * 70)
print("🚀 SAFE TRAINING START")
print(f"   Sample: {config.SAMPLE_RATIO*100:.0f}% | QLoRA: {config.USE_QLORA}")
print(f"   LoRA Rank: {config.LORA_R} (HIGH for Persian quality)")
print("=" * 70)

for i, phase in enumerate(config.CURRICULUM_PHASES, 1):
    print(f"\n{'='*70}")
    print(f"📖 PHASE {i}/{len(config.CURRICULUM_PHASES)}: {phase['name'].upper()}")
    print(f"{'='*70}")

    datasets = create_curriculum_dataset(phase["min_score"])
    args = get_training_args(phase["name"], phase["epochs"], phase["lr"])

    # 🔥 Checkpoint callback
    checkpoint_callback = SafeCheckpointCallback(config.BACKUP_DIR)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        callbacks=[checkpoint_callback],
    )

    # 🔥 Auto-resume
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
# 🔥 تست کیفیت فارسی
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
            repetition_penalty=1.5,  # جلوگیری از تکرار
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("پاسخ:")[-1].strip()
    print(f"\n❓ {q}")
    print(f"💬 {answer[:200]}...")  # اولین 200 کاراکتر

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print(f"📊 TensorBoard: tensorboard --logdir={config.OUTPUT_DIR}/logs")
print(f"💾 Backups: {config.BACKUP_DIR}")
print("=" * 70)
