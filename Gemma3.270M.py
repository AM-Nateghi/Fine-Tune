import os
import hashlib
import torch
from typing import Dict, Any, Optional
from functools import partial

import huggingface_hub
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ================================================================
# Configuration
# ================================================================
class Config:
    # Model
    MODEL_NAME = "google/gemma-3-270m"
    USE_QLORA = False  # 🔧 Toggle QLoRA on/off

    # Data
    DATA_FILE = "assets/dataset_output.filtered.jsonl"
    MAX_LENGTH = 512

    # QLoRA settings (optimized for 240M tokens)
    LORA_R = 64  # Higher rank for better learning on large dataset
    LORA_ALPHA = 128  # alpha = 2 * r is common practice
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # Attention
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP
    ]

    # 4-bit quantization
    LOAD_IN_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
    BNB_4BIT_QUANT_TYPE = "nf4"  # Normal Float 4
    BNB_4BIT_USE_DOUBLE_QUANT = True  # Nested quantization for memory

    # Training
    CURRICULUM_PHASES = [
        {"name": "phase1", "min_score": 0.8, "epochs": 1, "lr": 2e-4},
        {"name": "phase2", "min_score": 0.7, "epochs": 1, "lr": 1e-4},
    ]

    BATCH_SIZE = 4  # Per device
    GRADIENT_ACCUMULATION = 2  # Effective batch = 16
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 0.3  # Lower for QLoRA stability

    # Checkpointing
    SAVE_STEPS = 100
    EVAL_STEPS = 100
    LOGGING_STEPS = 25
    SAVE_TOTAL_LIMIT = 2

    # Output
    OUTPUT_DIR = "./outputs/gemma3_qlora"


config = Config()

# ================================================================
# Environment Setup
# ================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hf_token = ""
if hf_token:
    huggingface_hub.login(token=hf_token)


# ================================================================
# Quantization Config
# ================================================================
def get_bnb_config():
    """4-bit quantization configuration for QLoRA"""
    if not config.USE_QLORA:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )


# ================================================================
# Model & Tokenizer Setup
# ================================================================
def setup_model_and_tokenizer():
    """Load model with optional QLoRA"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model with QLoRA: {config.USE_QLORA}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    # Activate FlashAttention2
    model = model.to(memory_efficient_attention=True)

    # Apply QLoRA if enabled
    if config.USE_QLORA:
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Using full fine-tuning (no QLoRA)")
        model.gradient_checkpointing_enable()

    return model, tokenizer


model, tokenizer = setup_model_and_tokenizer()

# ================================================================
# Text Normalization
# ================================================================
ARABIC_TO_PERSIAN = {"\u064a": "ی", "\u0643": "ک", "\u06cc": "ی"}
PERSIAN_DIGITS_TO_EN = {f"۰۱۲۳۴۵۶۷۸۹"[i]: str(i) for i in range(10)}


def normalize_text(text: str) -> str:
    """Normalize Persian text"""
    if not isinstance(text, str):
        return ""
    for ar, fa in ARABIC_TO_PERSIAN.items():
        text = text.replace(ar, fa)
    for fa, en in PERSIAN_DIGITS_TO_EN.items():
        text = text.replace(fa, en)
    text = " ".join(text.split())
    text = text.replace(" ؟", "؟").replace("،", "، ")
    return " ".join(text.split())


# ================================================================
# Dataset Processing
# ================================================================
def split_key(question: str) -> int:
    """Deterministic hash-based split"""
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return int(h[:6], 16) % 100


def filter_by_split(example: Dict[str, Any], split_range: range) -> bool:
    """Filter by train/val/test split"""
    q = example.get("question", "")
    if not isinstance(q, str):
        return False
    return split_key(normalize_text(q)) in split_range


def preprocess_function(examples: Dict[str, list], min_score: float) -> Dict[str, list]:
    """Tokenize Q&A pairs"""
    texts = []

    for q, a, s in zip(
        examples["question"], examples["response"], examples["score_ratio"]
    ):
        # Validate
        if not (isinstance(q, str) and isinstance(a, str)):
            texts.append("")
            continue

        try:
            score = float(s)
            if not (min_score <= score <= 1.0):
                texts.append("")
                continue
        except:
            texts.append("")
            continue

        # Normalize and format
        q_norm = normalize_text(q)
        a_norm = normalize_text(a)
        text = f"سوال: {q_norm}\nپاسخ: {a_norm}"
        texts.append(text)

    # Tokenize
    return tokenizer(
        texts,
        truncation=True,
        max_length=config.MAX_LENGTH,
        padding=False,
    )


# Split ranges
SPLIT_RANGES = {
    "train": range(0, 95),
    "validation": range(95, 97),
    "test": range(97, 100),
}


def create_curriculum_dataset(min_score: float):
    """Create dataset for a curriculum phase"""
    print(f"\nCreating dataset with min_score={min_score}...")

    # Load base dataset
    raw = load_dataset("json", data_files={"train": config.DATA_FILE}, split="train")

    datasets = {}
    for split_name, split_range in SPLIT_RANGES.items():
        print(f"  Processing {split_name} split...")

        # Filter by split and score
        ds = raw.filter(
            lambda ex: filter_by_split(ex, split_range),
            num_proc=4,
        )

        # Tokenize
        ds = ds.map(
            partial(preprocess_function, min_score=min_score),
            batched=True,
            remove_columns=raw.column_names,
            num_proc=4,
            desc=f"Tokenizing {split_name}",
        )

        # Remove empty
        ds = ds.filter(lambda ex: len(ex.get("input_ids", [])) > 10, num_proc=4)

        if split_name == "train":
            ds = ds.shuffle(seed=42)

        datasets[split_name] = ds
        print(f"    {split_name}: {len(ds)} examples")

    return datasets


# ================================================================
# Training Arguments
# ================================================================
def get_training_args(phase_name: str, num_epochs: int, learning_rate: float):
    """Generate training arguments for a phase"""
    return TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR}/{phase_name}",
        num_train_epochs=num_epochs,
        # Batch sizes
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        # Optimization
        learning_rate=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        warmup_ratio=config.WARMUP_RATIO,
        # Mixed precision
        bf16=True,
        bf16_full_eval=True,
        # Logging
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
        # Efficiency
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=not config.USE_QLORA,  # QLoRA handles this
        optim="paged_adamw_8bit" if config.USE_QLORA else "adamw_torch",
        # Reporting
        report_to=["tensorboard"],
        run_name=f"gemma3_{phase_name}",
    )


# ================================================================
# Curriculum Training Loop
# ================================================================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n" + "=" * 70)
print("STARTING CURRICULUM LEARNING")
print("=" * 70)

for phase in config.CURRICULUM_PHASES:
    print(f"\n{'='*70}")
    print(f"PHASE: {phase['name'].upper()} (min_score={phase['min_score']})")
    print(f"{'='*70}")

    # Create dataset
    datasets = create_curriculum_dataset(min_score=phase["min_score"])

    # Setup trainer
    args = get_training_args(
        phase_name=phase["name"],
        num_epochs=phase["epochs"],
        learning_rate=phase["lr"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
    )

    # Train
    print(f"\nStarting training for {phase['epochs']} epochs...")
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\n{phase['name'].upper()} Results:")
    print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"  Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

    # Save
    trainer.save_model()

# ================================================================
# Final Test Evaluation
# ================================================================
print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_ds = create_curriculum_dataset(min_score=0.7)["test"]
test_results = trainer.evaluate(eval_dataset=test_ds)
print(f"Test Loss: {test_results['eval_loss']:.4f}")
print(f"Test Perplexity: {torch.exp(torch.tensor(test_results['eval_loss'])):.2f}")

# ================================================================
# Save Final Model
# ================================================================
final_path = f"{config.OUTPUT_DIR}/final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"\nFinal model saved to: {final_path}")

# ================================================================
# Inference Test
# ================================================================
print("\n" + "=" * 70)
print("INFERENCE TEST")
print("=" * 70)

model.eval()
test_questions = [
    "آیا خواب های ما انسان ها واقعی هستن؟ چطور میتونیم اثباتش کنیم؟",
    "چرا خدا وجود داره؟ خلاصه بگو",
]

for question in test_questions:
    prompt = f"سوال: {question}\nپاسخ:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nQ: {question}")
    print(f"A: {response.split('پاسخ:')[-1].strip()}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
