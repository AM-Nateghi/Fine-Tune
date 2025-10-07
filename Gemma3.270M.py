import os
import hashlib
import torch
from typing import Dict, Any
from functools import partial

import huggingface_hub
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ================================================================
# Environment & Login
# ================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hf_token = ""
if hf_token:
    huggingface_hub.login(token=hf_token)
    print(f"Welcome {huggingface_hub.whoami()['name']}!")

# ================================================================
# Model & Tokenizer
# ================================================================
MODEL_NAME = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    attn_implementation="eager",  # or "flash_attention_2" if available
    trust_remote_code=True,
)

# Optional: Use LoRA for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================================================================
# Persian Text Normalization
# ================================================================
ARABIC_TO_PERSIAN = {"\u064a": "ی", "\u0643": "ک", "\u06cc": "ی"}
PERSIAN_DIGITS_TO_EN = {
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Arabic -> Persian
    for ar, fa in ARABIC_TO_PERSIAN.items():
        text = text.replace(ar, fa)
    # Persian digits -> English
    for fa, en in PERSIAN_DIGITS_TO_EN.items():
        text = text.replace(fa, en)
    # Clean spacing
    text = " ".join(text.split())
    text = text.replace(" ؟", "؟").replace("،", "، ").replace(" ?", "?")
    return " ".join(text.split())


# ================================================================
# Dataset Loading & Preprocessing
# ================================================================
DATA_FILE = "assets/dataset_output.filtered.jsonl"


def split_key(question: str) -> int:
    """Deterministic hash-based splitting"""
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return int(h[:6], 16) % 100


def preprocess_function(
    examples: Dict[str, list], min_score: float = 0.0
) -> Dict[str, list]:
    """Tokenize and format examples"""
    texts = []

    for q, a, s in zip(
        examples["question"], examples["response"], examples["score_ratio"]
    ):
        # Validate and normalize
        if not isinstance(q, str) or not isinstance(a, str):
            texts.append("")
            continue

        try:
            score = float(s)
            if score < min_score or score < 0.0 or score > 1.0:
                texts.append("")
                continue
        except:
            texts.append("")
            continue

        q_norm = normalize_text(q)
        a_norm = normalize_text(a)

        # Format as instruction-response pair
        text = f"سوال: {q_norm}\nپاسخ: {a_norm}"
        texts.append(text)

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding=False,  # Dynamic padding in collator
    )

    return tokenized


def filter_by_split(example: Dict[str, Any], split_range: range) -> bool:
    """Filter examples by split using hash"""
    q = example.get("question", "")
    if not isinstance(q, str):
        return False
    bucket = split_key(normalize_text(q))
    return bucket in split_range


# Load dataset (NON-STREAMING for proper length calculation)
raw_dataset = load_dataset(
    "json",
    data_files={"train": DATA_FILE},
    split="train",
)

# Split ranges
SPLIT_RANGES = {
    "train": range(0, 95),
    "validation": range(95, 97),
    "test": range(97, 100),
}


def create_phase_dataset(min_score: float):
    """Create train/val/test splits for a curriculum phase"""
    datasets = {}

    for split_name, split_range in SPLIT_RANGES.items():
        # Filter by split
        ds = raw_dataset.filter(
            lambda ex: filter_by_split(ex, split_range),
            num_proc=4,  # Parallel processing
        )

        # Preprocess with min_score threshold
        ds = ds.map(
            partial(preprocess_function, min_score=min_score),
            batched=True,
            remove_columns=["question", "response", "score_ratio"],
            num_proc=4,
        )

        # Filter out empty examples
        ds = ds.filter(
            lambda ex: len(ex.get("input_ids", [])) > 0,
            num_proc=4,
        )

        if split_name == "train":
            ds = ds.shuffle(seed=42)

        datasets[split_name] = ds

    return datasets


# Phase 1: High-quality examples (score >= 0.8)
print("Creating Phase 1 datasets...")
phase1_datasets = create_phase_dataset(min_score=0.8)
print(f"Phase 1 - Train: {len(phase1_datasets['train'])} examples")
print(f"Phase 1 - Val: {len(phase1_datasets['validation'])} examples")

# Phase 2: Medium-quality examples (score >= 0.7)
print("\nCreating Phase 2 datasets...")
phase2_datasets = create_phase_dataset(min_score=0.7)
print(f"Phase 2 - Train: {len(phase2_datasets['train'])} examples")
print(f"Phase 2 - Val: {len(phase2_datasets['validation'])} examples")

# ================================================================
# Data Collator (using built-in HuggingFace collator)
# ================================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)


# ================================================================
# Training Arguments
# ================================================================
def get_training_args(output_dir: str, num_train_epochs: int = 3):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        # Optimization
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # Mixed precision
        bf16=True,
        bf16_full_eval=True,
        # Logging & Checkpointing
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        # Best model tracking
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Misc
        report_to=["tensorboard"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Disable cache for training
        use_cpu=False,
    )


# ================================================================
# Phase 1 Training (High-quality data)
# ================================================================
print("\n" + "=" * 60)
print("PHASE 1: Training on high-quality examples (score >= 0.8)")
print("=" * 60)

args_phase1 = get_training_args(
    output_dir="./outputs/gemma3_phase1",
    num_train_epochs=2,
)

trainer_phase1 = Trainer(
    model=model,
    args=args_phase1,
    train_dataset=phase1_datasets["train"],
    eval_dataset=phase1_datasets["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer_phase1.train()
trainer_phase1.save_model()
eval_results_1 = trainer_phase1.evaluate()
print(f"Phase 1 Eval Loss: {eval_results_1['eval_loss']:.4f}")

# ================================================================
# Phase 2 Training (Medium-quality data)
# ================================================================
print("\n" + "=" * 60)
print("PHASE 2: Training on medium-quality examples (score >= 0.7)")
print("=" * 60)

args_phase2 = get_training_args(
    output_dir="./outputs/gemma3_phase2",
    num_train_epochs=3,
)

trainer_phase2 = Trainer(
    model=model,  # Continue from phase 1
    args=args_phase2,
    train_dataset=phase2_datasets["train"],
    eval_dataset=phase2_datasets["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer_phase2.train()
trainer_phase2.save_model()
eval_results_2 = trainer_phase2.evaluate()
print(f"Phase 2 Eval Loss: {eval_results_2['eval_loss']:.4f}")

# ================================================================
# Final Test Evaluation
# ================================================================
print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_results = trainer_phase2.evaluate(eval_dataset=phase2_datasets["test"])
print(f"Test Loss: {test_results['eval_loss']:.4f}")
print(f"Test Perplexity: {torch.exp(torch.tensor(test_results['eval_loss'])):.4f}")

# ================================================================
# Save Final Model
# ================================================================
final_model_path = "./outputs/gemma3_final"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")

# ================================================================
# Inference Example
# ================================================================
print("\n" + "=" * 60)
print("INFERENCE TEST")
print("=" * 60)

model.eval()
test_question = "چرا خواب های ما واقعی هستند؟"
prompt = f"سوال: {test_question}\nپاسخ:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Question: {test_question}")
print(f"Response: {response}")
