import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import huggingface_hub
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# ========== تنظیمات اولیه ==========
hf_token = ""  # توکن خود را اینجا قرار دهید
huggingface_hub.login(token=hf_token)

# ========== بارگذاری مدل و توکنایزر ==========
cptk = "google/gemma-3-1b-pt"
tokenizer = AutoTokenizer.from_pretrained(cptk)
tokenizer.padding_side = "right"  # مهم برای causal LM

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# کوانتیزیشن 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 برای 40xx سریع‌تر است
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    cptk,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ========== تنظیمات LoRA ==========
lora_conf = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # بیشتر ماژول‌ها
    lora_dropout=0.05,  # کاهش dropout
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_conf)
model.print_trainable_parameters()

# ========== بارگذاری دیتاست ==========
dataset = load_dataset(
    "json", data_files={"train": "assets/qa_train.json", "test": "assets/qa_test.json"}
)


# ========== تابع پردازش دیتاست (بهینه‌شده) ==========
def formatting_func(example):
    """ترکیب input و output به فرمت مناسب"""
    text = f"{example['input']}{tokenizer.eos_token}{example['output']}{tokenizer.eos_token}"
    return {"text": text}


# اعمال formatting
dataset = dataset.map(formatting_func, remove_columns=dataset["train"].column_names)


# ========== تابع پردازش با tokenization ==========
def tokenize_function(examples):
    """توکنایز کردن با padding دینامیک"""
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,  # padding را data collator انجام می‌دهد
    )
    # labels برابر با input_ids (برای causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


# اعمال tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing dataset",
)

# Data collator ساده با padding دینامیک
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM (نه Masked LM)
    pad_to_multiple_of=8,  # برای کارایی بهتر
)

# ========== تنظیمات Training (بهینه شده برای 12GB) ==========
training_args = TrainingArguments(
    output_dir="./saved_models/gemma_qlora_optimized",
    # Batch settings
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # افزایش برای batch size مؤثر 8
    # Training duration
    num_train_epochs=3,
    max_steps=-1,
    # Optimization
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    # Precision
    bf16=True,  # بهتر از fp16 برای RTX 40xx
    bf16_full_eval=True,
    # Memory optimization
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",  # optimizer 8-bit
    max_grad_norm=0.3,
    # Logging
    logging_steps=10,
    logging_first_step=True,
    # Evaluation - کاهش فرکانس
    eval_strategy="steps",
    eval_steps=100,  # از 50 به 100
    eval_accumulation_steps=4,  # نگه‌داری کمتر در حافظه
    # Saving
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # Misc
    report_to="none",  # غیرفعال کردن wandb/tensorboard
    remove_unused_columns=False,
)

# ========== Callbacks ==========
callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

# ========== SFTTrainer ==========
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# ========== Training ==========
print("🚀 شروع Training...")
print(f"📊 تعداد نمونه‌های Train: {len(dataset['train'])}")
print(f"📊 تعداد نمونه‌های Test: {len(dataset['test'])}")

torch.cuda.empty_cache()
trainer.train()

# ========== ذخیره مدل ==========
print("💾 در حال ذخیره مدل...")
trainer.save_model("./saved_models/gemma_final_model")
tokenizer.save_pretrained("./saved_models/gemma_final_model")

print("✅ Fine-tuning کامل شد!")
