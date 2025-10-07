import os
import math
import torch
import random
import hashlib
from typing import Dict, List, Iterable, Iterator, Any

import huggingface_hub
from datasets import load_dataset, IterableDatasetDict, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
)

# ================================================================
# Enviroment & Login into HuggingFace
# ================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hf_token = ""  # if you need to login
if hf_token:
    huggingface_hub.login(token=hf_token)
    print(f"Welcome {huggingface_hub.whoami()['name']}!")

# ================================================================
# Model & Tokenizer
# ================================================================
cptk = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(cptk)
tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cptk, device_map="auto", torch_dtype=torch.bfloat16
)
model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ================================================================
# Normalization utilities for Persian text
# ================================================================
ARABIC_TO_PERSIAN = {
    "\u064a": "ی",  # ي -> ی
    "\u0643": "ک",  # ك -> ک
    "\u06cc": "ی",  # ی -> ی (یکنواخت‌سازی)
}
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
    # Arabic -> Persian letters
    out = "".join(ARABIC_TO_PERSIAN.get(chunk, chunk) for chunk in text)
    # Persian digits -> English
    out = "".join(PERSIAN_DIGITS_TO_EN.get(ch, ch) for ch in out)
    # Normalize spaces and punctuation spacing
    out = out.replace("\u200c", "\u200c")  # keep ZWNJ as-is, but you could refine rules
    # Trim redundant spaces
    out = " ".join(out.split())
    # Fix common punctuation spacing: remove space before ?, add space after ,
    out = out.replace(" ؟", "؟").replace("،", "، ").replace(" ?", "?")
    out = " ".join(out.split())  # re-trim
    return out


def valid_example(ex):
    return (
        isinstance(ex, dict)
        and "question" in ex
        and "response" in ex
        and "score_ratio" in ex
    )


# ================================================================
# Streaming dataset with split via deterministic hashing
# ================================================================
DATA_FILE = "assets/dataset_output.filtered.jsonl"  # JSONL with keys: question, response, score_ratio

raw_stream = load_dataset("json", data_files={"all": DATA_FILE}, streaming=True)["all"]


def split_key(question: str) -> int:
    # Deterministic split based on SHA1 hash of question
    h = hashlib.sha1((question or "").encode("utf-8")).hexdigest()
    return int(h[:6], 16) % 100  # 0,1,...,99


def filter_stream(
    stream: Iterable[Dict[str, Any]], min_score: float, split_ranges: Dict[str, range]
) -> IterableDatasetDict:
    def gen_for_split(target_split: str) -> Iterator[Dict[str, Any]]:
        r = split_ranges[target_split]
        for ex in stream:
            q = ex.get("question", "")
            a = ex.get("response", "")
            s = ex.get("score_ratio", None)

            if not isinstance(q, str) or not isinstance(a, str) or s is None:
                continue

            # Normalize
            q_norm = normalize_text(q)
            a_norm = normalize_text(a)

            # Enforce score bounds and curriculum filter
            try:
                s_float = float(s)
            except:
                continue
            if s_float < 0.0 or s_float > 1.0:
                continue
            if s_float < min_score:
                continue

            # split routing
            bucket = split_key(q_norm)
            if bucket in r:
                yield {
                    "question": q_norm,
                    "response": a_norm,
                    "score_ratio": s_float,
                }

    datasets = {}
    for name in split_ranges.keys():
        datasets[name] = IterableDataset.from_generator(lambda n=name: gen_for_split(n))
    return IterableDatasetDict(datasets)


# Define split ranges: 95/2/3
split_ranges = {
    "train": range(0, 95),
    "validation": range(95, 97),
    "test": range(97, 100),
}

# Curriculum phase 1: min_score = 0.8
stream_phase1 = filter_stream(raw_stream, min_score=0.8, split_ranges=split_ranges)
stream_phase1["train"] = stream_phase1["train"].filter(valid_example)
# Phase 2: min_score = 0.7
stream_phase2 = filter_stream(raw_stream, min_score=0.7, split_ranges=split_ranges)
stream_phase2["train"] = stream_phase2["train"].filter(valid_example)

# Optional shuffle buffers for streaming
stream_phase1["train"] = stream_phase1["train"].shuffle(buffer_size=1e4)
stream_phase2["train"] = stream_phase2["train"].shuffle(buffer_size=1e4)


# ================================================================
# Custom Data Collator with dynamic padding and label masking
# ================================================================
class QADataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []
        weights_batch = []

        for ex in features:
            if "question" not in ex or "response" not in ex:
                print(f"Malformed example: {ex}")
                continue
            question = ex["question"]
            answer = ex["response"]
            score = float(ex.get("score_ratio", 1.0))
            # Clamp score ratio to [0,1]
            score = max(0.0, min(1.0, score))

            # Build prompt: "سوال: ... \nپاسخ:"
            prompt = f"سوال: {question}\nپاسخ:"
            tok_prompt = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )
            tok_answer = self.tokenizer(
                answer + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )

            # Compos sequence: [prompt]+[answer]
            input_ids = tok_prompt["input_ids"] + tok_answer["input_ids"]
            attention_mask = [1] * len(input_ids)

            # Truncate to max_length from the end
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:, self.max_length]
                attention_mask = attention_mask[:, self.max_length]

            # Labels: mask prompt tokens as -100 train only on answer tokens
            # Number of prompt tokens that survived truncation:
            prompt_len = min(len(tok_prompt["input_ids"], len(input_ids)))
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            # Dynamic padding to the longest in batch will be applied by padding in Trainer;
            # but we can do manual right-padding to max in-batch length if needed.
            input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_batch.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            weights_batch.append(torch.tensor(score, dtype=torch.bfloat16))

        if not weights_batch:
            raise ValueError("Batch with only malformed examples encountered!")

        # Pad to longest length in batch (dynamic padding)
        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch,
            },
            padding=True,
            return_tensors="pt",
        )
        # Attach weights (no padding needed; one weight per example)
        batch["weights"] = torch.stack(weights_batch)
        return batch


collator = QADataCollator(tokenizer, max_length=1024)


# ================================================================
# Custom Trainer to apply per-example weights in loss
# ================================================================
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        weights = inputs.get("weights")
        outputs = model(**{k: v for k, v in inputs.items() if k not in ["weights"]})
        logits = outputs.get("logits")

        # Shift for causal LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        # (batch, seq)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        per_token_loss = per_token_loss.view(shift_labels.size(0), shift_labels.size(1))

        # Mask valid tokens
        mask = (shift_labels != -100).to(per_token_loss.dtype)
        per_example_loss = (per_token_loss * mask).sum(dim=1) / (
            mask.sum(dim=1).clamp(min=1)
        )

        if weights is not None:
            # Scale loss per example by score_ratio (already clamped 0..1)
            per_example_loss = per_example_loss * weights.to(per_example_loss.dtype)

        loss = per_example_loss.mean()
        return (loss, outputs) if return_outputs else loss


# ================================================================
# Training arguments
# ================================================================
def make_args(output_dir: str, max_steps: int = 5000):
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch size ~16
        bf16=True,  # compute bf16
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # ~5% warmup
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=800,
        save_strategy="steps",
        save_steps=800,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )


# ================================================================
# Build Dataloaders (streaming) for each phase
# ================================================================
def make_phase_datasets(stream_dict: IterableDatasetDict):
    return {
        "train": stream_dict["train"],
        "eval": stream_dict["validation"],
        "test": stream_dict["test"],
    }


phase1_data = make_phase_datasets(stream_phase1)
phase2_data = make_phase_datasets(stream_phase2)


# ================================================================
# Train: Curriculum Phase 1 (score_ratio >= 0.8)
# ================================================================
args_phase1 = make_args(output_dir="./assets/gemma3.270m.phase1", max_steps=5000)

trainer_phase1 = WeightedLossTrainer(
    model=model,
    args=args_phase1,
    train_dataset=phase1_data["train"],
    eval_dataset=phase1_data["eval"],
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

train_result_1 = trainer_phase1.train()
trainer_phase1.save_model()  # saves to output_dir

# Optionally evaluate phase 1
eval_result_1 = trainer_phase1.evaluate()

# ================================================================
# Train: Main Phase 2 (score_ratio >= 0.7), continue from phase1
# ================================================================
args_phase2 = make_args(output_dir="./assets/gemma3.270m.phase2", max_steps=5000)

trainer_phase2 = WeightedLossTrainer(
    model=trainer_phase1,  # continue training same model
    args=args_phase2,
    train_dataset=phase2_data["train"],
    eval_dataset=phase2_data["eval"],
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

train_result_2 = trainer_phase2.train()
trainer_phase2.save_model()
eval_result_2 = trainer_phase2.evaluate()

# ================================================================
# Final test evaluation (optional)
# ================================================================
test_metrics = trainer_phase2.evaluate(eval_dataset=phase2_data["test"])
print("Test metrics:", test_metrics)
