import os
import torch
import huggingface_hub
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ====== Login ======
hf_token = ""
huggingface_hub.login()

# ====== Load Base Model ======
cptk = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(cptk)
tokenizer.padding_side = "right"  # important for CausalLM

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====== 4Bit Quantization ======
# b4_conf = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
# )

model = AutoModelForCausalLM.from_pretrained(
    cptk, device_map="auto", torch_dtype=torch.bfloat16
)
# model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# ===== loading dataset =====
dataset = load_dataset(
    "json",
)
