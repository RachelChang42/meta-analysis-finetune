import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, DataCollatorForLanguageModeling

MODEL_PATH = "/root/autodl-tmp/NLP_Project/models/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_FILE = "/root/autodl-tmp/NLP_Project/data/folds/train_fold_1.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/NLP_Project/outputs/qwen25_7b_lora_fold1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_chat(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }

dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train"
)

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False
)

def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=4096,
        padding=False
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished. LoRA adapter saved to:", OUTPUT_DIR)
