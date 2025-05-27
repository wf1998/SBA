import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from geomloss import SamplesLoss

HF_TOKEN = ''

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH = "instruction_data.jsonl"
OUTPUT_DIR = "lora-llama32-intent-extractor"
MAX_LENGTH = 512

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=HF_TOKEN )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    user = example["user"].strip()
    assistant = example["assistant"].strip()
    prompt = f"{user}\n"
    labels = assistant
    return {"prompt": prompt, "labels": labels}

dataset = dataset.map(format_example)

def tokenize(example):
    prompt = example["prompt"]
    labels = example["labels"]

    text = prompt + labels
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"])
    inputs["labels"] = [-100] * prompt_len + inputs["input_ids"][prompt_len:]
    return inputs

dataset = dataset.map(tokenize, batched=False)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 适配Llama3结构
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)
model = get_peft_model(model, lora_config)

sinkhorn_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05)

def get_token_embeddings(input_ids, model, tokenizer):
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=(input_ids != tokenizer.pad_token_id),
            output_hidden_states=True
        )
    return outputs.hidden_states[-1][0]  # shape: [seq_len, hidden_dim]

def get_assistant_token_indices(labels):
    valid = (labels != -100).nonzero(as_tuple=True)[0]
    return valid.tolist() if len(valid) > 0 else []

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        lm_loss = outputs.loss

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        sinkhorn_total = 0.0
        batch_count = 0

        for b in range(input_ids.size(0)):
            indices = get_assistant_token_indices(labels[b])
            if not indices:
                continue
            if len(indices) > 64:
                indices = indices[:64]  # 限制长度防止OOM
            target_ids = labels[b][indices].unsqueeze(0)
            pred_ids = input_ids[b][indices].unsqueeze(0)
            target_emb = get_token_embeddings(target_ids, model, tokenizer)
            pred_emb = get_token_embeddings(pred_ids, model, tokenizer)
            with torch.cuda.amp.autocast(enabled=False):
                s_loss = sinkhorn_loss_fn(pred_emb.float(), target_emb.float())
            sinkhorn_total += s_loss
            batch_count += 1
        if batch_count > 0:
            sinkhorn_total = sinkhorn_total / batch_count
            lm_loss = lm_loss + 0.1 * sinkhorn_total

        return (lm_loss, outputs) if return_outputs else lm_loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = CustomTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
