from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
import torch
torch.cuda.empty_cache()

dataset_name = "jackliu2006/car_knowledge"

dataset = load_dataset(dataset_name, split="train")
dataset = dataset.rename_column("gpt_output", "output")

model_name = "Qwen/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokennize_dataset(dataset):
    formatted_texts = []
    for inst, out in zip(dataset["instruction"], dataset["output"]):
        messages = [
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_texts.append(text)
    return tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokennize_dataset, batched=True,remove_columns=dataset.column_names)

def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="Qwen3.5_0.8B_car_knowledge_finetuned_gpt5",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    push_to_hub=True,
    gradient_accumulation_steps=16, # High accumulation to compensate for batch 1
    fp16=False, 
    bf16=True,  # Use bfloat16 if supported by the hardware
    gradient_checkpointing=True,    # CRITICAL: Recomputes activations instead of storing them
    optim="adamw_bnb_8bit",
)

from transformers import Trainer
dataset_dict = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

trainer = Trainer(
    model=model,
    args=training_args,     
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.push_to_hub()