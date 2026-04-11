from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
from trl import SFTTrainer

import mlflow
import mlflow.pytorch
import mlflow.transformers
import os
import torch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "jackliu2006/car_knowledge"
MAX_SEQ_LENGTH = 256

# Maps each source column to a human-readable suffix used in the output dir name.
SOURCE_COLUMNS = {
    "gemini_output": "gemini3",
    "gpt_output": "gpt5",
}


def get_output_dir(source_suffix: str) -> str:
    return f"{MODEL_NAME.replace('/', '_')}_car_knowledge_finetuned_{source_suffix}"


def get_registered_model_name(source_suffix: str) -> str:
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks")
    model_name = f"{MODEL_NAME.replace('/', '-')}_{source_suffix}"
    if registry_uri == "databricks-uc":
        catalog = os.environ["MLFLOW_UC_CATALOG"]
        schema = os.environ["MLFLOW_UC_SCHEMA"]
        return f"{catalog}.{schema}.{model_name}"
    return model_name


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )


def get_peft_config() -> LoraConfig:
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def get_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        push_to_hub=True,
        gradient_checkpointing=True,
        report_to="none",  # MLflow logging is handled manually
    )


# ---------------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(use_quantization: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"dtype": "auto", "device_map": device_map}
    if use_quantization:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit quantization requires a CUDA-capable GPU.")
        model_kwargs["quantization_config"] = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _extract_text(out) -> str:
    if isinstance(out, dict):
        return out.get("text", "")
    if isinstance(out, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in out
        )
    return str(out)


def build_tokenize_fn(tokenizer):
    def tokenize_dataset(batch):
        instructions = batch["instruction"] if isinstance(batch["instruction"], list) else [batch["instruction"]]
        outputs = batch["output"] if isinstance(batch["output"], list) else [batch["output"]]

        formatted_texts = []
        for inst, out in zip(instructions, outputs):
            messages = [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": _extract_text(out)},
            ]
            formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))

        tokenized = tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return tokenize_dataset


def prepare_dataset(tokenizer, source_column: str) -> DatasetDict:
    dataset = load_dataset(DATASET_NAME, split="train")
    if source_column not in dataset.column_names:
        raise ValueError(f"Column '{source_column}' not found in dataset. Available: {dataset.column_names}")

    # Keep only the instruction and chosen source column, then rename to 'output'.
    columns_to_remove = [c for c in dataset.column_names if c not in ("instruction", source_column)]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.rename_column(source_column, "output")

    tokenized = dataset.map(
        build_tokenize_fn(tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized.train_test_split(test_size=0.2, seed=42)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(source_column: str, source_suffix: str) -> None:
    """Train and push a model for a single output column."""
    print(f"\n{'='*60}")
    print(f"Training with source column: {source_column}")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_dir = get_output_dir(source_suffix)
    model, tokenizer = load_model_and_tokenizer(use_quantization=False)
    dataset_dict = prepare_dataset(tokenizer, source_column)
    training_args = get_training_args(output_dir)
    peft_config = get_peft_config()

    run_name = f"{MODEL_NAME.replace('/', '_')}_{source_suffix}"
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "source_column": source_column,
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
        })

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_args,
        )
        trainer.train()

        # Log training metrics from trainer state
        for entry in trainer.state.log_history:
            step = entry.get("step")
            for key, value in entry.items():
                if key != "step" and isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)

        trainer.push_to_hub()
        mlflow.log_param("hub_repo", output_dir)

        # Log the model with the MLflow Transformers flavor so it appears under
        # the experiment's Models tab in Databricks and can be registered/served.
        # Merge LoRA adapter into the base model first so MLflow doesn't attempt
        # to resolve the local output dir as a HuggingFace Hub repository.
        merged_model = trainer.model.merge_and_unload()
        mlflow.transformers.log_model(
            transformers_model={"model": merged_model, "tokenizer": tokenizer},
            artifact_path="model",
            registered_model_name=get_registered_model_name(source_suffix),
        )

    print(f"Model pushed to HuggingFace: {output_dir}")


def train():
    load_dotenv()
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, training will run on CPU.")

    # Configure MLflow to use Databricks as the tracking server.
    # Requires MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME to be set in .env
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", "databricks"))
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    for source_column, suffix in SOURCE_COLUMNS.items():
        train_one(source_column, suffix)


if __name__ == "__main__":
    train()

