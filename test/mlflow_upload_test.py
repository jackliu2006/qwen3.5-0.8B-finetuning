"""
Quick smoke test: loads the smallest available fine-tuned adapter from disk,
logs it to Databricks MLflow, and registers it in Unity Catalog.
Run with:  uv run test/mlflow_upload_test.py
"""

import os
import mlflow
import mlflow.transformers
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

load_dotenv()

# ---------------------------------------------------------------------------
# Config — points at an already-saved local adapter checkpoint to avoid
# running a full training job just to test the upload path.
# ---------------------------------------------------------------------------
LOCAL_ADAPTER_DIR = "Qwen_Qwen3-0.6B_car_knowledge_finetuned_gemini3"
TEST_REGISTERED_NAME = (
    f"{os.environ['MLFLOW_UC_CATALOG']}."
    f"{os.environ['MLFLOW_UC_SCHEMA']}."
    f"{LOCAL_ADAPTER_DIR.replace('/', '-')}_upload_test"
)

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", "databricks"))
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])


def main():
    print(f"Loading adapter from: {LOCAL_ADAPTER_DIR}")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoPeftModelForCausalLM.from_pretrained(LOCAL_ADAPTER_DIR, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_ADAPTER_DIR)

    # Merge LoRA adapter weights into the base model so MLflow gets a plain
    # transformers model and doesn't try to resolve the local path as a HF repo.
    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print("Starting MLflow run...")
    with mlflow.start_run(run_name="upload_test"):
        mlflow.log_param("test", True)
        mlflow.log_param("adapter_dir", LOCAL_ADAPTER_DIR)

        print(f"Logging model to Databricks MLflow and registering as: {TEST_REGISTERED_NAME}")
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            registered_model_name=TEST_REGISTERED_NAME,
        )

    print("Done. Check your Databricks experiment and Models tab.")


if __name__ == "__main__":
    main()
