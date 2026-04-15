"""
Evaluate registered Qwen3-0.6B fine-tuned models.

Uses:
  mlflow.genai.evaluate  — for LLM-as-a-Judge metrics (answer_correctness,
                           answer_relevance, faithfulness) via Gemini judge
  mlflow.models.evaluate — for traditional lexical metrics (ROUGE, BLEU)

Required environment variables:
  MLFLOW_TRACKING_URI       - Databricks MLflow tracking server
  MLFLOW_REGISTRY_URI       - e.g. "databricks-uc" for Unity Catalog
  MLFLOW_EXPERIMENT_NAME    - experiment to log eval runs under
  MLFLOW_UC_CATALOG         - Unity Catalog catalog (if using UC)
  MLFLOW_UC_SCHEMA          - Unity Catalog schema (if using UC)
  NEXUS_BASE_URL            - Nexus gateway endpoint (same as prepare-dataset)
  NEXUS_API_KEY             - Nexus API key (same as prepare-dataset)

Run with:
  uv run train_eval/eval.py
"""

import os
import re
from datetime import datetime

import mlflow
import mlflow.deployments
import mlflow.genai
import mlflow.metrics
import mlflow.models
import pandas as pd
from databricks.sdk import WorkspaceClient
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from mlflow.genai.scorers import scorer

# ---------------------------------------------------------------------------
# Configuration (mirrors sft.py)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "jackliu2006/car_knowledge"
GEMINI_MODEL = "gemini-3.1-pro-preview"

SOURCE_SUFFIXES = {
    "gemini_output": "gemini3",
 #   "gpt_output": "gpt5",
}

ACCEPTANCE_THRESHOLDS = {
    "rougeL/v1/f1_score": 0.35,
    "bleu_score/v1/scores": 15.0,
    "answer_correctness/v1/mean": 3.5,
    "answer_relevance/v1/mean": 3.5,
    "faithfulness/v1/mean": 4.0,
}

# ---------------------------------------------------------------------------
# Judge prompts (1–5 scale)
# ---------------------------------------------------------------------------

def _build_gemini() -> ChatGoogleGenerativeAI:
    """Instantiate the same Gemini model used in prepare-dataset."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        client_options={"api_endpoint": os.getenv("NEXUS_BASE_URL")},
        google_api_key=os.getenv("NEXUS_API_KEY"),
        temperature=0.0,
    )


def _parse_score(text: str) -> float:
    """Extract first integer 1-5 from model response."""
    match = re.search(r"[1-5]", text.strip())
    return float(match.group()) if match else 3.0


# ---------------------------------------------------------------------------
# mlflow.genai custom scorers (LLM-as-a-Judge via Gemini)
# The @scorer decorator wraps a row-level function; mlflow.genai.evaluate
# calls it for every row passing inputs, outputs, targets as keyword args.
# ---------------------------------------------------------------------------

# Global counter for progress tracking in scorers
_eval_counter = {"current": 0, "total": 0, "metric_name": ""}

@scorer
def answer_correctness(inputs, outputs, targets):
    """Factual correctness of model output vs reference (1–5)."""
    _eval_counter["current"] += 1
    if _eval_counter["current"] % 5 == 0 or _eval_counter["current"] == _eval_counter["total"]:
        print(f"  [{_eval_counter['metric_name']}] Evaluated {_eval_counter['current']}/{_eval_counter['total']} samples...")
    
    prompt = f"""\
You are an expert evaluator. Score how factually correct the answer is compared to the reference.
Score from 1 (completely wrong) to 5 (perfectly correct).
Respond with only a single integer between 1 and 5.

Question: {inputs}
Reference answer: {targets}
Model answer: {outputs}

Score:"""
    try:
        return _parse_score(_build_gemini().invoke([HumanMessage(content=prompt)]).content)
    except Exception as e:
        print(f"  Warning: judge call failed ({e}), defaulting to 3.0")
        return 3.0


@scorer
def answer_relevance(inputs, outputs, targets):  # noqa: ARG001
    """Relevance of model output to the question (1–5)."""
    _eval_counter["current"] += 1
    if _eval_counter["current"] % 5 == 0 or _eval_counter["current"] == _eval_counter["total"]:
        print(f"  [{_eval_counter['metric_name']}] Evaluated {_eval_counter['current']}/{_eval_counter['total']} samples...")
    
    prompt = f"""\
You are an expert evaluator. Score how relevant and on-topic the answer is to the question.
Score from 1 (completely irrelevant) to 5 (perfectly relevant).
Respond with only a single integer between 1 and 5.

Question: {inputs}
Model answer: {outputs}

Score:"""
    try:
        return _parse_score(_build_gemini().invoke([HumanMessage(content=prompt)]).content)
    except Exception as e:
        print(f"  Warning: judge call failed ({e}), defaulting to 3.0")
        return 3.0


@scorer
def faithfulness(inputs, outputs, targets):  # noqa: ARG001
    """Faithfulness / no hallucination vs reference (1–5)."""
    _eval_counter["current"] += 1
    if _eval_counter["current"] % 5 == 0 or _eval_counter["current"] == _eval_counter["total"]:
        print(f"  [{_eval_counter['metric_name']}] Evaluated {_eval_counter['current']}/{_eval_counter['total']} samples...")
    
    prompt = f"""\
You are an expert evaluator. Score how faithful the answer is to the reference — does it avoid \
hallucination or unsupported claims?
Score from 1 (many hallucinations) to 5 (fully faithful).
Respond with only a single integer between 1 and 5.

Reference: {targets}
Model answer: {outputs}

Score:"""
    try:
        return _parse_score(_build_gemini().invoke([HumanMessage(content=prompt)]).content)
    except Exception as e:
        print(f"  Warning: judge call failed ({e}), defaulting to 3.0")
        return 3.0


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def get_latest_model_version(model_name: str) -> int | None:
    """Return the latest registered version number for the given model."""
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        return max(int(v.version) for v in versions)
    except Exception as e:
        print(f"Error fetching model version: {e}")
        return None


def resolve_serving_endpoint_name(model_name: str, version: int) -> str:
    """Discover serving endpoint name from Databricks metadata.

    Matches by served entity name and prefers exact version match.
    """
    ws_client = WorkspaceClient()
    candidates = []

    for endpoint in ws_client.serving_endpoints.list():
        endpoint_name = getattr(endpoint, "name", None)
        if not endpoint_name:
            continue

        # Get full endpoint details to access served entities reliably.
        details = ws_client.serving_endpoints.get(name=endpoint_name)
        config = getattr(details, "config", None)
        served_entities = getattr(config, "served_entities", None) or []

        for entity in served_entities:
            entity_name = getattr(entity, "entity_name", None)
            entity_version = getattr(entity, "entity_version", None)
            if entity_name != model_name:
                continue

            try:
                entity_version_int = int(entity_version) if entity_version is not None else -1
            except (TypeError, ValueError):
                entity_version_int = -1

            candidates.append((entity_version_int, endpoint_name))

    if not candidates:
        raise ValueError(
            f"No Databricks serving endpoint found for model '{model_name}'. "
            "Please deploy the model endpoint first."
        )

    # Prefer exact version match if available.
    for entity_version_int, endpoint_name in candidates:
        if entity_version_int == version:
            return endpoint_name

    # Fallback to newest available endpoint for the model.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def query_serving_endpoint(endpoint_name: str, texts: list[str], show_all_predictions: bool = False) -> list[str]:
    """Call the Databricks serving endpoint and return predictions as a list of strings."""
    print(f"  Querying endpoint '{endpoint_name}' for {len(texts)} predictions...")
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    
    # For large batches, make requests in chunks and show progress
    batch_size = 10
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if len(texts) > batch_size:
            print(f"    Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({i+1}-{min(i+len(batch), len(texts))}/{len(texts)})...")
        
        response = deploy_client.predict(
            endpoint=endpoint_name,
            inputs={"inputs": batch},
        )
        
        # Handle both dict-style and list-style responses
        if isinstance(response, dict):
            preds = response.get("predictions", response.get("outputs", []))
        else:
            preds = response
        
        # Normalise: each prediction may itself be a dict with 'generated_text'
        batch_results = []
        for p in preds:
            if isinstance(p, dict):
                batch_results.append(p.get("generated_text", str(p)))
            else:
                batch_results.append(str(p))
        
        # Log individual predictions if requested
        if show_all_predictions:
            for j, (inp, pred) in enumerate(zip(batch, batch_results)):
                idx = i + j + 1
                inp_display = inp[:80] + "..." if len(inp) > 80 else inp
                pred_display = pred[:80] + "..." if len(pred) > 80 else pred
                print(f"      [{idx}/{len(texts)}] Input: {inp_display}")
                print(f"      [{idx}/{len(texts)}] Prediction: {pred_display}")
        
        all_results.extend(batch_results)
    
    print(f"  ✓ Generated {len(all_results)} predictions")
    return all_results


def get_registered_model_name(source_suffix: str) -> str:
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks")
    model_name = f"{MODEL_NAME.replace('/', '_').replace('-', '_').replace('.', '')}_{source_suffix}"
    if registry_uri == "databricks-uc":
        catalog = os.environ["MLFLOW_UC_CATALOG"]
        schema = os.environ["MLFLOW_UC_SCHEMA"]
        return f"{catalog}.{schema}.{model_name}"
    return model_name


def build_eval_dataframe(source_column: str) -> pd.DataFrame:
    """Load dataset and return test-split DataFrame with 'inputs' and 'targets' columns."""
    dataset = load_dataset(DATASET_NAME, split="train")
    split = dataset.train_test_split(test_size=0.2, seed=42)
    test_split = split["test"]

    rows = []
    for sample in test_split:
        target = sample[source_column]
        if isinstance(target, dict):
            target = target.get("text", "")
        elif isinstance(target, list):
            target = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in target
            )
        rows.append({"inputs": sample["instruction"], "targets": str(target)})

    return pd.DataFrame(rows)


def evaluate_model(source_column: str, source_suffix: str) -> dict:
    """Run mlflow.genai.evaluate + mlflow.models.evaluate for a single model.
    
    Predictions are fetched from the Databricks serving endpoint (no local GPU needed).
    """
    model_name = get_registered_model_name(source_suffix)
    model_uri = f"models:/{model_name}@prod"
    version = get_latest_model_version(model_name)
    if version is None:
        raise ValueError(f"No registered versions found for '{model_name}'")
    endpoint_name = resolve_serving_endpoint_name(model_name, version)

    print(f"\n{'='*60}")
    print(f"Evaluating:     {model_name}")
    print(f"Model URI:      {model_uri}")
    print(f"Serving endpoint: {endpoint_name}")
    print(f"Judge:          {GEMINI_MODEL} (via Nexus)")
    print(f"{'='*60}")

    eval_df = build_eval_dataframe(source_column)
    print(f"Test samples: {len(eval_df)}")

    # Generate predictions via the Databricks serving endpoint (no local model load)
    print(f"\n[Step 1/3] Generating predictions from serving endpoint...")
    eval_df["outputs"] = query_serving_endpoint(endpoint_name, eval_df["inputs"].tolist())
    
    # Log sample predictions
    print(f"\n  Sample predictions (showing first 3):")
    for i in range(min(3, len(eval_df))):
        input_text = eval_df.iloc[i]["inputs"][:100] + "..." if len(eval_df.iloc[i]["inputs"]) > 100 else eval_df.iloc[i]["inputs"]
        output_text = eval_df.iloc[i]["outputs"][:100] + "..." if len(eval_df.iloc[i]["outputs"]) > 100 else eval_df.iloc[i]["outputs"]
        print(f"    Sample {i+1}:")
        print(f"      Input:  {input_text}")
        print(f"      Output: {output_text}")
    print(f"\n✓ Predictions complete\n")

    run_name = f"eval_{source_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    combined_metrics = {}

    print(f"Creating MLflow run '{run_name}'...")
    try:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            tracking_uri = mlflow.get_tracking_uri()
            
            print(f"✓ MLflow run created successfully!")
            print(f"  Run ID: {run_id}")
            print(f"  Experiment ID: {experiment_id}")
            print(f"  Tracking URI: {tracking_uri}")
            if "databricks" in tracking_uri.lower():
                # Construct Databricks UI URL
                base_url = tracking_uri.rstrip('/')
                print(f"  View run: {base_url}/#mlflow/experiments/{experiment_id}/runs/{run_id}")
            print()
            
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("endpoint_name", endpoint_name)
            mlflow.log_param("source_column", source_column)
            mlflow.log_param("num_samples", len(eval_df))
            mlflow.log_param("judge_model", GEMINI_MODEL)

            # 1. mlflow.genai.evaluate — LLM-as-a-Judge metrics
            # Requires DataFrame with 'inputs', 'outputs', 'targets' columns (pre-computed outputs)
            print(f"[Step 2/3] Running LLM-as-a-Judge evaluation ({GEMINI_MODEL})...")
            
            # Reset counter and evaluate each metric with progress tracking
            scorers_list = [answer_correctness, answer_relevance, faithfulness]
            print(f"  Evaluating {len(scorers_list)} metrics on {len(eval_df)} samples...")
            
            for scorer_obj in scorers_list:
                _eval_counter["current"] = 0
                _eval_counter["total"] = len(eval_df)
                _eval_counter["metric_name"] = scorer_obj.__name__
                print(f"  Starting {scorer_obj.__name__}...")
            
            genai_results = mlflow.genai.evaluate(
                data=eval_df,
                scorers=scorers_list,
            )
            combined_metrics.update(genai_results.metrics)
            print(f"✓ LLM-as-a-Judge evaluation complete\n")

            # 2. mlflow.models.evaluate — lexical / statistical metrics
            # model=None + predictions='outputs' uses the already-generated predictions
            print(f"[Step 3/3] Running lexical metrics evaluation...")
            print(f"  Computing ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and token count...")
            lexical_results = mlflow.models.evaluate(
                model=None,
                data=eval_df,
                targets="targets",
                predictions="outputs",
                model_type="text",
                extra_metrics=[
                    mlflow.metrics.rouge1(),
                    mlflow.metrics.rouge2(),
                    mlflow.metrics.rougeL(),
                    mlflow.metrics.bleu(),
                    mlflow.metrics.token_count(),
                ],
                evaluator_config={"log_model_explainability": False},
            )
            combined_metrics.update(lexical_results.metrics)
            print(f"✓ Lexical metrics evaluation complete\n")
            
            print(f"✓ MLflow run completed and logged successfully")
            print(f"  Run ID: {run_id}\n")
    
    except Exception as e:
        print(f"\n❌ Error during MLflow run: {e}")
        import traceback
        traceback.print_exc()
        raise

    return combined_metrics


def check_thresholds(source_suffix: str, metrics: dict) -> bool:
    """Print pass/fail for each acceptance threshold. Returns True if all pass."""
    all_pass = True
    print(f"\nAcceptance criteria for '{source_suffix}':")
    for metric_key, threshold in ACCEPTANCE_THRESHOLDS.items():
        value = metrics.get(metric_key)
        if value is None:
            print(f"  {metric_key}: N/A (not found)")
            continue
        passed = value >= threshold
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {metric_key}: {value:.4f} (threshold: {threshold})")
        if not passed:
            all_pass = False
    return all_pass


def evaluate():
    load_dotenv()

    # Validate required environment variables
    print("Validating environment configuration...")
    required_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "NEXUS_BASE_URL",
        "NEXUS_API_KEY",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please ensure these are set in your .env file")
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    print("✓ Environment variables validated\n")

    # MLflow setup
    print("Initializing MLflow...")
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks")
    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    try:
        experiment = mlflow.set_experiment(experiment_name)
        print(f"✓ MLflow configured successfully")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Registry URI: {registry_uri}")
        print(f"  Experiment: {experiment_name} (ID: {experiment.experiment_id})")
        if "databricks" in tracking_uri.lower():
            base_url = tracking_uri.rstrip('/')
            print(f"  View experiment: {base_url}/#mlflow/experiments/{experiment.experiment_id}")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize MLflow: {e}")
        raise

    summary = {}
    total_models = len(SOURCE_SUFFIXES)
    for idx, (source_column, source_suffix) in enumerate(SOURCE_SUFFIXES.items(), 1):
        print(f"\n{'='*70}")
        print(f"MODEL {idx}/{total_models}: Evaluating {source_suffix}")
        print(f"{'='*70}")
        metrics = evaluate_model(source_column, source_suffix)
        passed = check_thresholds(source_suffix, metrics)
        summary[source_suffix] = {"metrics": metrics, "passed": passed}
        print(f"\n✓ Model {idx}/{total_models} evaluation complete")

    # Final summary
    print(f"\n{'='*60}")
    print("Evaluation summary")
    print(f"{'='*60}")
    for suffix, result in summary.items():
        status = "ALL PASSED" if result["passed"] else "SOME FAILED"
        print(f"  {suffix}: {status}")


if __name__ == "__main__":
    evaluate()
