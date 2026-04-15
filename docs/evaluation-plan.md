# Model Evaluation Plan

## Overview

Evaluate the two fine-tuned Qwen3-0.6B models registered in MLflow/Databricks Unity Catalog against the `jackliu2006/car_knowledge` dataset from Hugging Face using **MLflow's built-in `mlflow.evaluate()`** API.

| Model alias | Registered model | Fine-tuned source |
|---|---|---|
| `gemini3` | `…_gemini3` | `gemini_output` column |
| `gpt5` | `…_gpt5` | `gpt_output` column |

Both models are evaluated on the **held-out test split** (20 % of the dataset, seeded with `seed=42`, matching the training split used in `sft.py`).

---

## Why `mlflow.evaluate()`

- **Zero extra dependencies** — metrics are built into MLflow, no need to install `rouge_score`, `sacrebleu`, or `bert_score` separately.
- **Automatic MLflow logging** — metrics, artifacts, and per-sample results are logged to the active run automatically.
- **Unified interface** — `mlflow.evaluate()` works with any registered model URI (`models:/name@alias`) and a pandas DataFrame.
- **Extensible** — custom `mlflow.metrics.make_metric()` scorers can be added alongside built-ins.

---

## Evaluation Metrics (MLflow built-ins)

Passed as the `extra_metrics` list to `mlflow.evaluate()`.

### Lexical / Statistical Metrics

| Metric | MLflow scorer | Description |
|---|---|---|
| ROUGE-1 | `mlflow.metrics.rouge1()` | Unigram overlap F1 |
| ROUGE-2 | `mlflow.metrics.rouge2()` | Bigram overlap F1 |
| ROUGE-L | `mlflow.metrics.rougeL()` | Longest common subsequence F1 |
| BLEU | `mlflow.metrics.bleu()` | Corpus-level BLEU (sacrebleu) |
| Token count | `mlflow.metrics.token_count()` | Average output token length |

**Target thresholds:**

| Metric | Minimum |
|---|---|
| ROUGE-L F1 | ≥ 0.35 |
| BLEU | ≥ 15 |

### LLM-as-a-Judge Metrics (`mlflow.metrics.genai`)

A judge LLM (configured via `JUDGE_MODEL_ENDPOINT` env var, e.g. a GPT-4o or Gemini endpoint) scores each prediction on the following dimensions:

| Metric | MLflow scorer | Scale | Description |
|---|---|---|---|
| Answer correctness | `mlflow.metrics.genai.answer_correctness()` | 1–5 | Is the answer factually correct vs. the reference? |
| Answer relevance | `mlflow.metrics.genai.answer_relevance()` | 1–5 | Is the answer on-topic and responsive to the question? |
| Faithfulness | `mlflow.metrics.genai.faithfulness()` | 1–5 | Does the answer avoid hallucination relative to the reference? |

**Target thresholds:**

| Metric | Minimum average score |
|---|---|
| Answer correctness | ≥ 3.5 / 5 |
| Answer relevance | ≥ 3.5 / 5 |
| Faithfulness | ≥ 4.0 / 5 |

Configuration:

```python
judge_model = mlflow.metrics.genai.make_genai_metric(
    model=os.environ["JUDGE_MODEL_ENDPOINT"]   # e.g. "endpoints:/my-gpt4o-endpoint"
)
```

Required environment variable:
- `JUDGE_MODEL_ENDPOINT`: MLflow-compatible model URI for the judge LLM (e.g. `endpoints:/databricks-meta-llama-3-3-70b-instruct` or an OpenAI-compatible endpoint)

---

## Evaluation Procedure

### Data

1. Load `jackliu2006/car_knowledge` from HuggingFace.
2. Re-create the same train/test split: `dataset.train_test_split(test_size=0.2, seed=42)`.
3. Use the **test** split only.
4. Build a pandas DataFrame with columns `inputs` (= `instruction`) and `targets` (= reference column).

### Inference & evaluation

```python
mlflow.evaluate(
    model=f"models:/{model_name}@prod",  # Unity Catalog alias
    data=eval_df,                         # DataFrame with 'inputs' / 'targets'
    targets="targets",
    model_type="text",
    extra_metrics=[
        mlflow.metrics.rouge1(),
        mlflow.metrics.rouge2(),
        mlflow.metrics.rougeL(),
        mlflow.metrics.bleu(),
        mlflow.metrics.token_count(),
        mlflow.metrics.genai.answer_correctness(model=judge_model_uri),
        mlflow.metrics.genai.answer_relevance(model=judge_model_uri),
        mlflow.metrics.genai.faithfulness(model=judge_model_uri),
    ],
)
```

`mlflow.evaluate()` automatically:
- Calls the model for each row
- Computes all metrics against `targets`
- Logs aggregate metrics and per-row results as an MLflow artifact

### MLflow run naming

Each evaluation is logged under a run named `eval_{source_suffix}_{timestamp}` in the same MLflow experiment used for training.

---

## Evaluation Outputs

All results are logged automatically by `mlflow.evaluate()`:

| Logged item | Type |
|---|---|
| `rouge1` / `rouge2` / `rougeL` | MLflow metric |
| `bleu` | MLflow metric |
| `token_count` | MLflow metric |
| `answer_correctness/v1/mean` | MLflow metric |
| `answer_relevance/v1/mean` | MLflow metric |
| `faithfulness/v1/mean` | MLflow metric |
| `eval_results_table.json` | MLflow artifact (per-row predictions + scores) |

---

## Acceptance Criteria

| Metric | Minimum threshold |
|---|---|
| ROUGE-L F1 | ≥ 0.35 |
| BLEU | ≥ 15 |
| Answer correctness | ≥ 3.5 / 5 |
| Answer relevance | ≥ 3.5 / 5 |
| Faithfulness | ≥ 4.0 / 5 |

A model that fails any threshold will be flagged in the output but will **not** automatically be removed from the registry.

---

## Implementation

The evaluation is implemented in `train_eval/eval.py`. Run with:

```bash
uv run train_eval/eval.py
```
