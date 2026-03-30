import os
import logging
import concurrent.futures
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

from datasets import load_dataset
from huggingface_hub import HfApi

load_dotenv()

"""Initialize the Gemini and GPT models with the appropriate configurations, including API keys and endpoints."""
gemini = ChatGoogleGenerativeAI(
    name="gemini-3.1-pro-preview",  # or your model name
    model="gemini-3.1-pro-preview",  # see table above to set the desired model id
    client_options={"api_endpoint": os.getenv("NEXUS_BASE_URL")},
    google_api_key=os.getenv("NEXUS_API_KEY"),  # Use your Nexus API key here
    #max_output_tokens=int(os.getenv("MAX_TOKENS", 2048)),
    temperature=float(os.getenv("TEMPERATURE", 0.2)),
)

gpt = AzureChatOpenAI(
    model="gpt-5",  # or your model name
    azure_deployment="gpt-5",  # or your deployment
    api_version="2024-10-21",  # or your api version
    azure_endpoint=os.getenv("NEXUS_BASE_URL"),
    api_key=os.getenv("NEXUS_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", 0.2)),
    #max_tokens=int(os.getenv("MAX_TOKENS", 400)),
    # timeout=None,
    # max_retries=2,
    # other params...
)


output_promt = """ you are data engineer to prepare dataset for llm model fine tuning. 
you need to generate output based on the given {instruction}. 
you only need to put the generated output in the response as {generation} without any other text.
"""


def generate_generations(dataset, prompt, batch_size=10, max_workers=5, repo_id=None, token=None):
    def process_record(record):
        # 1. Get existing data (use .get to avoid KeyError)
        instruction = record.get("instruction", "")
        gpt_res = record.get("gpt_output")
        gemini_res = record.get("gemini_output")
        temperature = float(os.getenv("TEMPERATURE", 0.2))
        max_tokens = int(os.getenv("MAX_TOKENS", 400))

        # Skip entirely if instruction is garbage
        if not instruction or instruction.strip() == "":
            logger.warning("Skipping record with empty instruction.")
            return {"gpt_output": gpt_res, "gemini_output": gemini_res,
                    "temperature": temperature, "max_tokens": max_tokens}

        # 2. GPT-5 Generation (Only if missing)
        if not gpt_res or gpt_res.strip() == "":
            try:
                logger.info("GPT generating for instruction: %.50s...", instruction)
                resp_gpt = gpt.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=f"instruction:{instruction}"),
                    ]
                )
                gpt_res = resp_gpt.content.strip()
                logger.info("GPT generation successful for instruction: %.50s...", instruction)
            except Exception as e:
                logger.error("GPT error for instruction '%.50s...': %s", instruction, e)

        # 3. Gemini Generation (Only if missing)
        if not gemini_res or gemini_res.strip() == "":
            try:
                logger.info("Gemini generating for instruction: %.50s...", instruction)
                resp_gem = gemini.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=f"instruction: {instruction}"),
                    ]
                )

                if isinstance(resp_gem.content, list):
                    gemini_res = " ".join(
                        [str(part) for part in resp_gem.content]
                    ).strip()
                else:
                    gemini_res = str(resp_gem.content).strip()
                logger.info("Gemini generation successful for instruction: %.50s...", instruction)
            except Exception as e:
                logger.error("Gemini error for instruction '%.50s...': %s", instruction, e)

        return {"gpt_output": gpt_res, "gemini_output": gemini_res,
                "temperature": temperature, "max_tokens": max_tokens}

    def process_batch(batch):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        records = [{k: batch[k][i] for k in keys} for i in range(n)]

        logger.info("Processing batch of %d records with %d workers.", n, max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_record, records))

        gpt_done = sum(1 for r in results if r.get("gpt_output"))
        gemini_done = sum(1 for r in results if r.get("gemini_output"))
        logger.info(
            "Batch complete — GPT outputs: %d/%d, Gemini outputs: %d/%d.",
            gpt_done, n, gemini_done, n,
        )

        # Collate individual record dicts back into a batch dict
        return {k: [r[k] for r in results] for k in results[0]}

    import tempfile
    from datasets import concatenate_datasets

    api = HfApi()
    total = len(dataset)
    processed_batches = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logger.info("Processing records %d–%d of %d...", start + 1, end, total)
        batch_ds = dataset.select(range(start, end))
        try:
            processed = batch_ds.map(process_batch, batched=True, batch_size=len(batch_ds))
            processed_batches.append(processed)
            if repo_id and token:
                # Upload only this batch shard — O(batch_size), not O(total)
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_path = tmp.name
                processed.to_parquet(tmp_path)
                shard_name = f"generated/batch-{start:07d}-{end:07d}.parquet"
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=shard_name,
                    repo_id=repo_id,
                    token=token,
                    repo_type="dataset",
                )
                logger.info("Uploaded shard %s (%d records).", shard_name, end - start)
        except Exception as e:
            logger.error("Batch %d–%d failed: %s. Skipping.", start + 1, end, e)

    return concatenate_datasets(processed_batches)


def main():
    repo_id = os.getenv("HF_DATASET")
    token = os.getenv("HF_API_KEY")
    api = HfApi()

    # Loading as a Dataset object (not Dict)
    dataset = load_dataset(
        repo_id,
        split="train",
        token=token,
        verification_mode="no_checks",
    )

    # Resume: find already-uploaded shards and skip those records
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, token=token, repo_type="dataset")
        shard_files = sorted(f for f in repo_files if f.startswith("generated/batch-"))
    except Exception:
        shard_files = []

    start_offset = 0
    completed_batches = []
    if shard_files:
        logger.info("Found %d existing shards, resuming...", len(shard_files))
        existing = load_dataset(
            repo_id,
            data_files=shard_files,
            split="train",
            token=token,
            verification_mode="no_checks",
        )
        completed_batches.append(existing)
        start_offset = len(existing)
        logger.info("Skipping first %d already-processed records.", start_offset)

    dataset = dataset.select(range(start_offset, len(dataset)))

    updated_dataset = generate_generations(
        dataset,
        output_promt,
        repo_id=repo_id,
        token=token,
    )

    if completed_batches:
        from datasets import concatenate_datasets
        updated_dataset = concatenate_datasets(completed_batches + [updated_dataset])

    # Final push: consolidates everything into the train split
    logger.info("Pushing %d records to HF Hub as train split...", len(updated_dataset))
    updated_dataset.push_to_hub(repo_id, token=token, split="train")

    # Clean up the intermediate shards from generated/
    for shard in api.list_repo_files(repo_id=repo_id, token=token, repo_type="dataset"):
        if shard.startswith("generated/batch-"):
            api.delete_file(path_in_repo=shard, repo_id=repo_id, token=token, repo_type="dataset")
            logger.info("Deleted shard %s.", shard)


if __name__ == "__main__":
    main()
