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

load_dotenv()

"""Initialize the Gemini and GPT models with the appropriate configurations, including API keys and endpoints."""
gemini = ChatGoogleGenerativeAI(
    name="gemini-3.1-pro-preview",  # or your model name
    model="gemini-3.1-pro-preview",  # see table above to set the desired model id
    client_options={"api_endpoint": os.getenv("NEXUS_BASE_URL")},
    google_api_key=os.getenv("NEXUS_API_KEY"),  # Use your Nexus API key here
    max_output_tokens=int(os.getenv("MAX_TOKENS", 2048)),
    temperature=float(os.getenv("TEMPERATURE", 0.2)),
)

gpt = AzureChatOpenAI(
    model="gpt-5",  # or your model name
    azure_deployment="gpt-5",  # or your deployment
    api_version="2024-10-21",  # or your api version
    azure_endpoint=os.getenv("NEXUS_BASE_URL"),
    api_key=os.getenv("NEXUS_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", 0.2)),
    max_tokens=int(os.getenv("MAX_TOKENS", 400)),
    # timeout=None,
    # max_retries=2,
    # other params...
)


output_promt = """ you are data engineer to prepare dataset for llm model fine tuning. 
you need to generate output based on the given {instruction}. 
you only need to put the generated output in the response as {generation} without any other text.
"""


def generate_generations(dataset, prompt, batch_size=10, max_workers=5):
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

    return dataset.map(process_batch, batched=True, batch_size=batch_size)


def main():
    # Loading as a Dataset object (not Dict)
    dataset = load_dataset(
        os.getenv("HF_DATASET"),
        split="train",
        token=os.getenv("HF_API_KEY"),
        verification_mode="no_checks",
    )

    updated_dataset = generate_generations(dataset, output_promt)

    # Push to Hub
    updated_dataset.push_to_hub(
        os.getenv("HF_DATASET"),
        token=os.getenv("HF_API_KEY"),
        split="train",
    )


if __name__ == "__main__":
    main()
