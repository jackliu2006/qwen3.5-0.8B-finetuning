import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

from datasets import Dataset
from huggingface_hub import HfApi

import time
import uuid
from pydantic import BaseModel, Field
from typing import List
load_dotenv()

"""Initialize the Gemini and GPT models with the appropriate configurations, including API keys and endpoints."""
gemini = ChatGoogleGenerativeAI(
    name="gemini-3.1-pro-preview",  # or your model name
    model="gemini-3.1-pro-preview",  # see table above to set the desired model id
    client_options={"api_endpoint": os.getenv("NEXUS_BASE_URL")},
    google_api_key=os.getenv("NEXUS_API_KEY"),  # Use your Nexus API key here
    temperature=float(os.getenv("TEMPERATURE", 0)),
)

gpt = AzureChatOpenAI(
    model="gpt-5",  # or your model name
    azure_deployment="gpt-5",  # or your deployment
    api_version="2024-10-21",  # or your api version
    azure_endpoint=os.getenv("NEXUS_BASE_URL"),
    api_key=os.getenv("NEXUS_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", 0)),
   # max_tokens=int(os.getenv("MAX_TOKENS", 400)),
    # timeout=None,
    # max_retries=2,
    # other params...
)

system_prompt = """
you are a data engineer of llm model fine tuning, you need to generate dataset for fine tuning based on the Human message, the dataset should be in json and with below format:
[
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about PEFT."},
            {"role": "assistant", "content": "PEFT stands for Parameter-Efficient Fine-Tuning..."},
            {"role": "user", "content": "Is it better than full fine-tuning?"},
            {"role": "assistant", "content": "It is much faster and uses less VRAM!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is TRL?"},
            {"role": "assistant", "content": "TRL is a library for Reinforcement Learning."}
        ]
    }
]
Scope of dataset:
* all car models belongs to the brand which human message mentioned.
* the dataset should cover all the knowledges of these car models, such as the history, the features, the price, the performance, etc.
* the dataset should be as comprehensive as possible, and should be in a format that can be easily used for fine tuning a llm model.
* the dataset should also give the reasons for the answers, which can help the model to learn better.
* the dataset should also cover the sales volumes of each model in different regions, which can help the model to learn the market trends and preferences.
"""

questions_promt = """ you are data engineer to prepare dataset for llm model fine tuning. 
your generated instruction should aims to get the model generate output of the instruction within {MAX_TOKENS} tokens. The limitation of the output should not be in the generated instruction.
you only put topic, model and instruction into the response, no other information is needed. 
you need to generate instructions for the dataset generation based on the {topic} which human message mentioned.
you need to generate number of {batch_size} detailed and specific instructions based on the {topic}.
the instructions should be detailed and specific to guide the model to generate the dataset for the give topic.
"""


class FineTuningRecord(BaseModel):
    model: str = Field(
        ...,
        description="The name of the model for which the dataset is being generated, e.g., 'gpt-5'.",
    )
    topic: str = Field(
        ...,
        description="The topic for which the instruction is generated, e.g., 'all mercedes benz car brands and models'.",
    )
    instruction: str = Field(
        ...,
        description="Instruction for the model to generate a specific part of the dataset.'",
    )
    temperature: float = Field(
        ...,
        description="The temperature setting used for generation, which controls the randomness of the output.",
    )



class FineTuningDataset(BaseModel):
    """A collection of records ready for SFT fine-tuning."""

    records: List[FineTuningRecord]


def upload_to_hf_hub(file_path: str, repo_id: str, token: str, path_in_repo: str):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
    )


def generate_questions(topic: str, target_questions: int, batch_size: int, prompt: str):
    total_generated = 0
    while total_generated < target_questions:
        dataset = []
        """Use Gpt-5 to generate instructions for dataset generation based on the topic."""
        structured_model = gpt.with_structured_output(FineTuningDataset)
        try:
            print(f"[{gpt.model}] Sending request for topic: {topic}...")
            t0 = time.perf_counter()
            response = structured_model.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=f"batch_size:{batch_size} , topic:{topic}, model:{gpt.model}, max_tokens:{os.getenv('MAX_TOKENS', 400)}, temperature:{os.getenv('TEMPERATURE', 0.2)}"
                    ),
                ]
            )
            elapsed = time.perf_counter() - t0
            print(
                f"[{gpt.model}] Received {len(response.records)} records in {elapsed:.2f}s"
            )
            dataset.extend(response.records)
        except Exception as e:
            print(
                f"Error generating instructions for topic: {topic} using model: {gpt.model}: {e}"
            )
        """ Use gemini to generate instructions for dataset generation based on the topic. """
        structured_model = gemini.with_structured_output(FineTuningDataset)
        try:
            print(f"[{gemini.model}] Sending request for topic: {topic}...")
            t0 = time.perf_counter()
            response = structured_model.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=f"batch_size:{batch_size} , topic:{topic}, model:{gemini.model}, max_tokens:{os.getenv('MAX_TOKENS', 400)}, temperature:{os.getenv('TEMPERATURE', 0.2)}"
                    ),
                ]
            )
            elapsed = time.perf_counter() - t0
            print(
                f"[{gemini.model}] Received {len(response.records)} records in {elapsed:.2f}s"
            )
            dataset.extend(response.records)
        except Exception as e:
            print(
                f"Error generating instructions for topic: {topic} using model: {gemini.model}: {e}"
            )
        total_generated += len(dataset)
        """convert the dataset to Hugging Face Dataset format and push to Hugging Face Hub"""
        dicts = [r.model_dump() for r in dataset]
        ds = Dataset.from_list(dicts)
        ds.to_parquet("new_data.parquet")
        """Upload the generated dataset to Hugging Face Hub using a unique name to avoid overwriting existing files."""
        unique_id = str(uuid.uuid4())[:8]
        upload_to_hf_hub(
            file_path="new_data.parquet",
            repo_id=os.getenv("HF_DATASET"),
            token=os.getenv("HF_API_KEY"),
            path_in_repo=f"data/train-{unique_id}.parquet",
        )


def main():
    topics = os.getenv("TOPICS").split(",")
    for topic in topics:
        generate_questions(
            topic,
            int(os.getenv("TARGET_DATASET_SIZE")),
            int(os.getenv("GENERATE_BATCH_SIZE")),
            questions_promt,
        )


if __name__ == "__main__":
    main()
