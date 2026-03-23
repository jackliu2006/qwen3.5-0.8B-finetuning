import os
from dotenv import load_dotenv

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


def generate_generations(dataset, prompt):
    def process_record(record):
        # 1. Get existing data (use .get to avoid KeyError)
        instruction = record.get("instruction", "")
        gpt_res = record.get("gpt_output")
        gemini_res = record.get("gemini_output")

        # Skip entirely if instruction is garbage
        if not instruction or instruction.strip() == "":
            return {"gpt_output": gpt_res, "gemini_output": gemini_res}

        # 2. GPT-5 Generation (Only if missing)
        if not gpt_res or gpt_res.strip() == "":
            try:
                print(
                    f"GPT Generating for instruction: {instruction[:50]}..."
                )  # Log the instruction being processed
                resp_gpt = gpt.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=f"instruction:{instruction}"),
                    ]
                )
                gpt_res = resp_gpt.content.strip()
                print(
                    f"GPT Generation successful for instruction: {instruction[:50]}..."
                )
            except Exception as e:
                print(f"GPT Error: {e}")

        # 3. Gemini Generation (Only if missing)
        if not gemini_res or gemini_res.strip() == "":
            try:
                print(
                    f"Gemini Generating for instruction: {instruction[:50]}..."
                )  # Log the instruction being processed
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
                print(
                    f"Gemini Generation successful for instruction: {instruction[:50]}..."
                )
            except Exception as e:
                print(f"Gemini Error: {e}")

        return {"gpt_output": gpt_res, "gemini_output": gemini_res, "temperature": float(os.getenv("TEMPERATURE", 0.2)), "max_tokens": int(os.getenv("MAX_TOKENS", 400))}

    return dataset.map(process_record)


def main():
    # Loading as a Dataset object (not Dict)
    dataset = load_dataset(
        os.getenv("HF_DATASET"),
        split="train",
        token=os.getenv("HF_API_KEY"),
        verification_mode="no_checks",
    )

    updated_dataset = generate_generations(dataset.select(range(2)), output_promt)

    # Push to Hub
    updated_dataset.push_to_hub(
        os.getenv("HF_DATASET"),
        token=os.getenv("HF_API_KEY"),
        split="train",
    )


if __name__ == "__main__":
    main()
