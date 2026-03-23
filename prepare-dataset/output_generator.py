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
)

gpt = AzureChatOpenAI(
    model="gpt-5",  # or your model name
    azure_deployment="gpt-5",  # or your deployment
    api_version="2024-10-21",  # or your api version
    azure_endpoint=os.getenv("NEXUS_BASE_URL"),
    api_key=os.getenv("NEXUS_API_KEY"),
    # temperature=0,
    # max_tokens=None,
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
        # Initialize empty results
        gpt_res = None
        gemini_res = None

        # Only generate if there is an instruction
        if record.get("instruction") and record["instruction"].strip() != "" and not record.get("generation"):

            # 1. GPT-5 Generation
            try:
                resp_gpt = gpt.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=f"instruction:{record['instruction']}"),
                    ]
                )
                gpt_res = resp_gpt.content.strip()
            except Exception as e:
                print(f"GPT Error for ID {record.get('instruction')}: {e}")

            # 2. Gemini Generation
            try:
                resp_gem = gemini.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=f"instruction:{record['instruction']}"),
                    ]
                )
                gemini_res = resp_gem.content.strip()
            except Exception as e:
                print(f"Gemini Error for ID {record.get('instruction')}: {e}")

        # Return a dict with TWO new keys
        # This creates two separate columns in your HF Dataset
        print(f"Processed record with instruction: {record.get('instruction')}, GPT output: {gpt_res}, Gemini output: {gemini_res}")
        return {"gpt_output": gpt_res, "gemini_output": gemini_res}

    # Use .map to transform the entire dataset
    return dataset.map(process_record)


def main():
    dataset = load_dataset(os.getenv("HF_DATASET"), split="train", token=os.getenv("HF_API_KEY"), verification_mode="no_checks")

    # Capture the result of the function!
    updated_dataset = generate_generations(dataset, output_promt)
    updated_dataset.show(5)  # Show a few examples to verify the new columns are added correctly

    # Push the NEW version
    # updated_dataset.push_to_hub(
    #     os.getenv("HF_DATASET"), token=os.getenv("HF_API_KEY"), split="train"
    # )


if __name__ == "__main__":
    main()
