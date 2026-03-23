from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

from datasets import Dataset
from huggingface_hub import HfApi

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

def main():
    try:
        print("Testing Gemini with string...")
        print(gemini.invoke("Write a haiku about the sea.").content)
    except Exception as e:
        print(f"String test failed: {e}")
    try:
        print("\nTesting Gemini with Message List...")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Write a haiku about the sea.")
        ]
        # Use .content to avoid the 'AIMessage' attribute error we discussed
        response = gemini.invoke(messages)
        print(response.content)
    except Exception as e:
        print(f"Message list test failed: {e}")


if __name__ == "__main__":
    main()
