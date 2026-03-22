import os
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# client = ChatGoogleGenerativeAI(
#     model=os.getenv("MODEL_ID"),  # see table above to set the desired model id
#     client_options={"api_endpoint": os.getenv("NEXUS_BASE_URL")},
#     google_api_key=os.getenv("NEXUS_API_KEY"),  # Use your Nexus API key here
# )

from langchain_openai import AzureChatOpenAI

client = AzureChatOpenAI(
    name= "gpt-5",  # or your model name
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
you only put topic, model and instruction into the response, no other information is needed. 
you need to generate instructions for the dataset generation based on the {topic} which human message mentioned.
you need to generate number of {num_questions} detailed and specific instructions based on the {topic}.
the instructions should be detailed and specific to guide the model to generate the dataset for the give topic.
"""
# response = client.generate([
#     [
#         SystemMessage(content=ask_for_prompt),
#         HumanMessage(content="I need llm to prepare dataset for car models .")
#     ]
# ])
# print(response.generations[0][0].message.content)
from pydantic import BaseModel, Field
from typing import List

class FineTuningRecord(BaseModel):
    model: str = Field(
        ..., 
        description="The name of the model for which the dataset is being generated, e.g., 'gpt-5'."
    )
    topic: str = Field(
        ..., 
        description="The topic for which the instruction is generated, e.g., 'all mercedes benz car brands and models'."
    )
    instruction: str = Field(
        ..., 
        description="Instruction for the model to generate a specific part of the dataset.'"
    )
    generation: str | None = Field(
        default=None, 
        description="Generation is the expected output from the model when given the instruction."
    )

class FineTuningDataset(BaseModel):
    """A collection of records ready for SFT fine-tuning."""
    records: List[FineTuningRecord]

def generate_questions(topic: str, num_questions: int, prompt: str) -> str:
    structured_model = client.with_structured_output(FineTuningDataset)
    response = structured_model.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=f"numbernum_questions:{num_questions} , topic:{topic}"
                ),
            ]
    )
    return response


print(generate_questions("all mercedes benz car brands and models",1, questions_promt))
# def generate_dataset(
#     topic: str,
# ) -> str:
#     response = client.generate(
#         [
#             [
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=f"Please generate dataset for {topic} ."),
#             ]
#         ]
#     )
#     return response.generations[0][0].message.content

