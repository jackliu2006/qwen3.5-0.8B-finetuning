from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch


def generate_answer(question, model, tokenizer):
    messages = [{"role": "user", "content": question}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)


# Load the fine-tuned model and tokenizer once.
model_name = "jackliu2006/Qwen_Qwen3-0.6B_car_knowledge_finetuned_gemini3"
model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded. Ask questions in a loop. Type 'exit' or 'quit' to stop.")
while True:
    question = input("Enter your question: ").strip()
    if question.lower() in {"exit", "quit"}:
        print("Exiting chat.")
        break

    if not question:
        print("Please enter a question.")
        continue

    generated_text = generate_answer(question, model, tokenizer)
    print(generated_text)
