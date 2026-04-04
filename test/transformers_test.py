from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3.5-0.8B"

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
