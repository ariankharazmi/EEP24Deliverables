import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = "/Users/kharazmimac/Downloads/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_text = "Hello, tell me something interesting."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

print(outputs)