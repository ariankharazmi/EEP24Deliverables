import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
from datetime import date

ts = int(time.time())
today = date.today()
print(today)

st.write("You are visiting on: " + str(today))
st.title("GPT-2 Interactive Page (Web Wrapper)")
st.write("User prompt will generate text")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)

def generate_text(prompt, max_length=200):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


prompt = st.text_area("Prompt:", value="Enter prompt")

if st.button("Generate Response"):
    with st.spinner("..."):
        generated_text = generate_text(prompt)
        st.write("Generated Response:")
        st.write(generated_text)