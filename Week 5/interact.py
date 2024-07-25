from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Loads the trained model and tokenizer, replace path with your own file pathname
model_path = '/Users/kharazmimac/PycharmProjects/LLMScriptTest9.6/results/checkpoint-125'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set up pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Interactive Prompt
while True:
    text = input("Enter a prompt: ")
    if text.lower() == 'exit':
        break
    result = text_generator(text, max_length=100, num_return_sequences=1)
    print(result[0]['generated_text'])

