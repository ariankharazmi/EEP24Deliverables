import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the tokenizer and model
model_name = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Sets the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Loads AG News Dataset
dataset = load_dataset("ag_news")

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask'])

# Partition dataset into smaller sets, to save time and compute power
small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(200))

# Define training args
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Increase batch size
    per_device_eval_batch_size=8,   # Increase batch size
    num_train_epochs=1,  # Reduce number of epochs
    weight_decay=0.01,
    save_total_limit=2,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer is set up to work with smaller datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train model function
trainer.train()
