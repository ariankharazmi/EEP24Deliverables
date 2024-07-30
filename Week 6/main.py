import torch
import accelerate
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the tokenizer and GPT-2 (no distilGPT2)
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Load OpenWebText Dataset for Training, split selects first 1% of training set
dataset = load_dataset("openwebtext", split = 'train[:1%]', trust_remote_code = True)

# Preprocess the OpenWebText dataset for tokenization
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation = True, padding='max_length', max_length =256)

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched = True)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask'])

# Partition dataset into smaller sets, split training and evaluation training
small_train_dataset = tokenized_datasets.shuffle(seed = 42).select(range(1000))
small_eval_dataset = tokenized_datasets.shuffle(seed = 42).select(range(200))

# Define training args
training_args = TrainingArguments(
    output_dir = './results',
    evaluation_strategy = 'epoch',
    learning_rate = 2e-5,
    per_device_train_batch_size = 2, #Smaller batch size for faster processing
    per_device_eval_batch_size = 2, # Smaller batch size for faster processing
    num_train_epochs = 1,
    weight_decay = 0.01,
    save_total_limit = 2, #This is the number of checkpoints
)

# Data collator function (batching samples from training set), disabling Masked Language Modelling (we have GPT-2)
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False,
)

# Trainer is set up to work with smaller datasets
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = small_train_dataset,
    eval_dataset = small_eval_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer,
)

# Train model function
trainer.train()
