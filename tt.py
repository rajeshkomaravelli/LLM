from nltk import sent_tokenize
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling


file_path= "C:/Users/rajes/Downloads/New folder/48lawsofpower1.txt"

with open(file_path,'r',encoding='utf-8') as file:
    data=file.read().replace('\n','')
    

sentence_tokenized=sent_tokenize(data)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_data, valid_data = train_test_split(sentence_tokenized, test_size=0.2, random_state=42)

train_file = "train.txt"
valid_file = "valid.txt"

with open(train_file, 'w') as f:
    for sentence in train_data:
        f.write(sentence + '\n')

with open(valid_file, 'w') as f:
    for sentence in valid_data:
        f.write(sentence + '\n')

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128  
)

valid_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=valid_file,
    block_size=128  
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,  
    per_device_train_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()


model.save_pretrained("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")      

