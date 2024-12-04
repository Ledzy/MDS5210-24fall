import os
import requests
import tiktoken
import numpy as np

import pickle
import json

input_file_path = os.path.join(os.path.dirname(__file__), 'mathInstruct.json')

with open(input_file_path, 'r') as file:
    text_dataset = json.load(file)

input_ids = []
enc = tiktoken.get_encoding("gpt2")

def preprocess_function(sample):
    """
    Formatting function returning a list of samples (kind of necessary for SFT API).
    """
    text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
    text = text + "<|endoftext|>"
    return text

print("Start encoding the dataset...")
for text_sample in text_dataset:
    processed_text = preprocess_function(text_sample)
    input_id_sample = enc.encode(processed_text, allowed_special={"<|endoftext|>"})
    input_ids.append(input_id_sample)

n = len(input_ids)
train_data = input_ids[:int(n*0.9)]
val_data = input_ids[int(n*0.9):]

text_sample = enc.decode(train_data[0])
print("\nSample input text:\n", text_sample)

print("\nnumber of train data: ", len(train_data), "number of validation data: ", len(val_data))
print("max sequence length: ", max([len(x) for x in input_ids]), "average sequence length: ", sum([len(x) for x in input_ids]) / len(input_ids))

pickle.dump(train_data, open(os.path.join(os.path.dirname(__file__), 'train.pkl'), "wb"))
pickle.dump(val_data, open(os.path.join(os.path.dirname(__file__), 'val.pkl'), "wb"))