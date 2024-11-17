import os
import requests
import tiktoken
import numpy as np

import pickle
import json


def tokenize_dataset(text_dataset, tokenizer):
    """
    Process the dataset and tokenize it.
    Return: list of tokenized ids.
    """
    # TODO
    pass

input_file_path = os.path.join(os.path.dirname(__file__), 'alpaca_gpt4_data_en.json')

with open(input_file_path, 'r') as file:
    text_dataset = json.load(file)

tokenizer = tiktoken.get_encoding("gpt2")
input_ids = tokenize_dataset(text_dataset, tokenizer)
n = len(input_ids)
train_data = input_ids[:int(n*0.9)]
val_data = input_ids[int(n*0.9):]

# print out some stats
text_sample = tokenizer.decode(train_data[0])
print("\nSample input text:\n", text_sample)
print("\nnumber of train data: ", len(train_data), "number of validation data: ", len(val_data))
print("max sequence length: ", max([len(x) for x in input_ids]), "average sequence length: ", sum([len(x) for x in input_ids]) / len(input_ids))

# save the data
pickle.dump(train_data, open(os.path.join(os.path.dirname(__file__), 'train.pkl'), "wb"))
pickle.dump(val_data, open(os.path.join(os.path.dirname(__file__), 'val.pkl'), "wb"))