import torch
from transformers import BertTokenizer
from config import config
import numpy as np
import pandas as pd


class Tokenizer():
    def __init__(self):
        self.base = None
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name, 
                                                       cache_dir=config.cache_dir)
    

    def tokenize(self, data, labels, max_length=config.max_length):
        encoded_dict = self.tokenizer(
            data,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        output_labels = torch.tensor(data=labels, dtype=torch.float32)

        return encoded_dict['input_ids'], encoded_dict['attention_mask'], output_labels


    def tokenize_val(self, data, max_length=config.max_length):
        encoded_dict = self.tokenizer(
            data,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    
    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)