import torch
from transformers import BertTokenizer


class Tokenizer():
    def __init__(self):
        self.base = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='data')
    

    def tokenize(self, data: list, max_lenght=128):
        input_ids = []
        attention_mask = []

        for comment in data:
            encoded_dict = self.tokenizer._encode_plus(
                comment,
                add_special_tokens=True,
                max_lenght=max_lenght,
                pad_to_max_lenght=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        return input_ids, attention_mask