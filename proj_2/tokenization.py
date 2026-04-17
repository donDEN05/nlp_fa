import torch
from transformers import BertTokenizer


class Tokenizer():
    def __init__(self):
        self.base = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                                       cache_dir='data')
    

    def tokenize(self, data, labels, max_length=128):
        input_ids = []
        attention_mask = []

        for comment in data:
            encoded_dict = self.tokenizer._encode_plus(
                comment,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        output_labels = torch.tensor(data=labels, dtype=torch.float32)

        return input_ids, attention_mask, output_labels
    

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)