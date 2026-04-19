from torch.utils.data import DataLoader, TensorDataset
from config import config


class CustomDataLoader():
    def __init__(self):
        self.base = None
    

    def dataloader(self, input_ids, attention_mask, labels):
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader_ = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        return dataloader_
    

    def dataloader_val(self, input_ids, attention_mask):
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader_ = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        return dataloader_