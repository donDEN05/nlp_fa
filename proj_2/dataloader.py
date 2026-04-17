from torch.utils.data import DataLoader, TensorDataset
from config import config


class CustomDataLoader():
    def __init__(self):
        self.base = None
    

    def dataloader(self, input_ids, attention_mask, labels):
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader_ = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        return dataloader_