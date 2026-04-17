from torch.utils.data import DataLoader, TensorDataset


class CustomDataLoader():
    def __init__(self):
        self.base = None
    

    def dataloader(self, input_ids, attention_mask, labels):
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        