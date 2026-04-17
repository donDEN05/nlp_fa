from transformers import BertForSequenceClassification, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config


class Model():
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(config.madel_name, num_labels=config.num_labels)
        self.device = 'cuda'
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
    

    def fit(self, dataloader):
        for epoch in range(config.num_epoch):
            total_loss = 0
            for batch in dataloader:
                input_ids, attention_mask, labels = [k.to(config.device) for k in batch]
                self.optimizer.zero_grad()

                output = self.model(input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                loss = output.loss
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            
            print(f'Эпоха - {epoch}, Лосс - {total_loss}')
