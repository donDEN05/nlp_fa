from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
import torch
import numpy as np
from tqdm import tqdm


class Model():
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels, cache_dir=config.cache_dir).to(config.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.tuned = False
    

    def fit(self, dataloader):
        self.model.train()
        for epoch in range(config.num_epoch):
            total_loss = 0
            count = 0
            for batch in tqdm(dataloader, desc="Predicting"):
                count += 1
                input_ids, attention_mask, labels = [k.to(config.device) for k in batch]
                self.optimizer.zero_grad()

                output = self.model(input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                loss = output.loss
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                print(count, loss)
            self.scheduler.step()
            
            print(f'Эпоха - {epoch}, Лосс - {total_loss}')

        self.model.save_pretrained(config.save_weights_dir)
        self.tuned = True
        print('Fitted')


    def predict(self, dataloader):
        if self.tuned:
            self.model = BertForSequenceClassification._load_from_state_dict(config.save_weights_dir, num_labels=config.num_labels)
        self.model.eval()
        self.model.to(config.device)
        all_predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids, attention_mask = [k.to(config.device) for k in batch]
                output = self.model(input_ids,attention_mask=attention_mask)
                preds = output.logits.cpu()
                all_predictions.extend(preds.numpy())

        return all_predictions