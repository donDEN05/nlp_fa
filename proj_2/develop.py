from tokenization import Tokenizer
import pandas as pd


t = Tokenizer()
data = pd.read_csv('data/toxic_classification/train.csv')
print(data.iloc[1]['comment_text'])
data = t.tokenize(data['comment_text'], data.drop(columns=['id', 'comment_text']))
print(data[0].shape, data[1].shape, data[2].shape)
print(data[0], data[1], data[2])