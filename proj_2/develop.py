from tokenization import Tokenizer


t = Tokenizer()

data = t.tokenize(['Now, an additional step for faster processing of the model. You can move the model to the GPU if available, or to the CPU if not.'],
    [1])
print(data[1].shape)