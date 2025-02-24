import tiktoken
from torch.utils.data.dataloader import default_collate
import torch

class Tokenizer():
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def encode(self, str):
        return self.encoding.encode(str)
    
    def encode_batch(self, strs):
        ## add padding to make all sequences the same length
        return self.encoding.encode_batch(strs)

    def decode(self, str):
        return self.encoding.decode(str)
    
    def decode_batch(self, strs):
        return self.encoding.decode_batch(strs)
    
    def size(self):
        return self.encoding.max_token_value
    
class Padder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def pad(self, x, max_len):
        return x + [tokenizer.encoding.eot_token] * (max_len - len(x))
    
    def collate_fn(self, batch):
        # Array of couple

        new_batch = []
        # get the max len of x and y
        max_len = max(len(x) for x, _ in batch)
        max_len = max(max(len(y) for _, y in batch), max_len)

        # Pad x and y
        for x, y in batch:
            x_padded = self.pad(x, max_len)
            y_padded = self.pad(y, max_len)
            new_batch.append((x_padded, y_padded))
        batch = new_batch

        return default_collate(batch)


tokenizer = Tokenizer()
padder = Padder(tokenizer)