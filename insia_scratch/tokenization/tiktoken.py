import tiktoken
from torch.utils.data.dataloader import default_collate

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
    
    def collate_fn(batch):
        print(batch)

        return default_collate(batch)


tokenizer = Tokenizer()
padder = Padder(tokenizer)