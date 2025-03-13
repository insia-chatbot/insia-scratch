import tiktoken
import torch

from torch.utils.data.dataloader import default_collate

class TiktokenTokenizer:
    def __init__(self, block_size):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.block_size = block_size
    
    def encode(self, str, padding=None):
        encoded = self.encoding.encode(str)

        # add eot token until size is equals padding
        # to add padding = len(encoded) - padding
        if padding is not None:
            padding = padding - len(encoded)
            if padding > 0:
                encoded += [self.encoding.eot_token] * padding

        return encoded

    def decode(self, str, ignore_eot=False):
        if not ignore_eot:
            # if encoutring eot token, remove it and all the tokens after it    
            try :
                eot_index = str.index(self.encoding.eot_token)
            except ValueError:
                eot_index = -1

            if eot_index != -1:
                str = str[:eot_index]

        return self.encoding.decode(str)
    
    def size(self):
        return self.encoding.max_token_value