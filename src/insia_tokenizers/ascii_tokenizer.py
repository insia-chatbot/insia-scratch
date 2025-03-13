import torch

class ASCIITokenizer:
    eot_token = 0

    def __init__(self, block_size):
        self.block_size = block_size

    def encode(self, str, padding=None):
        encoded = [ord(c) for c in str]

        # remove token > 255
        encoded = [c for c in encoded if c <= 255]
        
        # add eot token until size is equals padding
        # to add padding = len(encoded) - padding
        if padding is not None:
            padding = padding - len(encoded)
            if padding > 0:
                encoded += [self.eot_token] * padding

        return encoded
    
    def decode(self, str, ignore_eot=False):
        # if encoutring eot token, remove it and all the tokens after it
        if not ignore_eot:
            try :
                eot_index = str.index(self.eot_token)
            except ValueError:
                eot_index = -1
            
            if eot_index != -1:
                str = str[:eot_index]

        return ''.join([chr(c) for c in str])

    def size(self):
        return 256