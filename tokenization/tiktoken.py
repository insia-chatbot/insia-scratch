import tiktoken

class Tokenizer():
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def encode(self, str):
        return self.encoding.encode(str)
    
    def encode_batch(self, strs):
        return self.encoding.encode_batch(strs)

    def decode(self, str):
        return self.encoding.decode(str)
    
    def decode_batch(self, strs):
        return self.encoding.decode_batch(strs)
    
    def size(self):
        return self.encoding.max_token_value
    
tokenizer = Tokenizer()