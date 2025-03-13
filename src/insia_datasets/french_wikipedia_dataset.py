import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class FrenchWikipediaDataset(Dataset):
    def __init__(self, tokenizer, device, block_size=128):
        h_dataset = load_dataset("Kant1/French_Wikipedia_articles")
        self.dataset = h_dataset['train']
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        encoded = self.tokenizer.encode(text, padding=1)
        
        return encoded
    
    def collate_truncate_fn(self, batch):
        x_batch = []
        y_batch = []

        minimum = min([len(data) for data in batch])

        for i, data in enumerate(batch):
            if (len(data) < 16):
                batch[i] = self[torch.randint(0, len(self.dataset), (1,)).item()]
                return self.collate_truncate_fn(batch)
            
            # Truncate data and pick a random start point
            length = min(self.block_size, minimum)
            start = torch.randint(0, max(1, len(data) - length), (1,)).item()

            x = data[start:start+length-1]
            y = data[start+1:start+length]

            x_batch.append(torch.tensor(x, device=self.device))
            y_batch.append(torch.tensor(y, device=self.device))

        sx, sy = torch.stack(x_batch), torch.stack(y_batch)

        sx.to(self.device)
        sy.to(self.device)

        return sx, sy

        
    
    def collate_padding_fn(self, batch):
        x_batch = []
        y_batch = []

        longuest = max([len(data) for data in batch])

        for i, data in enumerate(batch):
            if (len(data) < 16):
                batch[i] = self[torch.randint(0, len(self.dataset), (1,)).item()]
                return self.collate_padding_fn(batch)
            
            # Pad data
            padding = longuest - len(data)
            if padding > 0:
                data += [self.tokenizer.eot_token] * padding

            # pick a random start point
            start = torch.randint(0, max(1, len(data) - self.block_size), (1,)).item()

            x = data[start:start+min(self.block_size, len(data)-start)-1]
            y = data[start+1:start+min(self.block_size, len(data)-start)]

            x_batch.append(torch.tensor(x, device=self.device))
            y_batch.append(torch.tensor(y, device=self.device))

        sx, sy = torch.stack(x_batch), torch.stack(y_batch)

        sx.to(self.device)
        sy.to(self.device)

        return sx, sy