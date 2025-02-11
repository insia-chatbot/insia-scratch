import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class FrWikipediaDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

h_dataset = load_dataset("Kant1/French_Wikipedia_articles")

fr_wikipedia_dataset = FrWikipediaDataset(h_dataset['train'])