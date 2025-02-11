import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get a single sample (input, target pair)
        item = self.dataset[idx]
        text = item['text']  # for IMDB dataset, the text is under the 'text' key
        label = item['label']  # the label is under the 'label' key

        return text, label
