import torch
from torch.utils.data import Dataset
from datasets import load_dataset

h_dataset = load_dataset("Kant1/French_Wikipedia_articles")

# class MyDataset(Dataset):
#     def __init__(self):
#         # Your dataset initialization here
#         pass

#     def __len__(self):
#         # Return the dataset length
#         return 1000

#     def __getitem__(self, idx):
#         # Return a sample and target for the given index
#         sample = torch.randn(28, 28)  # Example tensor
#         target = torch.tensor(1)  # Example target
#         return sample, target

print(h_dataset.get('train'))