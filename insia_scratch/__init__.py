import torch
from data.fr_wikipedia import fr_wikipedia_dataset
from torch.utils.data import DataLoader
from models.gpt_v0 import GPTLanguageModel
import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
batch_size = 64
epochs = 1000

dataset = fr_wikipedia_dataset
train_loader = DataLoader(dataset, batch_size, shuffle=True)

model = GPTLanguageModel(device)
model.to(device)

train.train_model(model, train_loader, epochs, learning_rate, save="gpt_v0.pth")

