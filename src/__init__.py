import os
import torch

from insia_models.gpt_v0 import GPTLanguageModel

from insia_datasets.french_wikipedia_dataset import FrenchWikipediaDataset

from insia_tokenizers.ascii_tokenizer import ASCIITokenizer
from insia_tokenizers.tiktoken_tokenizer import TiktokenTokenizer

from prompting import prompt
from training import train
from losses import get_last_loss

# --- Constants ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
batch_size = 64
epochs = 10
iterations = 5000
estimate_iterations = 500

model_name = "gpt_v0_ctx1024_tiktoken_truncate"
model_class = GPTLanguageModel
# --- Constants ---

tokenizer = TiktokenTokenizer(model_class.get_block_size())
dataset = FrenchWikipediaDataset(tokenizer, device, model_class.get_block_size())



# -- Loading existing model if exists
base_model = None
model = None

if os.path.exists("models/" + model_name + "/"):
    i,_,_ = get_last_loss("models/" + model_name + "/data.csv")

    if i is not None:
        print(f"Loading model from iteration {i}...", )
        base_model = torch.load("models/" + model_name + "/epoch_" + str(i) + ".pth", weights_only=False, map_location=device)
else:
    # create missing directories
    os.makedirs("models/" + model_name + "/", exist_ok=True)

if base_model is None:
    base_model = model_class(device, tokenizer.size())

model = base_model

# -- Check if multiple GPUs are available and use DataParallel
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = torch.nn.DataParallel(base_model)
    

# -- Load model to GPU device(s)
model.to(device)

print("--------")
print("Model loaded from:", model_name)
print("Device:", device)
print("Learning rate:", learning_rate)
print("Batch size:", batch_size)
print("Epochs:", epochs)
print("Iterations:", iterations)
print("Estimate loss iterations:", estimate_iterations)
print("Model size: ", sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print("Tokensizer:", tokenizer)
print("Dataset:", dataset)
print("--------")

action = input("What do you want to do? (train or prompt)? ")

if action == "train":
    train(model, dataset, epochs, batch_size, iterations, estimate_iterations, learning_rate, save=model_name)
else:
    prompt(base_model, tokenizer, device)