import torch

def prompt(model, tokenizer, device):
    # ask to the user to enter maximum number of tokens
    max_tokens = int(input("Enter maximum number of tokens: "))
    prompt = input("Enter initial prompt (empty for no prompt): ")

    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)


    print("Prompting...")

    print(tokenizer.decode(model.generate(context, max_new_tokens=max_tokens)[0].tolist(), ignore_eot=True))
