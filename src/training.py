import torch

@torch.no_grad()
def estimate_loss(model, train_dataset, test_dataset, collate_fn, iterations, batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    out = {}
    model.eval()
 
    # train losses
    train_losses = torch.zeros(iterations)
    for i, (x, y) in enumerate(train_loader):
        if i >= iterations: break
        _, loss = model(x, y)
        train_losses[i] = loss
    out['train_loss'] = train_losses.mean().item()

    # test losses
    test_losses = torch.zeros(iterations)
    for i, (x, y) in enumerate(test_loader):
        if i >= iterations: break
        _, loss = model(x, y)
        test_losses[i] = loss
    out['test_loss'] = test_losses.mean().item()

    model.train()

    return out

def train(model, dataset, num_epochs=10, batch_size = 4, iterations=900, estimate_iterations=300, learning_rate=0.001, save=None):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        for i, (x, y) in enumerate(train_loader):
            if i % iterations == 0: 
                losses = estimate_loss(model, train_dataset, test_dataset, dataset.collate_fn, estimate_iterations, batch_size)
                print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {losses['train_loss']}, Test loss: {losses['test_loss']}")

                if save:
                    torch.save(model, save)
                    print(f"Model saved as {save}")

            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Iteration {i+1}/{len(train_dataset)}, Loss: {loss.item()}")
        

