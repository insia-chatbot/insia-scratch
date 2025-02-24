import torch
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001, save=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            print(x_batch, y_batch)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = F.cross_entropy(output.view(-1, len(model.idx_to_char)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if save:
            torch.save(model.state_dict(), save)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")
