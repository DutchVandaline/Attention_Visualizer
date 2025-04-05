import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=3):
    model.to(device)
    train_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 평가
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                predictions = torch.sigmoid(outputs) > 0.5
                correct += (predictions.long() == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        val_accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Val Acc: {accuracy:.4f}")
    return train_losses, val_accuracies

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)