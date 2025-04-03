import torch

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_running_loss = 0.0
    train_correct = 0
    total_train_samples = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = train_running_loss / len(dataloader)
    train_accuracy = train_correct / total_train_samples
    return avg_train_loss, train_accuracy