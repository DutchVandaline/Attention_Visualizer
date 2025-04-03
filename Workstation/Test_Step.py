import torch
from sklearn.metrics import f1_score


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    total_val_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            val_running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == labels).sum().item()
            total_val_samples += labels.size(0)

            # Collect labels and predictions for F1 score calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_val_loss = val_running_loss / len(dataloader)
    val_accuracy = val_correct / total_val_samples

    # Calculate F1 score
    val_f1_score = f1_score(all_labels, all_predictions, average='weighted')

    return avg_val_loss, val_accuracy, val_f1_score
