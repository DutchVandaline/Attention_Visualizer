import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader


from AGNewsDataset import AGNewsDataset
from Train_Test_Step import train_model, count_parameters
from Models.TextViT import TextViT
from Models.BERT import BertClassifier
from Models.LSTM import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_len = 128
embedding_dim = 768
num_epochs = 20
lr = 0.001
num_classes = 4
vocab_size = len(tokenizer.get_vocab())

train_file_path = "/home/junha/TextViT_AG_DBPedia/Dataset/train.csv"
train_data = pd.read_csv(train_file_path)
test_file_path = "/home/junha/TextViT_AG_DBPedia/Dataset/test.csv"
test_data = pd.read_csv(test_file_path)

train_dataset = AGNewsDataset(train_data, tokenizer, max_len=max_seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = AGNewsDataset(test_data, tokenizer, max_len=max_seq_len)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

textvit_model = TextViT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_transformer_layers=2,
    embedding_dim=embedding_dim,
    mlp_size=256,
    num_heads=4,
    attn_dropout=0.1,
    mlp_dropout=0.1,
    embedding_dropout=0.1
)
bert_model = BertClassifier(
    num_classes=num_classes,
    vocab_size=vocab_size,
    max_len=max_seq_len
)
lstm_model = LSTMClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=384,
    num_layers=1,
    num_classes=num_classes
)

criterion = nn.CrossEntropyLoss()

models = {
    "TextViT": textvit_model,
    "SimpleBERT": bert_model,
    "LSTM": lstm_model
}

results = {}
for name, model in models.items():
    print(f"\n===== Training {name} =====")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    train_losses, val_accuracies = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs=num_epochs)
    elapsed = time.time() - start_time
    param_count = count_parameters(model)
    results[name] = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "training_time": elapsed,
        "param_count": param_count
    }
    print(f"{name} - Total training time: {elapsed:.2f}s, Parameters: {param_count}\n")

model_names = list(results.keys())
param_counts = [results[m]["param_count"] for m in model_names]
training_times = [results[m]["training_time"] for m in model_names]
best_accuracies = [max(results[m]["val_accuracies"]) for m in model_names]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(model_names, param_counts, color='skyblue')
plt.ylabel("Parameter Count")
plt.title("Model parameter comparison")
plt.subplot(1, 2, 2)
plt.bar(model_names, training_times, color='lightgreen')
plt.ylabel("Training Time (s)")
plt.title("Model train time comparison")
plt.tight_layout()
plt.savefig("model_comparison_parameters_training_time.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(model_names, best_accuracies, color='salmon')
plt.ylabel("Best Validation Accuracy")
plt.title("Model validation accuracy comparison")
plt.ylim(0, 1)
plt.savefig("model_comparison_best_validation_accuracy.png")
plt.close()

plt.figure(figsize=(8, 6))
for name in model_names:
    plt.plot(range(1, num_epochs + 1), results[name]["train_losses"], marker='o', label=name)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Train loss change per epoch")
plt.legend()
plt.savefig("train_loss_change_per_epoch.png")
plt.close()
