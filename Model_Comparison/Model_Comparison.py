import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import numpy as np
import random

from Train_Test_Step import train_model, count_parameters
from SyntheticTextDataset import SyntheticTextDataset
from Models.TextViT import TextViT
from Models.BERT import BertClassifier
from Models.LSTM import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

vocab_size = 30522
max_seq_len = 50
embedding_dim = 768
batch_size = 32
num_train = 1000
num_val = 200
num_epochs = 3
lr = 0.001
num_classes = 1

train_dataset = SyntheticTextDataset(num_train, max_seq_len, vocab_size)
val_dataset = SyntheticTextDataset(num_val, max_seq_len, vocab_size)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)



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
bert_model = BertClassifier(num_classes=num_classes, vocab_size=vocab_size, max_len=max_seq_len)
lstm_model = LSTMClassifier(vocab_size=vocab_size,
                            embedding_dim=embedding_dim,
                            hidden_size=384,
                            num_layers=1,
                            num_classes=num_classes)

criterion = nn.BCEWithLogitsLoss()

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
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, device,
                                               num_epochs=num_epochs)
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
final_accuracies = [results[m]["val_accuracies"][-1] for m in model_names]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(model_names, param_counts, color='skyblue')
plt.ylabel("Parameter Count")
plt.title("모델 파라미터 수 비교")

plt.subplot(1, 2, 2)
plt.bar(model_names, training_times, color='lightgreen')
plt.ylabel("Training Time (s)")
plt.title("모델 학습 시간 비교")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(model_names, final_accuracies, color='salmon')
plt.ylabel("Validation Accuracy")
plt.title("최종 검증 정확도 비교")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(8, 6))
for name in model_names:
    plt.plot(range(1, num_epochs + 1), results[name]["train_losses"], marker='o', label=name)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Epoch별 Training Loss 변화")
plt.legend()
plt.show()
