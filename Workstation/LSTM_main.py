import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from AGNewsDataset import AGNewsDataset
from Train_Step import train_step
from Test_Step import test_step

from tqdm.auto import tqdm
import pandas as pd
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
train_file_path = "/home/junha/AGNews/Dataset/train.csv"
train_data = pd.read_csv(train_file_path)
test_file_path = "/home/junha/AGNews/Dataset/test.csv"
test_data = pd.read_csv(test_file_path)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_SEQ_LEN = 128

train_dataset = AGNewsDataset(train_data, tokenizer, max_len=MAX_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = AGNewsDataset(test_data, tokenizer, max_len=MAX_SEQ_LEN)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 하이퍼파라미터 설정
NUM_CLASSES = 4
VOCAB_SIZE = len(tokenizer.get_vocab())
EMBEDDING_DIM = 768
HIDDEN_SIZE = 512  # LSTM의 hidden size
NUM_LSTM_LAYERS = 2  # LSTM layer 수
DROPOUT = 0.1

LEARNING_RATE = 5e-5
NUM_EPOCHS = 20


# LSTM 기반 분류 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 num_layers: int, num_classes: int, dropout: float = 0.1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # 양방향 LSTM 사용 (batch_first=True)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # 양방향이므로 hidden_size*2
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, attention_mask=None):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        # 평균 풀링: 시퀀스 차원을 평균내어 고정 길이 벡터 생성
        pooled = torch.mean(lstm_out, dim=1)  # (batch, hidden_size*2)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, num_classes)
        return logits


# 모델 초기화
model = LSTMClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LSTM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

save_dir = "/home/junha/AGNews/Checkpoint/LSTM/"
os.makedirs(save_dir, exist_ok=True)

for epoch in tqdm(range(NUM_EPOCHS)):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    train_loss, train_accuracy = train_step(model, train_dataloader, loss_fn, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    torch.cuda.empty_cache()

    test_loss, test_accuracy, val_f1_score = test_step(model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1 Score: {val_f1_score:.4f}")
    torch.cuda.empty_cache()

    epoch_model_path = os.path.join(save_dir, f"LSTM_{epoch + 1}.pth")
    torch.save(model.state_dict(), epoch_model_path)
    print(f"Model for epoch {epoch + 1} saved to {epoch_model_path}\n")
