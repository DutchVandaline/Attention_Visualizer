import torch.nn
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from AGNewsDataset import AGNewsDataset
from Models.TextViT import TextViT
from Train_Step import train_step
from Test_Step import test_step

from tqdm.auto import tqdm
import pandas as pd
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

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

NUM_CLASSES = 4
EMBEDDING_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
MAX_LEN = 128
VOCAB_SIZE = len(tokenizer.get_vocab())
MLP_SIZE = 3072

LEARNING_RATE = 5e-5
NUM_EPOCHS = 20

model = TextViT(
    num_classes=NUM_CLASSES,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_transformer_layers=NUM_LAYERS,
    max_seq_len=MAX_LEN,
    vocab_size=VOCAB_SIZE,
    mlp_size=MLP_SIZE,
)

model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

save_dir = "/home/junha/AGNews/Checkpoint/TextViT/"
for epoch in tqdm(range(NUM_EPOCHS)):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    train_loss, train_accuracy = train_step(model, train_dataloader, loss_fn, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    torch.cuda.empty_cache()

    test_loss, test_accuracy, val_f1_score = test_step(model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1 Score: {val_f1_score:.4f}")
    torch.cuda.empty_cache()

    epoch_model_path = os.path.join(save_dir, f"TextViT_{epoch + 1}.pth")
    torch.save(model.state_dict(), epoch_model_path)
    print(f"Model for epoch {epoch + 1} saved to {epoch_model_path}\n")