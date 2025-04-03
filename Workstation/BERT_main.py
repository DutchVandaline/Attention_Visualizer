import torch
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertConfig, BertModel
from torch.utils.data import DataLoader

from AGNewsDataset import AGNewsDataset
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
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

NUM_CLASSES = 4

class BertClassifier(nn.Module):
    def __init__(self, num_labels=NUM_CLASSES):
        super(BertClassifier, self).__init__()
        config = BertConfig()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

model = BertClassifier(num_labels=NUM_CLASSES)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-5)
NUM_EPOCHS = 20

save_dir = "/home/junha/AGNews/Checkpoint/BERT/"
os.makedirs(save_dir, exist_ok=True)

for epoch in tqdm(range(NUM_EPOCHS)):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss, train_accuracy = train_step(model, train_dataloader, loss_fn, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    torch.cuda.empty_cache()

    test_loss, test_accuracy, val_f1_score = test_step(model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1 Score: {val_f1_score:.4f}")
    torch.cuda.empty_cache()

    epoch_model_path = os.path.join(save_dir, f"BERT_{epoch + 1}.pth")
    torch.save(model.state_dict(), epoch_model_path)
    print(f"Model for epoch {epoch + 1} saved to {epoch_model_path}\n")
