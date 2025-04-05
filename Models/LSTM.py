import torch.nn
from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, num_classes: int, dropout: float = 0.1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x, attention_mask=None):
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        pooled = torch.mean(lstm_out, dim=1)  # (batch, hidden_size*2)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, num_classes)
        return logits
