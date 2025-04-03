import torch
from torch.utils.data import Dataset


class AGNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.texts = (data['Title'] + " " + data['Description']).tolist()
        self.labels = (data['Class Index'] - 1).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # (1, max_len) -> (max_len)
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
