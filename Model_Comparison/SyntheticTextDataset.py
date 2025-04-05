import torch
import torch.utils.data as data

class SyntheticTextDataset(data.Dataset):
    def __init__(self, num_samples, max_seq_len, vocab_size):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 임의의 토큰 시퀀스
        tokens = torch.randint(0, self.vocab_size, (self.max_seq_len,))
        # 임의의 이진 라벨
        label = torch.randint(0, 2, (1,)).item()
        return tokens, torch.tensor(label, dtype=torch.long)
