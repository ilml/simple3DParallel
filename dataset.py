import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, tokenizer, file_path=None):
        self.data = ["this is a test"] * 100000
        self.max_length = 10   
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return inputs