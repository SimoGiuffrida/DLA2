# data_utils.py
from datasets import load_dataset
from torch.utils.data import Dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

class HiddenLabelsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "hidden_labels": item["labels"]
        }

    def __len__(self):
        return len(self.dataset)