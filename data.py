import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_and_prepare_data(tokenizer_name="bert-base-uncased", batch_size=8):
    dataset = load_dataset("dair-ai/emotion")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_eval = dataset["validation"].map(tokenize_function, batched=True)
    tokenized_test = dataset["test"].map(tokenize_function, batched=True)

    for split in [tokenized_train, tokenized_eval, tokenized_test]:
        split.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    train_loader = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size)
    eval_loader = DataLoader(tokenized_eval, batch_size=batch_size)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)

    return train_loader, eval_loader, test_loader, tokenizer
