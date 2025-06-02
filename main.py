import torch
from transformers import AutoModelForSequenceClassification, AdamW
from data import load_and_prepare_data
from trainer import SentimentTrainer
from rl_trainer import RLFeedbackTrainer
from metrics import compute_metrics
from plots import plot_pre_post_rl_comparison

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return compute_metrics({"predictions": all_preds, "label_ids": all_labels})

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, eval_loader, test_loader, tokenizer = load_and_prepare_data()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    trainer = SentimentTrainer(model, optimizer, train_loader, eval_loader, tokenizer, device)
    trainer.train(num_epochs=1)

    print("\nStarting final test evaluation...")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Final Test Metrics: {test_metrics}")

    print("\nEvaluating pre-RL performance...")
    pre_rl_metrics = evaluate_model(model, test_loader, device)
    print("Pre-RL Test Metrics:", pre_rl_metrics)

    print("\nStarting Reinforcement Learning Phase...")
    rl_trainer = RLFeedbackTrainer(model, optimizer, train_loader, tokenizer, device)
    rl_trainer.rl_training_phase(num_epochs=2)

    print("\nEvaluating post-RL performance...")
    post_rl_metrics = evaluate_model(model, test_loader, device)
    print("Post-RL Test Metrics:", post_rl_metrics)

    plot_pre_post_rl_comparison(pre_rl_metrics, post_rl_metrics)

if __name__ == "__main__":
    main()
