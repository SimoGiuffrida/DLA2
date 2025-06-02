import os
import torch
import shutil
from tqdm.auto import tqdm
from metrics import compute_metrics

class SentimentTrainer:
    def __init__(self, model, optimizer, train_loader, eval_loader, tokenizer, device, patience=3, save_frequency=1, output_dir="./sentiment_checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.device = device
        self.patience = patience
        self.save_frequency = save_frequency
        self.output_dir = output_dir
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        os.makedirs(self.output_dir, exist_ok=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint salvato a {checkpoint_path}")

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Modello caricato da {path}")

    def train(self, num_epochs=1):
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
            all_train_preds = []
            all_train_labels = []
            train_progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
            for batch in train_progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                train_progress_bar.set_postfix(loss=loss.item())
            avg_train_loss = total_train_loss / len(self.train_loader)
            train_metrics = compute_metrics({"predictions": all_train_preds, "label_ids": all_train_labels})
            self.train_loss_history.append(avg_train_loss)
            self.train_metrics_history.append(train_metrics)
            print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

            # Valutazione
            self.model.eval()
            total_eval_loss = 0
            all_preds = []
            all_labels = []
            eval_progress_bar = tqdm(self.eval_loader, desc=f"Epoch {epoch+1} Evaluation")
            with torch.no_grad():
                for batch in eval_progress_bar:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_eval_loss += loss.item()
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    eval_progress_bar.set_postfix(loss=loss.item())
            avg_eval_loss = total_eval_loss / len(self.eval_loader)
            metrics = compute_metrics({"predictions": all_preds, "label_ids": all_labels})
            self.val_loss_history.append(avg_eval_loss)
            self.val_metrics_history.append(metrics)
            print(f"Epoch {epoch+1} Validation Loss: {avg_eval_loss:.4f}, Metrics: {metrics}")

            # Early stopping
            if avg_eval_loss < self.best_val_loss:
                self.best_val_loss = avg_eval_loss
                self.counter = 0
                self.best_model = self.model.state_dict()
                print(f"New best validation loss: {self.best_val_loss:.4f}. Resetting early stopping counter.")
            else:
                self.counter += 1
                print(f"Early stopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping!")
                    break
            if (epoch + 1) % self.save_frequency == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1, avg_eval_loss)
        if self.best_model:
            self.model.load_state_dict(self.best_model)
            print("Best model loaded based on validation loss.")
