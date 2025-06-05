# main.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from trainers import SentimentTrainer, RLFeedbackTrainer
from data_utils import tokenize_function, HiddenLabelsDataset
from utils import compute_metrics, plot_pre_post_rl_comparison, plot_confusion_matrix

# Configurazione iniziale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Caricamento e preparazione dataset
dataset = load_dataset("dair-ai/emotion")
tokenized_ds = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
tokenized_ds = tokenized_ds.rename_column("label", "labels")

# Creazione DataLoader
train_dl = DataLoader(tokenized_ds["train"], batch_size=8, shuffle=True)
eval_dl = DataLoader(tokenized_ds["validation"], batch_size=8)
test_dl = DataLoader(tokenized_ds["test"], batch_size=8)

# Training supervisionato
optimizer = AdamW(model.parameters(), lr=5e-5)
trainer = SentimentTrainer(model, optimizer, train_dl, eval_dl, tokenizer, device)

for epoch in range(3):
    # Training loop...
    # Validation loop...

# Training RL
rl_trainer = RLFeedbackTrainer(
    model,
    train_dl,
    tokenizer,
    device,
    rl_hparams={"lr": 3e-6, "kl_beta": 0.02}
)
rl_trainer.rl_training_phase(num_epochs=2)

# Valutazione e plotting
pre_rl_metrics = evaluate_model(model, test_dl, device)
post_rl_metrics = evaluate_model(rl_trainer.actor_critic_model, test_dl, device)
plot_pre_post_rl_comparison(pre_rl_metrics, post_rl_metrics)
plot_confusion_matrix(test_labels, test_preds, class_names)