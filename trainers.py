# trainers.py
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical
from tqdm.auto import tqdm
from models import ActorCriticBERT

class SentimentTrainer:
    def __init__(self, model, optimizer, train_dl, eval_dl, tokenizer, device, patience=3):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dl
        self.eval_dataloader = eval_dl
        self.tokenizer = tokenizer
        self.device = device
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model = None

    def save_checkpoint(self, epoch, val_loss, output_dir="./checkpoints"):
        os.makedirs(output_dir, exist_ok=True)
        path = f"{output_dir}/epoch_{epoch}_loss_{val_loss:.4f}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")

class RLFeedbackTrainer:
    def __init__(self, supervised_model, train_dl, tokenizer, device, rl_hparams=None):
        self.train_dataloader = train_dl
        self.tokenizer = tokenizer
        self.device = device
        
        # Inizializzazione modello Actor-Critic
        self.actor_critic_model = ActorCriticBERT(
            config=supervised_model.config,
            num_labels=supervised_model.config.num_labels
        )
        
        # Copia pesi dal modello supervisionato
        state_dict = supervised_model.state_dict()
        self.actor_critic_model.load_state_dict({
            k: v for k, v in state_dict.items() 
            if not k.startswith('classifier')
        }, strict=False)
        self.actor_critic_model.to(device)
        
        # Modello di riferimento per KL divergence
        self.reference_model = copy.deepcopy(self.actor_critic_model)
        self.reference_model.eval()
        
        # Configurazione ottimizzatore
        self.optimizer = AdamW(
            self.actor_critic_model.parameters(), 
            lr=rl_hparams.get("lr", 5e-6)
        )
        
        # Iperparametri RL
        self.hparams = {
            "kl_beta": 0.05,
            "ppo_epsilon": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "ppo_epochs_per_batch": 3,
            **rl_hparams
        }

    def compute_rewards(self, pred_actions, true_labels):
        return (pred_actions == true_labels).float() * 2 - 1

    def rl_training_phase(self, num_epochs):
        for epoch in range(num_epochs):
            self.actor_critic_model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"RL Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Preparazione batch dati
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                true_labels = batch["labels"].to(self.device)
                
                # ... [codice di training PPO] ...
                
        print("RL training completed")