import torch
from tqdm.auto import tqdm

class RLFeedbackTrainer:
    def __init__(self, model, optimizer, train_dataloader, tokenizer, device, reward_scale=1.0):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.reward_scale = reward_scale

    def calculate_reward(self, predictions, true_labels):
        correct = (predictions == true_labels).float()
        return (correct * 2 - 1) * self.reward_scale

    def rl_training_phase(self, num_epochs=2):
        self.model.train()
        for epoch in range(num_epochs):
            total_reward = 0
            rl_progress_bar = tqdm(self.train_dataloader, desc=f"RL Epoch {epoch+1}")
            for batch in rl_progress_bar:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device)
                }
                true_labels = batch["labels"].to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                policy_dist = torch.distributions.Categorical(logits=logits)
                actions = policy_dist.sample()
                rewards = self.calculate_reward(actions, true_labels)
                log_probs = policy_dist.log_prob(actions)
                loss = -torch.mean(log_probs * rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_reward += rewards.sum().item()
                rl_progress_bar.set_postfix(avg_reward=total_reward/(len(rl_progress_bar)+1))
