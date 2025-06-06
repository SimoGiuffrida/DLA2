﻿Sure! Here's the full translation of your project README into English:

---

# 🤖 DLA2 Project - Emotion Classification with BERT and RL

<div align="center">

### Advanced Emotion Classification System with RL Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/transformers)

</div>

## 📝 Description

This project implements an **emotion classification** system for text using **Transformer models (BERT)** and advanced **Reinforcement Learning** techniques for fine-tuning. The new modular architecture ensures maximum flexibility and code reusability.

## 🗂 Dataset: dair-ai/emotion

The [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset contains over 20,000 English sentences annotated with 6 main emotions:

| Emotion     | Example                            | Count |
| ----------- | ---------------------------------- | ----- |
| 😊 Joy      | "I am feeling great today!"        | 5,369 |
| 😢 Sadness  | "I lost my job yesterday"          | 5,185 |
| 😠 Anger    | "This product is a complete scam!" | 2,159 |
| 😨 Fear     | "I heard strange noises outside"   | 2,247 |
| ❤️ Love     | "I adore spending time with you"   | 1,584 |
| 😲 Surprise | "You got me a puppy?!"             | 2,187 |

```python
from datasets import load_dataset
dataset = load_dataset("dair-ai/emotion")
```

## 🏗 Updated Project Structure

```plaintext
dla2-project/
├── main.py              # Main execution script
├── models.py            # Model definitions (ActorCriticBERT)
├── data_utils.py        # Dataset handling and preprocessing
├── trainers.py          # Supervised and RL training
├── utils.py             # Metrics and utility functions
├── requirements.txt     # Project dependencies
├── notebooks/           # Jupyter notebooks for analysis
│   └── Emotion_Analysis.ipynb
└── outputs/             # Training outputs and results
    ├── checkpoints/     # Saved models
    ├── metrics/         # Evaluation metrics
    └── plots/           # Visualizations
```

## ⚙️ Technical Requirements

| Component    | Version | Description                     |
| ------------ | ------- | ------------------------------- |
| Python       | 3.8+    | Main programming language       |
| PyTorch      | 2.0+    | Deep learning framework         |
| Transformers | 4.30+   | State-of-the-art NLP models     |
| Datasets     | 2.12+   | Hugging Face dataset management |
| Scikit-learn | 1.2+    | Evaluation metrics              |
| Matplotlib   | 3.7+    | Result visualization            |
| TQDM         | 4.65+   | Progress bars                   |

### 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/your-user/dla2-project.git
   cd dla2-project
   ```

2. Run full training:

   ```bash
   python main.py --epochs 3 --batch_size 16 --rl_epochs 2
   ```

3. Advanced options:

   ```bash
   python main.py \
     --model_name "bert-large-uncased" \
     --learning_rate 3e-5 \
     --rl_learning_rate 1e-6 \
     --output_dir "results/experiment1"
   ```


## 🧩 Module Details

### models.py

Defines the advanced Actor-Critic architecture based on BERT:

```python
class ActorCriticBERT(PreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.policy_head = nn.Linear(config.hidden_size, num_labels)  # Actor head
        self.value_head = nn.Linear(config.hidden_size, 1)  # Critic head
```

### data\_utils.py

Advanced data handling:

* Tokenization with dynamic padding
* Dataset with hidden labels for RL
* PyTorch-friendly formatting

```python
class HiddenLabelsDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "input_ids": ...,
            "attention_mask": ...,
            "hidden_labels": ...  # For RL
        }
```

### trainers.py

Implements two training strategies:

1. **SentimentTrainer**: Supervised training with early stopping
2. **RLFeedbackTrainer**: Fine-tuning via Reinforcement Learning

```python
class RLFeedbackTrainer:
    def rl_training_phase(self, num_epochs):
        # PPO algorithm implementation
        for epoch in range(num_epochs):
            # Compute advantage and policy gradient
            # Optimization with PPO clipping
```

### utils.py

Essential functions for:

* Metric computation (accuracy, F1, precision, recall)
* Result visualization
* Model saving and loading

```python
def plot_pre_post_rl_comparison(pre_metrics, post_metrics):
    # Visual comparison of performance before/after RL
```

### main.py

Complete workflow orchestration:

```python
def main():
    # 1. Initial configuration
    # 2. Data loading
    # 3. Supervised training
    # 4. Pre-RL evaluation
    # 5. RL fine-tuning
    # 6. Post-RL evaluation
    # 7. Result saving
```


## 🧪 Advanced Features

1. **Smart Early Stopping**:

   * Monitors validation loss
   * Automatically saves best model
   * Auto-recovery from overfitting

2. **Configurable Reward System**:

   ```python
   def compute_rewards(self, pred_actions, true_labels):
       # Custom reward function
       return torch.where(pred_actions == true_labels, 1.0, -0.1)
   ```

3. **PPO Optimization**:

   * Policy clipping
   * KL regularization
   * Entropy bonus

## 💡 Use Cases

1. Advanced sentiment analysis
2. Emotion-aware recommender systems
3. Empathetic chatbots
4. Mental wellness monitoring
5. Emotion-driven market research

---
# 📊 Results

### *Comparative Analysis of Metrics Before and After Applying Reinforcement Learning*

| Metrics   | Pre-RL | Post-RL | Difference |
| --------- | ------ | ------- | ----------- |
| Accuracy  | 0.9205  | 0.931   | +1.14%       |
| F1-Score  | 0.9227  | 0.9298   | +0.77%       |
| Precision | 0.9325  | 0.9305   | −021%       |
| Recall    | 0.9205  | 0.931   | +1.14%       |


![Pre-Post RL Comparison](grafici/download.png)

This bar chart compares four performance metrics (accuracy, f1-score, precision, recall) of the model before and after the Reinforcement Learning (RL) phase. The blue bars indicate "Pre-RL" performance, while the orange bars indicate "Post-RL" performance.

* **Performance Degradation After RL**: The chart shows a clear and dramatic decrease in all performance metrics (accuracy, f1, precision, recall) following the application of the Reinforcement Learning phase. For instance, accuracy drops from around 0.9 (Pre-RL) to about 0.4 (Post-RL).
* **Ineffective or Harmful RL**: This outcome suggests that the Reinforcement Learning phase, as currently implemented, not only failed to improve the model’s performance, but significantly degraded it.

#### Loss Analysis

![Loss Function](grafici/download%20\(2\).png)

#### *Class Distribution in the Test Set*

This histogram displays the distribution of different emotion classes (sadness, joy, love, anger, fear, surprise) within the test set. Each bar represents the count of instances for a specific class.

#### *Prediction Distribution*

![Distribution](grafici/download%20\(3\).png)

#### *Confusion Matrix of Predictions*

Each row represents instances in a true class, while each column represents instances in a predicted class. Values along the main diagonal indicate correct predictions (True Positives).

The values on the diagonal (548 for sadness, 621 for joy, 157 for love, 268 for anger, 179 for fear, 62 for surprise) indicate a high number of correct classifications for the most frequent classes.

60 instances of "love" were misclassified as "joy," which is one of the majority classes. This is a clear example of the model’s bias toward more represented classes.

---

## 🔗 Pretrained Model on Hugging Face

You can access and download the final fine-tuned model directly from the Hugging Face Model Hub:

👉 [**SimoGiuffrida/SentimentRL**](https://huggingface.co/SimoGiuffrida/SentimentRL)

This repository includes:

* Model weights (`pytorch_model.bin`)
* Configuration files (`config.json`)
* README with usage instructions

### 📥 Example Usage

You can load the model in just a few lines of code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("SimoGiuffrida/SentimentRL")

text = "I feel so happy today!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(dim=-1).item()
```

Use this for quick inference, benchmarking, or integration into other NLP pipelines.

---

<div align="center">

**Department of Computer Science**
*University of Cagliari*
Academic Year 2024–2025

</div>
