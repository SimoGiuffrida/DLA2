# utils.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(pred_dict):
    labels = pred_dict["label_ids"]
    preds = pred_dict["predictions"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def plot_pre_post_rl_comparison(pre_metrics, post_metrics):
    metrics_names = list(pre_metrics.keys())
    pre_values = [pre_metrics[name] for name in metrics_names]
    post_values = [post_metrics[name] for name in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, pre_values, width, label='Pre-RL')
    plt.bar(x + width/2, post_values, width, label='Post-RL')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.show()

def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()