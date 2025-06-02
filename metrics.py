from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred_dict):
    labels = pred_dict["label_ids"]
    preds = pred_dict["predictions"]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
