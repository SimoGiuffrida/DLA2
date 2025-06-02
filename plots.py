import numpy as np
import matplotlib.pyplot as plt

def plot_pre_post_rl_comparison(pre_metrics, post_metrics):
    metrics_names = list(pre_metrics.keys())
    pre_values = [pre_metrics[name] for name in metrics_names]
    post_values = [post_metrics[name] for name in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, pre_values, width, label='Pre-RL')
    plt.bar(x + width/2, post_values, width, label='Post-RL')
    plt.title('Performance Comparison Before and After RL')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True)
    plt.show()
