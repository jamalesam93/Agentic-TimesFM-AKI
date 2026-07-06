import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set academic styling globally
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2
})

def generate_loss_curve(output_dir):
    epochs = [1, 2, 3, 4, 5]
    loss = [0.4355, 0.3257, 0.3078, 0.2949, 0.2876]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss, marker='s', linestyle='-', color='#000000', linewidth=2, markersize=8, label='Training Loss')
    
    ax.set_title('Figure 1: Fine-Tuning Convergence of TimesFM LoRA Adapter\n(Evaluation on Real-World HDHI Parameters)', pad=20, fontweight='bold')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Mean Absolute Error (Proxy Loss)')
    ax.set_xticks(epochs)
    
    # Add vertical padding so the top annotation (0.4355) isn't cut off by the top border
    ax.set_ylim(0.27, 0.46)
    
    ax.grid(True, linestyle=':', alpha=0.6, color='black')
    
    # Add value labels
    for i, txt in enumerate(loss):
        ax.annotate(f"{txt:.4f}", (epochs[i], loss[i]), textcoords="offset points", xytext=(0,12), ha='center', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timesfm_loss_curve.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print("Generated academic timesfm_loss_curve.png")

def generate_confusion_matrix(output_dir):
    # TP=97, FP=0, FN=1, TN=102
    # Format: [[TN, FP], [FN, TP]]
    cm = np.array([[102, 0], [1, 97]])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Custom academic monochrome-blue colormap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (No AKI)', 'Positive (AKI)'], 
                yticklabels=['Negative (No AKI)', 'Positive (AKI)'],
                annot_kws={"size": 18, "fontweight": "bold"},
                cbar_kws={'label': 'Number of Trajectories'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_title('Figure 2: Gemma-4 12B Clinical Accuracy\nConfusion Matrix (n=200 Holdout)', pad=20, fontweight='bold')
    ax.set_xlabel('Predicted Clinical Diagnosis')
    ax.set_ylabel('Actual Ground-Truth Diagnosis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gemma_confusion_matrix.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print("Generated academic gemma_confusion_matrix.png")

if __name__ == "__main__":
    output_dir = "reports/graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating academic graphs for PAPER...")
    generate_loss_curve(output_dir)
    generate_confusion_matrix(output_dir)
    print(f"All graphs saved to {output_dir}/")

