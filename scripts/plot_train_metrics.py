import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(file_path):
    """Load metrics from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_training_metrics():
    # Define experiment paths and labels
    experiments = {
        '1 Patient': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/1_patient_rationale/training_metrics.json',
        '5 Patients': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/5_patient_rationale/training_metrics.json',
        '10 Patients': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/10_patient_rationale/training_metrics.json',
        '15 Patients': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/15_patient_rationale/training_metrics.json',
        '20 Patient': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/20_patient/training_metrics.json',
        '30 Patients': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/30_patient/training_metrics.json',
        '40 Patients': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/40_patient/training_metrics.json',
        '50 Patients': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/50_patient/training_metrics.json',

        'Full Shot': '/home/nicoc/Vand/Vand3D/experiments_3D/3D/full_shot/training_metrics.json'
    }
    
    # Load all data
    data = {}
    for label, path in experiments.items():
        try:
            data[label] = load_metrics(path)
            print(f"Loaded {len(data[label])} epochs for {label}")
        except FileNotFoundError:
            print(f"Warning: Could not find {path}")
            continue
    
    # Create subplots (2x3 layout)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 10))
    
    # Define colors for each experiment
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c','#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#d62728']
    
    # Plot 1: Training Loss
    ax1.set_title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    for i, (label, metrics) in enumerate(data.items()):
        epochs = [m['epoch'] for m in metrics]
        train_loss = [m['train_loss'] for m in metrics]
        ax1.plot(epochs, train_loss, marker='o', label=label, color=colors[i], linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Dice Score
    ax2.set_title('Validation Dice Score vs Epochs', fontsize=14, fontweight='bold')
    for i, (label, metrics) in enumerate(data.items()):
        epochs = [m['epoch'] for m in metrics]
        val_dice = [m['val_dice_mean'] for m in metrics]
        val_dice_std = [m['val_dice_std'] for m in metrics]
        ax2.errorbar(epochs, val_dice,  marker='o', label=label, 
                    color=colors[i], linewidth=2, capsize=3, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation F1 Score (NEW)
    """ 
    ax3.set_title('Validation F1 Score vs Epochs', fontsize=14, fontweight='bold')
    for i, (label, metrics) in enumerate(data.items()):
        epochs = [m['epoch'] for m in metrics]
        val_f1 = [m['val_f1_mean'] for m in metrics]
        val_f1_std = [m['val_f1_std'] for m in metrics]
        ax3.errorbar(epochs, val_f1, marker='o', label=label, 
                    color=colors[i], linewidth=2, capsize=3, alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    """
    
    # Plot 4: Focal Loss
    ax4.set_title('Training Focal Loss vs Epochs', fontsize=14, fontweight='bold')
    for i, (label, metrics) in enumerate(data.items()):
        epochs = [m['epoch'] for m in metrics]
        focal_loss = [m['train_focal_loss'] for m in metrics]
        ax4.plot(epochs, focal_loss, marker='o', label=label, color=colors[i], linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Focal Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Dice Loss
    ax5.set_title('Training Dice Loss vs Epochs', fontsize=14, fontweight='bold')
    for i, (label, metrics) in enumerate(data.items()):
        epochs = [m['epoch'] for m in metrics]
        dice_loss = [m['train_dice_loss'] for m in metrics]
        ax5.plot(epochs, dice_loss, marker='o', label=label, color=colors[i], linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Dice Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Hide the 6th subplot
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_comparison_all_k.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final metrics summary
    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    for label, metrics in data.items():
        final_epoch = metrics[-1]
        print(f"\n{label}:")
        print(f"  Final Train Loss: {final_epoch['train_loss']:.4f}")
        print(f"  Final Val Dice: {final_epoch['val_dice_mean']:.4f} ± {final_epoch['val_dice_std']:.4f}")
        print(f"  Final Val F1: {final_epoch['val_f1_mean']:.4f} ± {final_epoch['val_f1_std']:.4f}")
        print(f"  Best Val Dice: {max(m['val_dice_mean'] for m in metrics):.4f}")
        print(f"  Best Val F1: {max(m['val_f1_mean'] for m in metrics):.4f}")

if __name__ == "__main__":
    plot_training_metrics()