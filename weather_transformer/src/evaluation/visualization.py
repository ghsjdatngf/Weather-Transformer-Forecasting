import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src import config


def plot_loss_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Total Loss (LM + Forecast Classification)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_lm_loss"], label="Train LM Loss")
    axes[1].plot(history["val_lm_loss"], label="Val LM Loss")
    axes[1].set_title("Language Modeling Loss (Cross-Entropy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss curves saved to {path}")


def plot_confusion_matrix(cm, label_names, title, filename):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {path}")
    return path
