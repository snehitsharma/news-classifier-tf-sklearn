import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def save_confusion(y_true, y_pred, names, path):
    """Save a confusion-matrix image and return the matrix."""
    labels = list(range(len(names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def top_confusions(cm, names, k=5):
    """The k most frequent wrong (true, predicted) pairs, for error analysis."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:k]
    pairs = [np.unravel_index(i, off.shape) for i in flat]
    return [(names[i], names[j], int(off[i, j])) for i, j in pairs if off[i, j] > 0]
