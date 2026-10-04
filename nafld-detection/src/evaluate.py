"""
evaluate.py - metrics and plots used to judge the models.

Metrics (all from the synopsis):
  Accuracy  - share of all predictions that were right
  Precision - of the people flagged positive, how many really were
  Recall    - of the people who really were positive, how many we caught
  F1        - balance of precision and recall
  ROC-AUC   - how well the model ranks positives above negatives (0.5 = coin flip)
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.config import RANDOM_STATE


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """All five metrics for one model, rounded for readability."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def plot_roc_curves(fitted_models: dict, X_test, y_test, save_path):
    """One ROC curve per model on a single chart."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in fitted_models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - all models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(model, X_test, y_test, labels, title, save_path):
    """Confusion matrix of one model (rows = truth, columns = prediction)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=labels, cmap="Reds", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, X_test, y_test, features, title, save_path) -> pd.DataFrame:
    """
    Permutation importance: shuffle one feature at a time and see how much the
    ROC-AUC drops. A big drop = the model relies on that feature.
    Works for ANY model type, which keeps the comparison fair and simple.
    """
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": features, "importance": result.importances_mean})
                  .sort_values("importance"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance["feature"], importance["importance"], color="#C0392B")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)
