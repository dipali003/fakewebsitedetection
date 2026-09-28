"""Evaluation helpers: metrics, confusion matrices and ROC curves."""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

import config

PLOTS_DIR = config.ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_binary(y_true, proba: np.ndarray, threshold: float = config.THRESHOLD) -> dict:
    y_true = np.asarray(y_true).ravel().astype(int)
    pred = (proba >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "n": int(len(y_true)),
    }


def plot_confusion(y_true, proba: np.ndarray, title: str, out_name: str,
                   threshold: float = config.THRESHOLD) -> Path:
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).ravel().astype(int), pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Legit", "Phish"])
    ax.set_yticks([0, 1], labels=["Legit", "Phish"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i, j in itertools.product(range(2), range(2)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out = PLOTS_DIR / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_roc(y_true, proba: np.ndarray, title: str, out_name: str) -> Path:
    fpr, tpr, _ = roc_curve(np.asarray(y_true).ravel().astype(int), proba)
    auc = roc_auc_score(np.asarray(y_true).ravel().astype(int), proba)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout()
    out = PLOTS_DIR / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_training_history(hist: dict, title: str, out_name: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(hist.get("loss", []), label="train")
    axes[0].plot(hist.get("val_loss", []), label="val")
    axes[0].set_title(f"{title} — loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    acc_key = "accuracy" if "accuracy" in hist else "acc"
    axes[1].plot(hist.get(acc_key, []), label="train")
    axes[1].plot(hist.get("val_" + acc_key, []), label="val")
    axes[1].set_title(f"{title} — accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    out = PLOTS_DIR / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
