from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .data_loader import load_data
from .preprocessing import engineer_features, split_xy

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _load_artifacts():
    pipeline = joblib.load(MODEL_DIR / "best_model.joblib")
    artifacts = joblib.load(MODEL_DIR / "artifacts.joblib")
    return pipeline, artifacts


def _prepare_test_split():
    df = engineer_features(load_data())
    X, y = split_xy(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    return X_test, y_test


def plot_curves() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline, artifacts = _load_artifacts()
    X_test, y_test = _prepare_test_split()

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    best_name = artifacts["best_model_name"]

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(top_k: int = 20) -> None:
    pipeline, artifacts = _load_artifacts()
    clf = pipeline.named_steps["classifier"]
    feature_names = np.array(artifacts["feature_names"])

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        title = "Feature Importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).ravel()
        title = "Feature Importance (|Coefficient|)"
    else:
        return

    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    ax.barh(range(len(order)), importances[order][::-1], color="#3b7dd8")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(feature_names[order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def run() -> None:
    plot_curves()
    plot_feature_importance()
    print(f"Saved plots to {REPORTS_DIR}")


if __name__ == "__main__":
    run()
