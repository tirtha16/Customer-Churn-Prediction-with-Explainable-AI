from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass
class LocalExplanation:
    customer_index: int
    predicted_proba: float
    predicted_label: int
    top_drivers: list[dict]


def _load_pipeline_and_artifacts():
    pipeline = joblib.load(MODEL_DIR / "best_model.joblib")
    artifacts = joblib.load(MODEL_DIR / "artifacts.joblib")
    sample_path = MODEL_DIR / "background_sample.csv"
    if sample_path.exists():
        artifacts["X_train_sample"] = pd.read_csv(sample_path)
    return pipeline, artifacts


def _transform(pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].transform(X)


def _build_shap_explainer(pipeline, background: np.ndarray):
    clf = pipeline.named_steps["classifier"]
    if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
        return shap.TreeExplainer(clf)
    return shap.LinearExplainer(clf, background)


def _shap_values_positive_class(explainer, X_transformed: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def global_summary_plot(max_display: int = 20, sample_size: int = 500) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline, artifacts = _load_pipeline_and_artifacts()
    feature_names = artifacts["feature_names"]
    sample_df = artifacts["X_train_sample"]
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(sample_size, random_state=42)

    X_t = _transform(pipeline, sample_df)
    explainer = _build_shap_explainer(pipeline, X_t)
    shap_values = _shap_values_positive_class(explainer, X_t)

    shap.summary_plot(
        shap_values,
        features=X_t,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    out = REPORTS_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    return out


def explain_customer(customer_row: pd.DataFrame, top_k: int = 6) -> LocalExplanation:
    pipeline, artifacts = _load_pipeline_and_artifacts()
    feature_names = np.array(artifacts["feature_names"])

    proba = float(pipeline.predict_proba(customer_row)[0, 1])
    label = int(proba >= 0.5)

    X_t = _transform(pipeline, customer_row)
    bg = _transform(pipeline, artifacts["X_train_sample"])
    explainer = _build_shap_explainer(pipeline, bg)
    shap_values = _shap_values_positive_class(explainer, X_t)[0]

    order = np.argsort(np.abs(shap_values))[::-1][:top_k]
    drivers = []
    for idx in order:
        drivers.append(
            {
                "feature": str(feature_names[idx]),
                "value": float(X_t[0, idx]),
                "shap_value": float(shap_values[idx]),
                "direction": "increases churn" if shap_values[idx] > 0 else "decreases churn",
            }
        )

    return LocalExplanation(
        customer_index=int(customer_row.index[0]) if len(customer_row.index) else -1,
        predicted_proba=proba,
        predicted_label=label,
        top_drivers=drivers,
    )


def explain_with_lime(customer_row: pd.DataFrame, num_features: int = 6) -> list[tuple[str, float]]:
    pipeline, artifacts = _load_pipeline_and_artifacts()
    feature_names = artifacts["feature_names"]
    background_df = artifacts["X_train_sample"]

    bg_t = _transform(pipeline, background_df)
    x_t = _transform(pipeline, customer_row)

    explainer = LimeTabularExplainer(
        training_data=bg_t,
        feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    def _predict(arr):
        return pipeline.named_steps["classifier"].predict_proba(arr)

    exp = explainer.explain_instance(x_t[0], _predict, num_features=num_features)
    return exp.as_list()


def run_global() -> None:
    out = global_summary_plot()
    print(f"Saved global SHAP summary to {out}")


if __name__ == "__main__":
    run_global()
