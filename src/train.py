from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from .data_loader import load_data
from .preprocessing import (
    build_preprocessor,
    engineer_features,
    get_feature_names,
    split_xy,
)

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


@dataclass
class ModelMetrics:
    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    confusion_matrix: list[list[int]]
    classification_report: dict


def _candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _build_pipeline(estimator) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", estimator),
        ]
    )


def _evaluate(name, pipeline, X_test, y_test, X_train, y_train) -> ModelMetrics:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)

    return ModelMetrics(
        name=name,
        roc_auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        f1=float(f1_score(y_test, preds)),
        cv_roc_auc_mean=float(cv_scores.mean()),
        cv_roc_auc_std=float(cv_scores.std()),
        confusion_matrix=confusion_matrix(y_test, preds).tolist(),
        classification_report=classification_report(y_test, preds, output_dict=True),
    )


def train_all() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    df = engineer_features(df)
    X, y = split_xy(df)
    print(f"Rows: {len(df):,} | churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    results: dict[str, ModelMetrics] = {}
    fitted: dict[str, ImbPipeline] = {}

    for name, estimator in _candidate_models().items():
        print(f"\nTraining {name}")
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        metrics = _evaluate(name, pipeline, X_test, y_test, X_train, y_train)
        results[name] = metrics
        fitted[name] = pipeline
        print(
            f"  ROC-AUC: {metrics.roc_auc:.4f}  PR-AUC: {metrics.pr_auc:.4f}  "
            f"F1: {metrics.f1:.4f}  CV ROC-AUC: "
            f"{metrics.cv_roc_auc_mean:.4f} +/- {metrics.cv_roc_auc_std:.4f}"
        )

    best_name = max(results, key=lambda k: results[k].roc_auc)
    best_pipeline = fitted[best_name]
    print(f"\nBest model: {best_name}")

    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])

    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

    sample = X_train.sample(min(500, len(X_train)), random_state=RANDOM_STATE)
    sample.to_csv(MODEL_DIR / "background_sample.csv", index=False)

    joblib.dump(
        {
            "feature_names": list(feature_names),
            "best_model_name": best_name,
        },
        MODEL_DIR / "artifacts.joblib",
    )
    for name, pipe in fitted.items():
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

    summary = {
        "best_model": best_name,
        "metrics": {k: asdict(v) for k, v in results.items()},
        "feature_names": feature_names,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": float(y.mean()),
    }
    with (REPORTS_DIR / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    train_all()
