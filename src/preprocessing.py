from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ENGINEERED_NUMERIC = [
    "AvgChargesPerMonth",
    "ChargeToTenureRatio",
    "NumAddOns",
    "TenureGroupOrdinal",
]


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    safe_tenure = out["tenure"].replace(0, 1)
    out["AvgChargesPerMonth"] = (out["TotalCharges"] / safe_tenure).round(3)
    out["ChargeToTenureRatio"] = (out["MonthlyCharges"] / safe_tenure).round(4)

    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    out["NumAddOns"] = sum((out[c] == "Yes").astype(int) for c in addon_cols)

    tenure_bins = [-1, 12, 24, 48, np.inf]
    out["TenureGroupOrdinal"] = pd.cut(
        out["tenure"], bins=tenure_bins, labels=[0, 1, 2, 3]
    ).astype(int)

    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=[c for c in (TARGET_COL, ID_COL) if c in df.columns])
    return X, y


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())
