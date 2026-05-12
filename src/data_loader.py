from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_FILENAME = "telco_churn.csv"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    path = _project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_synthetic_telco(n_samples: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, size=n_samples)

    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22]
    )

    def _internet_dep(prob_yes: float) -> np.ndarray:
        choice = rng.choice(["Yes", "No"], size=n_samples, p=[prob_yes, 1 - prob_yes])
        return np.where(internet_service == "No", "No internet service", choice)

    online_security = _internet_dep(0.29)
    online_backup = _internet_dep(0.35)
    device_protection = _internet_dep(0.34)
    tech_support = _internet_dep(0.29)
    streaming_tv = _internet_dep(0.38)
    streaming_movies = _internet_dep(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_charge = np.where(
        internet_service == "Fiber optic",
        rng.normal(80, 12, n_samples),
        np.where(
            internet_service == "DSL",
            rng.normal(55, 10, n_samples),
            rng.normal(22, 5, n_samples),
        ),
    )
    addon_charge = (
        (online_security == "Yes").astype(float) * 5
        + (online_backup == "Yes").astype(float) * 5
        + (device_protection == "Yes").astype(float) * 5
        + (tech_support == "Yes").astype(float) * 5
        + (streaming_tv == "Yes").astype(float) * 10
        + (streaming_movies == "Yes").astype(float) * 10
        + (multiple_lines == "Yes").astype(float) * 5
    )
    monthly_charges = np.clip(base_charge + addon_charge + rng.normal(0, 3, n_samples), 18.0, 120.0)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 25, n_samples), 2)
    total_charges = np.clip(total_charges, 0.0, None)

    logit = (
        -1.4
        + 1.4 * (contract == "Month-to-month").astype(float)
        - 0.9 * (contract == "Two year").astype(float)
        - 0.04 * tenure
        + 0.018 * monthly_charges
        + 0.6 * (internet_service == "Fiber optic").astype(float)
        - 0.4 * (tech_support == "Yes").astype(float)
        - 0.3 * (online_security == "Yes").astype(float)
        + 0.45 * (payment_method == "Electronic check").astype(float)
        + 0.3 * senior
        - 0.2 * (partner == "Yes").astype(float)
        - 0.15 * (dependents == "Yes").astype(float)
        + 0.15 * (paperless == "Yes").astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(size=n_samples) < prob).astype(int)

    customer_ids = [f"CUST-{i:06d}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": np.round(monthly_charges, 2),
            "TotalCharges": total_charges,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )

    return df


def load_data(generate_if_missing: bool = True) -> pd.DataFrame:
    csv_path = data_dir() / RAW_DATA_FILENAME
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif generate_if_missing:
        df = generate_synthetic_telco()
        df.to_csv(csv_path, index=False)
    else:
        raise FileNotFoundError(f"No dataset found at {csv_path}")

    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} rows from {data_dir() / RAW_DATA_FILENAME}")
    print(df["Churn"].value_counts(normalize=True).round(3))
