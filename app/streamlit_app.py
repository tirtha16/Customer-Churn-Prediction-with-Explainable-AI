from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_data
from src.explain import explain_customer, explain_with_lime
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
)

MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


@st.cache_resource
def _load_pipeline():
    return joblib.load(MODEL_DIR / "best_model.joblib")


@st.cache_resource
def _load_artifacts():
    artifacts = joblib.load(MODEL_DIR / "artifacts.joblib")
    sample_path = MODEL_DIR / "background_sample.csv"
    if sample_path.exists():
        artifacts["X_train_sample"] = pd.read_csv(sample_path)
    return artifacts


@st.cache_data
def _load_dataset():
    return engineer_features(load_data())


@st.cache_data
def _load_training_summary():
    path = REPORTS_DIR / "training_summary.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def _score_dataframe(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    proba = pipeline.predict_proba(df)[:, 1]
    out = df.copy()
    out["ChurnProbability"] = proba
    out["RiskTier"] = pd.cut(
        proba,
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return out


def page_overview(df, scored, summary):
    st.header("Portfolio Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Observed Churn", f"{(df['Churn'].str.lower() == 'yes').mean():.1%}")
    c3.metric("Predicted High-Risk", f"{(scored['RiskTier'] == 'High').mean():.1%}")
    if summary:
        best = summary["best_model"]
        roc = summary["metrics"][best]["roc_auc"]
        c4.metric(f"Best Model ({best})", f"ROC-AUC {roc:.3f}")

    st.subheader("Predicted churn probability distribution")
    fig = px.histogram(
        scored, x="ChurnProbability", nbins=40, color="RiskTier",
        color_discrete_map={"Low": "#3ba776", "Medium": "#f4a261", "High": "#d6504a"},
    )
    fig.update_layout(bargap=0.02, height=380)
    st.plotly_chart(fig, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Churn rate by contract type")
        rate = df.groupby("Contract")["Churn"].apply(lambda s: (s.str.lower() == "yes").mean())
        st.bar_chart(rate)
    with cB:
        st.subheader("Tenure vs monthly charges")
        sample = scored.sample(min(2000, len(scored)), random_state=0)
        fig = px.scatter(
            sample, x="tenure", y="MonthlyCharges", color="ChurnProbability",
            color_continuous_scale="RdYlGn_r", opacity=0.7, height=380,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_model_performance(summary):
    st.header("Model Performance")
    if not summary:
        st.warning("No training summary found. Run `python main.py train` first.")
        return

    rows = []
    for name, m in summary["metrics"].items():
        rows.append({
            "Model": name,
            "ROC-AUC": round(m["roc_auc"], 4),
            "PR-AUC": round(m["pr_auc"], 4),
            "F1": round(m["f1"], 4),
            "CV ROC-AUC (mean)": round(m["cv_roc_auc_mean"], 4),
            "CV ROC-AUC (std)": round(m["cv_roc_auc_std"], 4),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.caption(f"Best model: {summary['best_model']}")

    st.subheader("Plots")
    cols = st.columns(2)
    plots = [
        ("ROC Curve", REPORTS_DIR / "roc_curve.png"),
        ("Precision-Recall", REPORTS_DIR / "pr_curve.png"),
        ("Confusion Matrix", REPORTS_DIR / "confusion_matrix.png"),
        ("Feature Importance", REPORTS_DIR / "feature_importance.png"),
    ]
    for i, (title, path) in enumerate(plots):
        with cols[i % 2]:
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)
            else:
                st.info(f"{title} not generated. Run `python main.py evaluate`.")


def page_high_risk(scored):
    st.header("High-Risk Customers")
    high = scored[scored["RiskTier"] == "High"].sort_values("ChurnProbability", ascending=False)
    st.write(f"{len(high):,} customers with churn probability >= 0.60.")

    cols_to_show = [
        c for c in [
            "customerID", "tenure", "Contract", "InternetService",
            "MonthlyCharges", "PaymentMethod", "ChurnProbability",
        ] if c in scored.columns
    ]
    st.dataframe(high[cols_to_show].head(200), use_container_width=True)

    st.download_button(
        "Download high-risk list (CSV)",
        data=high[cols_to_show].to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )


def page_single_prediction(pipeline, df):
    st.header("Score a Customer")

    mode = st.radio("Input mode", ["Pick existing customer", "Build manually"], horizontal=True)

    if mode == "Pick existing customer":
        idx = st.selectbox("Customer index", df.index.tolist()[:500])
        row = df.loc[[idx]].drop(columns=[c for c in ("Churn", "customerID") if c in df.columns])
    else:
        with st.form("manual"):
            inputs = {}
            for col in NUMERIC_FEATURES:
                default = float(df[col].median())
                inputs[col] = st.number_input(col, value=default)
            for col in CATEGORICAL_FEATURES:
                options = sorted(df[col].unique().tolist())
                inputs[col] = st.selectbox(col, options)
            submitted = st.form_submit_button("Build customer")
        if not submitted:
            st.stop()
        row = pd.DataFrame([inputs])
        row = engineer_features(row.assign(Churn="No"))
        row = row.drop(columns=[c for c in ("Churn", "customerID") if c in row.columns])

    proba = float(pipeline.predict_proba(row)[0, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{proba:.1%}")
    c2.metric("Predicted label", "Churn" if proba >= 0.5 else "Retain")
    c3.metric("Risk tier", "High" if proba >= 0.6 else ("Medium" if proba >= 0.3 else "Low"))

    st.subheader("SHAP top drivers")
    explanation = explain_customer(row, top_k=6)
    drivers = pd.DataFrame(explanation.top_drivers)
    drivers["impact"] = drivers["shap_value"].round(3)
    fig = px.bar(
        drivers.sort_values("impact"),
        x="impact", y="feature", orientation="h",
        color="impact", color_continuous_scale="RdBu_r",
        hover_data=["direction", "value"],
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("LIME explanation"):
        try:
            lime_pairs = explain_with_lime(row, num_features=6)
            st.table(pd.DataFrame(lime_pairs, columns=["Condition", "Weight"]))
        except Exception as e:
            st.info(f"LIME unavailable: {e}")


def page_global_explain():
    st.header("Global Feature Drivers")
    path = REPORTS_DIR / "shap_summary.png"
    if path.exists():
        st.image(str(path), caption="SHAP summary", use_container_width=True)
    else:
        st.info("SHAP summary not generated. Run `python main.py explain`.")


def main():
    st.set_page_config(
        page_title="Customer Churn Prediction with Explainable AI",
        layout="wide",
    )
    st.title("Customer Churn Prediction with Explainable AI")

    if not (MODEL_DIR / "best_model.joblib").exists():
        st.error("No trained model found. Run `python main.py all` from the project root first.")
        st.stop()

    pipeline = _load_pipeline()
    _ = _load_artifacts()
    df = _load_dataset()
    summary = _load_training_summary()

    scoring_df = df.drop(columns=[c for c in ("Churn",) if c in df.columns])
    scored = _score_dataframe(pipeline, scoring_df)
    scored = scored.assign(Churn=df["Churn"].values, customerID=df["customerID"].values)

    page = st.sidebar.radio(
        "Section",
        ["Portfolio Overview", "Model Performance", "High-Risk Customers",
         "Score a Customer", "Global Explainability"],
    )

    if page == "Portfolio Overview":
        page_overview(df, scored, summary)
    elif page == "Model Performance":
        page_model_performance(summary)
    elif page == "High-Risk Customers":
        page_high_risk(scored)
    elif page == "Score a Customer":
        page_single_prediction(pipeline, df)
    elif page == "Global Explainability":
        page_global_explain()


if __name__ == "__main__":
    main()
