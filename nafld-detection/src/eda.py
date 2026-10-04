"""
eda.py - Exploratory Data Analysis: look at the data before modelling.

Builds on the reference notebook (notebooks/01_reference_eda.ipynb) and saves
every chart to reports/figures/ so the README and Streamlit app can show them.
Differences from the notebook, on purpose:
  * BMI groups follow WHO cut-offs (the notebook called every BMI above 25 "obese").
  * We also compare NAFLD cases vs. controls, not only alive vs. dead.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR
from src.data_loader import load_dataset
from src.preprocessing import bmi_category

FIG_DIR = REPORTS_DIR / "figures"
sns.set_theme(style="whitegrid")


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame = None) -> pd.DataFrame:
    """Create all EDA figures and return a small summary table."""
    if df is None:
        df = load_dataset()
    df = df.copy()
    df["bmi_group"] = df["bmi"].apply(lambda v: bmi_category(v) if pd.notna(v) else None)
    df["sex"] = df["male"].map({0: "Female", 1: "Male"})

    # 1. Missing values --------------------------------------------------------
    missing = (df[["age", "weight", "height", "bmi"]].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    missing.plot(kind="barh", ax=ax, color="#C0392B")
    ax.set_xlabel("% missing (after BMI-formula recovery)")
    ax.set_title("Missing values per feature")
    _save(fig, "01_missing_values.png")

    # 2. Class balance of both targets ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    df["nafld_case"].map({0: "Control", 1: "NAFLD case"}).value_counts().plot(
        kind="bar", ax=axes[0], color=["#7F8C8D", "#C0392B"])
    axes[0].set_title("NAFLD case vs control")
    df["status"].map({0: "Alive", 1: "Dead"}).value_counts().plot(
        kind="bar", ax=axes[1], color=["#2E86AB", "#C0392B"])
    axes[1].set_title("Alive vs dead at follow-up")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("Number of people")
        ax.tick_params(axis="x", rotation=0)
    _save(fig, "02_class_balance.png")

    # 3. Feature distributions split by NAFLD case / control ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["age", "weight", "bmi"]):
        sns.histplot(data=df, x=col, hue=df["nafld_case"].map({0: "Control", 1: "NAFLD case"}),
                     bins=40, stat="density", common_norm=False, element="step", ax=ax)
        ax.set_title(f"{col.capitalize()} - NAFLD case vs control")
    _save(fig, "03_distributions_by_nafld.png")

    # 4. Mortality by age band ------------------------------------------------------
    df["age_band"] = pd.cut(df["age"], [0, 30, 40, 50, 60, 70, 80, 120])
    death_rate = df.groupby("age_band", observed=True)["status"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    death_rate.plot(kind="bar", ax=ax, color="#C0392B")
    ax.set_ylabel("% who died during follow-up")
    ax.set_xlabel("Age band at entry")
    ax.set_title("Mortality rises steeply with age")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "04_mortality_by_age.png")

    # 5. NAFLD share by BMI group -------------------------------------------------------
    share = df.groupby("bmi_group")["nafld_case"].mean().mul(100)
    order = [g for g in ["Underweight", "Healthy", "Overweight", "Obese"] if g in share.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    share[order].plot(kind="bar", ax=ax, color="#C0392B")
    ax.set_ylabel("% NAFLD cases")
    ax.set_xlabel("WHO BMI group")
    ax.set_title("NAFLD share by BMI group")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, "05_nafld_by_bmi_group.png")

    # 6. Correlation heat-map ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr = df[["age", "male", "weight", "height", "bmi", "nafld_case", "status"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, "06_correlation.png")

    summary = df.groupby("nafld_case")[["age", "weight", "height", "bmi", "male", "status"]].mean().round(2)
    summary.index = summary.index.map({0: "Control", 1: "NAFLD case"})
    summary.to_csv(FIG_DIR / "group_summary.csv")
    print("EDA figures saved to", FIG_DIR)
    print(summary)
    return summary
