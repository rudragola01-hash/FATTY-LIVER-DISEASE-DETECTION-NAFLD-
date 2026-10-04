"""
data_loader.py - load the nafld1 CSV, check it, and create the target columns.
"""
import numpy as np
import pandas as pd

from src.config import DATA_PATH

REQUIRED_COLUMNS = ["id", "age", "male", "weight", "height", "bmi", "case_id", "futime", "status"]


def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    """Read nafld1.csv and return a clean DataFrame with simple column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'nafld1.csv' from Kaggle and place it in the data/ folder "
            "(see data/README.md)."
        )

    df = pd.read_csv(path)

    # Kaggle's export often has an extra unnamed index column - drop it.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # "case.id" has a dot, which is awkward in Python -> rename to "case_id".
    df = df.rename(columns={"case.id": "case_id"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df


def fill_from_bmi_formula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recover missing weight / height / BMI using  BMI = weight(kg) / height(m)^2.

    If two of the three values are known, the third can be calculated exactly,
    so we do that BEFORE falling back to statistical imputation. This keeps real
    information instead of guessing it.
    """
    df = df.copy()
    height_m = df["height"] / 100.0

    # BMI missing, weight + height known
    mask = df["bmi"].isna() & df["weight"].notna() & df["height"].notna()
    df.loc[mask, "bmi"] = df.loc[mask, "weight"] / (height_m[mask] ** 2)

    # weight missing, BMI + height known
    mask = df["weight"].isna() & df["bmi"].notna() & df["height"].notna()
    df.loc[mask, "weight"] = df.loc[mask, "bmi"] * (height_m[mask] ** 2)

    # height missing, BMI + weight known
    mask = df["height"].isna() & df["bmi"].notna() & df["weight"].notna()
    df.loc[mask, "height"] = np.sqrt(df.loc[mask, "weight"] / df.loc[mask, "bmi"]) * 100.0

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the `nafld_case` column.

    In the nafld1 study every NAFLD case is matched with several controls, and
    `case_id` stores the id of the matched case. For a case, `case_id == id`;
    for a control, `case_id` points to somebody else.
    """
    df = df.copy()
    df["nafld_case"] = (df["id"] == df["case_id"]).astype(int)
    return df


def validate_nafld_case(df: pd.DataFrame) -> None:
    """Sanity-check the derived target so a wrong assumption is caught early."""
    share = df["nafld_case"].mean()
    if not 0.05 <= share <= 0.60:
        raise ValueError(
            f"Only {share:.1%} of rows were flagged as NAFLD cases (id == case_id). "
            "That is not plausible for this dataset - check the CSV, or train on "
            "the 'status' target instead:  python main.py --target status"
        )


def load_dataset() -> pd.DataFrame:
    """Full loading pipeline used by every other module."""
    df = load_raw_data()
    df = fill_from_bmi_formula(df)
    df = add_targets(df)
    validate_nafld_case(df)
    return df
