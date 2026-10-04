"""
preprocessing.py - turn raw feature columns into model-ready numbers.

The steps live inside a scikit-learn Pipeline, so they are learned from the
TRAINING data only and then re-used unchanged on test data and in the app.
That prevents "data leakage" from the test set.
"""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    """
    1. SimpleImputer(median): fill any value still missing with the column median
       (median is robust to outliers, unlike the mean).
    2. StandardScaler: rescale every feature to mean 0 / std 1 so that, for
       example, weight (~85) does not drown out `male` (0 or 1) in SVM / KNN.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def bmi_category(bmi: float) -> str:
    """WHO adult BMI categories (used in EDA and the app)."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy"
    if bmi < 30:
        return "Overweight"
    return "Obese"
