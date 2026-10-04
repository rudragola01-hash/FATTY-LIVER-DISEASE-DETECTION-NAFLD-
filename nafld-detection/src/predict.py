"""
predict.py - load a saved model and score one person.
Used by the Streamlit app, and handy from a Python shell too.
"""
import joblib
import pandas as pd

from src.config import FEATURES, MODELS_DIR, TARGETS


def available_targets() -> list:
    """Targets that already have a trained model saved on disk."""
    return [t for t in TARGETS if (MODELS_DIR / t / "model.joblib").exists()]


def load_bundle(target: str) -> dict:
    """Load the saved dictionary (model + metadata) for a target."""
    return joblib.load(MODELS_DIR / target / "model.joblib")


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """BMI = weight (kg) / height (m) squared."""
    return weight_kg / (height_cm / 100.0) ** 2


def predict_probability(bundle: dict, age: float, male: int,
                        weight_kg: float, height_cm: float) -> float:
    """Return the model's probability (0-1) for the positive class."""
    row = pd.DataFrame([{
        "age": age,
        "male": male,
        "weight": weight_kg,
        "height": height_cm,
        "bmi": calculate_bmi(weight_kg, height_cm),
    }])[FEATURES]
    return float(bundle["model"].predict_proba(row)[0, 1])
