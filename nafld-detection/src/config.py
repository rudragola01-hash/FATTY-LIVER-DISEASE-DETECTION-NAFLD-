"""
config.py - one place for every setting used in the project.

Change a value here and the whole pipeline (EDA, training, Streamlit app)
picks it up. Nothing else in the project hard-codes paths or column names.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "nafld1.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# ----------------------------------------------------------------------------
# Input features the model is allowed to see
# ----------------------------------------------------------------------------
# NOTE: `futime` (follow-up time) is deliberately NOT here. It is only known
# AFTER the outcome happens, so using it would be "data leakage" and would give
# unrealistically good scores.
FEATURES = ["age", "male", "weight", "height", "bmi"]

# ----------------------------------------------------------------------------
# The two prediction tasks this project supports
# ----------------------------------------------------------------------------
TARGETS = {
    "nafld_case": {
        "title": "NAFLD detection",
        "question": "Is this person likely to be a NAFLD case (vs. a matched control)?",
        "positive_label": "NAFLD case",
        "negative_label": "Control",
    },
    "status": {
        "title": "Mortality risk in NAFLD cohort",
        "question": "Did the person die during follow-up?",
        "positive_label": "Died during follow-up",
        "negative_label": "Alive at last follow-up",
    },
}

# ----------------------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------------------
RANDOM_STATE = 42     # makes results reproducible
TEST_SIZE = 0.20      # 20 % of rows are kept hidden for the final exam
CV_FOLDS = 5          # cross-validation folds used while tuning
MODEL_SELECTION_METRIC = "roc_auc"   # best model = highest ROC-AUC on test set
