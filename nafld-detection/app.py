"""
app.py - Streamlit web app.   Run locally:  streamlit run app.py

Tabs
  1. Risk prediction   - enter age, sex, weight, height -> model probability
  2. Model performance - comparison table + ROC / confusion matrix / importance
  3. Data insights     - the EDA charts
  4. About             - method, limitations, ethics
"""
import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR, TARGETS
from src.predict import available_targets, calculate_bmi, load_bundle, predict_probability
from src.preprocessing import bmi_category

st.set_page_config(page_title="NAFLD Detection", page_icon="🩺", layout="wide")


@st.cache_resource  # load each model from disk only once
def get_bundle(target: str) -> dict:
    return load_bundle(target)


def show_image(path, caption=None):
    """Show a saved figure if it exists (keeps the app from crashing otherwise)."""
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure not found: {path.name}. Run `python main.py` to create it.")


# ------------------------------------------------------------------ header
st.title("🩺 NAFLD Detection & Risk Screening")
st.caption("Machine-learning screening demo built on the `nafld1` population study. "
           "**Not a medical diagnosis.**")

targets = available_targets()
if not targets:
    st.error("No trained model found. In a terminal run:  `python main.py`  "
             "(see README), then commit the files in `models/` and reload this page.")
    st.stop()

target = st.sidebar.radio(
    "Prediction task", targets, format_func=lambda t: TARGETS[t]["title"])
info = TARGETS[target]
bundle = get_bundle(target)
st.sidebar.markdown(f"**Question answered:** {info['question']}")
st.sidebar.markdown(f"**Best model:** {bundle['model_name']}")
st.sidebar.markdown(f"**Test ROC-AUC:** {bundle['metrics']['ROC-AUC']:.3f}")
st.sidebar.warning("Screening / decision-support only. Always consult a clinician.")

tab_predict, tab_perf, tab_eda, tab_about = st.tabs(
    ["Risk prediction", "Model performance", "Data insights", "About"])

# ------------------------------------------------------------ 1. prediction
with tab_predict:
    st.subheader(info["title"])
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age (years)", 18, 100, 50)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        weight = st.number_input("Weight (kg)", 35.0, 220.0, 75.0, step=0.5)
        height = st.number_input("Height (cm)", 120.0, 220.0, 168.0, step=0.5)

    bmi = calculate_bmi(weight, height)
    with right:
        st.metric("Calculated BMI", f"{bmi:.1f}", bmi_category(bmi))
        st.caption("BMI = weight (kg) / height (m)². WHO groups: <18.5 underweight, "
                   "18.5-24.9 healthy, 25-29.9 overweight, 30+ obese.")

    if st.button("Predict", type="primary"):
        prob = predict_probability(bundle, age, 1 if sex == "Male" else 0, weight, height)
        base = bundle["positive_rate"]
        ratio = prob / base if base else 0

        if ratio < 0.75:
            band = "Lower than the study average"
        elif ratio < 1.5:
            band = "Around the study average"
        else:
            band = "Higher than the study average"

        c1, c2 = st.columns(2)
        c1.metric(f"Probability: {info['positive_label']}", f"{prob:.1%}")
        c2.metric("Study average", f"{base:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.markdown(f"**Result:** {band}  (the model's estimate is {ratio:.1f}x the study average)")

        if target == "nafld_case":
            st.info("The model only sees age, sex, weight, height and BMI, so this is a "
                    "coarse screening signal. Confirming fatty liver needs blood tests, "
                    "ultrasound/elastography and a doctor.")
        else:
            st.info("This estimates the chance of dying during follow-up in this particular "
                    "study cohort. It is not a life-expectancy calculator.")

# --------------------------------------------------------- 2. performance
with tab_perf:
    st.subheader("How well do the models perform on unseen data?")
    csv_path = REPORTS_DIR / target / "model_comparison.csv"
    if csv_path.exists():
        table = pd.read_csv(csv_path, index_col="Model")
        shown = table[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV ROC-AUC"]]
        st.dataframe(shown.style.highlight_max(axis=0, color="#F6D5D1").format("{:.3f}"))
        st.caption("Best model is chosen by ROC-AUC because it does not depend on a "
                   "probability cut-off and is not fooled by class imbalance.")
    else:
        st.info("Run `python main.py` to generate the comparison table.")

    a, b = st.columns(2)
    with a:
        show_image(REPORTS_DIR / target / "roc_curves.png")
    with b:
        show_image(REPORTS_DIR / target / "confusion_matrix.png")
    show_image(REPORTS_DIR / target / "feature_importance.png")

# -------------------------------------------------------------- 3. insights
with tab_eda:
    st.subheader("Exploratory data analysis")
    fig_dir = REPORTS_DIR / "figures"
    names = sorted(p.name for p in fig_dir.glob("*.png")) if fig_dir.exists() else []
    if not names:
        st.info("Run `python main.py --step eda` to create the charts.")
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[i:i + 2]):
            with col:
                show_image(fig_dir / name)

# ----------------------------------------------------------------- 4. about
with tab_about:
    st.markdown(f"""
### What this project does
Compares five classifiers (Logistic Regression, linear SVM, Random Forest,
Gradient Boosting, KNN) on the `nafld1` dataset and serves the best one here.

### Data
Trained on **{bundle['n_train']:,}** people, tested on **{bundle['n_test']:,}** unseen people.

### Honest limitations
- Only **age, sex, weight, height and BMI** are available in this dataset. There are no
  liver enzymes (ALT/AST/GGT), diabetes status or lifestyle fields, so accuracy is modest.
- Controls in the study were **matched to cases by age and sex**, so those two features
  carry little signal for the NAFLD-detection task; BMI/weight do most of the work.
- Results come from one population study and have **not** been validated on other centres.
- This is a learning / screening demo, **not** a medical device.

### Ethics & privacy
Nothing you type is stored. The model supports, and never replaces, professional
clinical judgment.
""")
