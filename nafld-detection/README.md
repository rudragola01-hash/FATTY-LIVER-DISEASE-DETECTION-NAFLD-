# 🩺 NAFLD Detection using Machine Learning

A beginner-friendly, end-to-end machine-learning project on **Non-Alcoholic Fatty Liver Disease (NAFLD)**:
data cleaning → EDA → five models compared → best model saved → **Streamlit web app**.

> B.Tech Final Year Project · I.T.S Engineering College, Greater Noida (AKTU) · Session 2026-27
> Team: Reema Kumari, Rahul Attri, Rudra Gola, Shobha Kumari · Supervisor: Mr. Tarun Kumar Chugh

⚠️ **Disclaimer:** this is a screening / learning project, **not** a medical diagnosis tool.

---

## 1. What problem does it solve?

Fatty liver often has no symptoms in early stages, while biopsy is invasive and imaging is costly.
This project asks: *can simple, routinely available information flag people who deserve a closer look?*

It supports **two tasks** on the same dataset:

| Task (`--target`) | Question | Positive class |
|---|---|---|
| `nafld_case` | Is this person a NAFLD case or a matched control? (**detection**) | NAFLD case |
| `status` | Did this person die during follow-up? (**mortality risk**, the target used in the reference EDA notebook) | Died |

## 2. Dataset

[`nafld1`](https://www.kaggle.com/) from Kaggle (originally the R `survival` package) - ~17,500 people.
Columns: `id, age, male, weight, height, bmi, case.id, futime, status`.
See [`data/README.md`](data/README.md) for how to download it. **The CSV is not stored in this repo.**

How the targets are built:
- `nafld_case = 1` when `id == case.id` (the person *is* the matched NAFLD case), otherwise `0` (a control).
- `status` is used as given (0 = alive, 1 = dead).

### Honest limitations (please read)
- The dataset has **no liver enzymes (ALT/AST/GGT), diabetes status or lifestyle data**. The synopsis lists these as
  desirable features, but they do not exist in `nafld1`. Only `age, male, weight, height, bmi` are used.
- Controls were **matched to cases by age and sex**, so for NAFLD detection mostly weight/BMI carries signal.
  Expect **moderate** accuracy; the project does not claim the 90 %+ target mentioned in the synopsis.
- `futime` (follow-up time) is **excluded**: it is only known after the outcome (data leakage).
- One population study, no external validation.

## 3. Project structure

```
nafld-detection/
├── README.md              <- you are here
├── main.py                <- run the whole pipeline (EDA + training)
├── app.py                 <- Streamlit web app
├── requirements.txt       <- Python packages
├── data/                  <- put nafld1.csv here (see data/README.md)
├── src/
│   ├── config.py          <- paths, feature list, settings
│   ├── data_loader.py     <- load CSV, recover missing BMI values, build targets
│   ├── preprocessing.py   <- imputation + scaling pipeline, BMI categories
│   ├── models.py          <- 5 classifiers + tuning grids
│   ├── train.py           <- split, tune, evaluate, save best model
│   ├── evaluate.py        <- metrics and plots
│   ├── eda.py             <- exploratory charts
│   └── predict.py         <- load model & score one person
├── notebooks/01_reference_eda.ipynb   <- original EDA notebook
├── models/                <- saved models (created by main.py)
├── reports/               <- charts + comparison tables (created by main.py)
├── tests/                 <- automated tests (use fake data)
└── .streamlit/config.toml <- app theme
```

## 4. How it works (pipeline)

1. **Load & clean** - drop the extra index column; where two of weight/height/BMI are known, calculate the third
   with `BMI = kg / m²`; remaining gaps are filled with the median (inside the training pipeline - no leakage).
2. **EDA** - missing values, class balance, distributions, mortality by age, NAFLD share by WHO BMI group, correlations.
3. **Split** - 80 % train / 20 % test, stratified so class ratios match.
4. **Models** - Logistic Regression, linear SVM (probability-calibrated), Random Forest, Gradient Boosting, KNN.
5. **Tuning** - `GridSearchCV`, 5-fold cross-validation, scored by ROC-AUC.
6. **Evaluation** - Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, permutation feature importance.
7. **Selection** - best ROC-AUC on the unseen test set is saved to `models/<target>/model.joblib`.

> Why ROC-AUC? Both targets are imbalanced, so accuracy alone is misleading
> (predicting "everyone is alive" already scores ~92 % accuracy on the mortality task).

## 5. Run it locally

```bash
git clone https://github.com/<your-username>/nafld-detection.git
cd nafld-detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. put nafld1.csv in the data/ folder
# 2. run the pipeline (a few minutes)
python main.py                   # add --quick for a fast test run
# 3. start the app
streamlit run app.py
```

Useful options: `python main.py --step eda`, `--step train`, `--target nafld_case`, `--target status`, `--quick`.
Run the tests with `pytest -q`.

## 6. Results

After running `python main.py`, open:
- `reports/nafld_case/model_comparison.csv` and `reports/status/model_comparison.csv`
- the figures in `reports/`

Paste your own numbers here so the README reflects your real run:

| Task | Best model | ROC-AUC | F1 |
|---|---|---|---|
| NAFLD detection | _fill in_ | _fill in_ | _fill in_ |
| Mortality risk | _fill in_ | _fill in_ | _fill in_ |

## 7. Deploy on Streamlit Community Cloud (free)

1. Run `python main.py` locally so `models/` and `reports/` are filled.
2. Commit **everything including `models/*/model.joblib` and `reports/`** (the CSV stays ignored):
   ```bash
   git add .
   git commit -m "Add trained models and reports"
   git push
   ```
3. Go to <https://share.streamlit.io>, sign in with GitHub → **New app**.
4. Choose your repo, branch `main`, main file path **`app.py`** → **Deploy**.

The deployed app only needs the saved models, not the dataset.

## 8. Ethics & privacy
The app stores nothing you enter. Real clinical data would require informed consent and anonymisation. Predictions
support, and never replace, professional medical judgment.

## 9. References
1. Rabbi et al. (2020) - Prediction of Liver Disorders using ML Algorithms: A Comparative Study.
2. Ghazal et al. (2022) - Intelligent Model to Predict Early Liver Disease using ML.
3. Wu et al. (2019) - Prediction of fatty liver disease using ML algorithms. *Comput Methods Programs Biomed* 170.
4. Ghosh et al. (2021) - Comparative Analysis of ML Algorithms to Predict Liver Disease.
5. Afreen et al. (2021) - ML Approach Using Boosting Algorithm for Liver Disease Classification.
6. Powell, Wong & Rinella (2021) - Non-alcoholic fatty liver disease. *The Lancet* 397.
7. Younossi et al. (2016) - Global epidemiology of NAFLD. *Hepatology* 64(1).

## 10. License
MIT - see `LICENSE`.
