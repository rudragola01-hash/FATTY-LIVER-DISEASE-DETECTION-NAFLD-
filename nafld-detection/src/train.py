"""
train.py - train, tune, compare and save the models for ONE target.

Steps (same order as the synopsis methodology):
  load data -> split -> tune each model -> evaluate on unseen test data
  -> pick the best -> save model, metrics table and figures.
"""
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import evaluate
from src.config import (CV_FOLDS, FEATURES, MODEL_SELECTION_METRIC, MODELS_DIR,
                        RANDOM_STATE, REPORTS_DIR, TARGETS, TEST_SIZE)
from src.data_loader import load_dataset
from src.models import get_models


def train_target(target: str, quick: bool = False, df: pd.DataFrame = None) -> dict:
    """Train all models for `target` ('nafld_case' or 'status'). Returns a summary dict."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {list(TARGETS)}")
    info = TARGETS[target]

    # 1. Data ------------------------------------------------------------------
    if df is None:
        df = load_dataset()
    X, y = df[FEATURES], df[target]
    print(f"\n=== {info['title']} ===")
    print(f"Rows: {len(df):,} | positives: {y.mean():.1%}")

    # stratify=y keeps the class ratio identical in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # 2. Tune + 3. Evaluate each model -----------------------------------------
    cv = StratifiedKFold(n_splits=3 if quick else CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    fitted, rows = {}, []
    for name, (pipeline, grid) in get_models(quick).items():
        start = time.time()
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best

        metrics = evaluate.compute_metrics(
            y_test, best.predict(X_test), best.predict_proba(X_test)[:, 1])
        metrics.update({"Model": name,
                        "CV ROC-AUC": search.best_score_,
                        "Best params": json.dumps(
                            {k.replace("clf__", ""): v for k, v in search.best_params_.items()})})
        rows.append(metrics)
        print(f"  {name:<20} AUC={metrics['ROC-AUC']:.3f}  "
              f"F1={metrics['F1']:.3f}  ({time.time() - start:.0f}s)")

    # 4. Compare and pick the best ------------------------------------------------
    results = (pd.DataFrame(rows).set_index("Model")
               .sort_values("ROC-AUC", ascending=False))
    best_name = results.index[0]
    best_model = fitted[best_name]
    print(f"Best model: {best_name}")

    # 5. Save everything -----------------------------------------------------------
    model_dir = MODELS_DIR / target
    report_dir = REPORTS_DIR / target
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results.round(4).to_csv(report_dir / "model_comparison.csv")
    evaluate.plot_roc_curves(fitted, X_test, y_test, report_dir / "roc_curves.png")
    evaluate.plot_confusion_matrix(
        best_model, X_test, y_test,
        [info["negative_label"], info["positive_label"]],
        f"Confusion matrix - {best_name}", report_dir / "confusion_matrix.png")
    importance = evaluate.plot_feature_importance(
        best_model, X_test, y_test, FEATURES,
        f"Feature importance - {best_name}", report_dir / "feature_importance.png")
    importance.round(4).to_csv(report_dir / "feature_importance.csv", index=False)

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "target": target,
            "features": FEATURES,
            "metrics": results.loc[best_name, ["Accuracy", "Precision", "Recall",
                                               "F1", "ROC-AUC"]].astype(float).to_dict(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "positive_rate": float(y.mean()),
        },
        model_dir / "model.joblib", compress=3)

    return {"target": target, "best_model": best_name, "results": results}
