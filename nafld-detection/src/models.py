"""
models.py - the five classifiers from the project synopsis plus their
hyper-parameter search grids.

Each entry: model name -> (pipeline, parameter grid)
Parameter names use the  "clf__<param>"  style because the classifier sits
inside a Pipeline step called "clf".
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def _make_pipeline(classifier) -> Pipeline:
    """preprocessing + classifier glued into one object."""
    return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])


def get_models(quick: bool = False) -> dict:
    """
    Return {name: (pipeline, param_grid)}.

    quick=True uses tiny grids - handy for tests and a fast first run.
    `class_weight="balanced"` makes the models pay more attention to the rarer
    class, which matters because the data is imbalanced.
    """
    models = {
        "Logistic Regression": (
            _make_pipeline(LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "SVM": (
            # A linear SVM is used because the RBF-kernel SVC needs several
            # minutes on ~17,500 rows. LinearSVC is fast; CalibratedClassifierCV
            # wraps it so it can output probabilities (needed for the app).
            # To try the kernel version, swap in:
            #   SVC(kernel="rbf", probability=True, class_weight="balanced")
            _make_pipeline(CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", max_iter=5000,
                          random_state=RANDOM_STATE), cv=3)),
            {"clf__estimator__C": [0.1, 1.0]},
        ),
        "Random Forest": (
            _make_pipeline(RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 8],
                "clf__min_samples_leaf": [10, 30],
            },
        ),
        "Gradient Boosting": (
            _make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
        "KNN": (
            _make_pipeline(KNeighborsClassifier()),
            {"clf__n_neighbors": [15, 31, 51]},
        ),
    }

    if quick:  # keep only the first value of every grid -> a single fit each
        models = {
            name: (pipe, {k: v[:1] for k, v in grid.items()})
            for name, (pipe, grid) in models.items()
        }
    return models
