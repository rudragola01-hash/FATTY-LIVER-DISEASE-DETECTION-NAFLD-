"""
Run with:  pytest -q
Uses fake data, so no Kaggle download is needed.
"""
import numpy as np

from src import data_loader
from src.predict import calculate_bmi
from tests.synthetic import make_fake_nafld


def _prepare(df):
    df = df.rename(columns={"case.id": "case_id"})
    return data_loader.add_targets(data_loader.fill_from_bmi_formula(df))


def test_bmi_formula_recovers_missing_values():
    df = make_fake_nafld()
    before = df[["weight", "height", "bmi"]].isna().sum().sum()
    after = _prepare(df)[["weight", "height", "bmi"]].isna().sum().sum()
    assert after < before


def test_bmi_formula_is_correct():
    assert round(calculate_bmi(70, 175), 1) == 22.9


def test_nafld_case_target_is_binary_and_plausible():
    df = _prepare(make_fake_nafld())
    assert set(df["nafld_case"].unique()) == {0, 1}
    data_loader.validate_nafld_case(df)  # must not raise


def test_quick_training_runs_and_beats_chance(tmp_path, monkeypatch):
    from src import train
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path / "reports")
    out = train.train_target("nafld_case", quick=True, df=_prepare(make_fake_nafld()))
    assert out["results"]["ROC-AUC"].max() > 0.6
    assert (tmp_path / "models" / "nafld_case" / "model.joblib").exists()
