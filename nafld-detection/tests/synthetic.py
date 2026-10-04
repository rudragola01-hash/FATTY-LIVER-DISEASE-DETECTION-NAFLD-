"""
synthetic.py - builds a FAKE nafld1-shaped DataFrame for automated tests.

It only exists so the code can be tested without the real Kaggle file.
Never use it for real results or put its numbers in the README.
"""
import numpy as np
import pandas as pd


def make_fake_nafld(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_case = rng.random(n) < 0.25
    age = rng.normal(52, 14, n).clip(18, 95).round()
    male = rng.integers(0, 2, n)
    height = rng.normal(168 + 9 * male, 8, n).clip(140, 205).round()
    bmi = rng.normal(27 + 4 * is_case, 5, n).clip(16, 60)
    weight = (bmi * (height / 100) ** 2).round(1)

    ids = np.arange(1, n + 1)
    case_ids = ids.copy()
    case_pool = ids[is_case]
    case_ids[~is_case] = rng.choice(case_pool, size=(~is_case).sum())

    death_p = 1 / (1 + np.exp(-(age - 75) / 8))
    status = (rng.random(n) < death_p).astype(int)

    df = pd.DataFrame({"id": ids, "age": age, "male": male, "weight": weight,
                       "height": height, "bmi": bmi.round(2), "case.id": case_ids,
                       "futime": rng.integers(100, 8000, n), "status": status})
    # punch random holes, like the real data
    for col in ["weight", "height", "bmi"]:
        df.loc[rng.random(n) < 0.2, col] = np.nan
    return df
