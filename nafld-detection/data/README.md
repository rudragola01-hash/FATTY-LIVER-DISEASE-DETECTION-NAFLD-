# Dataset

Place the Kaggle file here as **`data/nafld1.csv`**.

- Dataset: *NAFLD (Non-Alcoholic Fatty Liver Disease) - `nafld1`* (Kaggle)
- It is the `nafld1` table from the R `survival` package: a population study where
  each NAFLD case is matched with control subjects.

| Column    | Meaning                                                  |
|-----------|----------------------------------------------------------|
| `id`      | Subject identifier                                       |
| `age`     | Age at entry to the study (years)                        |
| `male`    | 0 = female, 1 = male                                     |
| `weight`  | Weight in kg                                             |
| `height`  | Height in cm                                             |
| `bmi`     | Body-mass index                                          |
| `case.id` | Id of the NAFLD case this subject is matched to          |
| `futime`  | Days from entry to death or last follow-up               |
| `status`  | 0 = alive at last follow-up, 1 = dead                    |

The CSV is git-ignored so the repository stays small. Anyone cloning the repo
downloads the file from Kaggle and drops it in this folder.
