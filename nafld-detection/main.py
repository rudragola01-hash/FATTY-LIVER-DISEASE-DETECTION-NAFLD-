"""
main.py - the single entry point of the project.

    python main.py                       # EDA + train both tasks
    python main.py --step eda            # only make the EDA charts
    python main.py --step train          # only train the models
    python main.py --target nafld_case   # one task only
    python main.py --quick               # tiny search grids (fast smoke test)

Afterwards launch the app with:   streamlit run app.py
"""
import argparse

from src.config import TARGETS
from src.eda import run_eda
from src.train import train_target
from src.data_loader import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="NAFLD detection pipeline")
    parser.add_argument("--step", choices=["eda", "train", "all"], default="all")
    parser.add_argument("--target", choices=[*TARGETS, "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="use small search grids")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset()  # load once, share between steps

    if args.step in ("eda", "all"):
        run_eda(df)

    if args.step in ("train", "all"):
        targets = list(TARGETS) if args.target == "both" else [args.target]
        for target in targets:
            train_target(target, quick=args.quick, df=df)


if __name__ == "__main__":
    main()
