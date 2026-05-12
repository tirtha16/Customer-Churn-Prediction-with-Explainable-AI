from __future__ import annotations

import argparse
import sys

from src.data_loader import load_data
from src.evaluate import run as run_eval
from src.explain import run_global
from src.train import train_all


def cmd_data() -> None:
    df = load_data()
    print(f"Dataset ready: {len(df):,} rows.")


def cmd_train() -> None:
    summary = train_all()
    print(f"Best model: {summary['best_model']}")


def cmd_evaluate() -> None:
    run_eval()


def cmd_explain() -> None:
    run_global()


def cmd_all() -> None:
    cmd_data()
    cmd_train()
    cmd_evaluate()
    cmd_explain()


COMMANDS = {
    "data": cmd_data,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "explain": cmd_explain,
    "all": cmd_all,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args()
    COMMANDS[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
