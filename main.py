"""Project launcher for the PS40 Network Intrusion Detection repository."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_module(module: str, args: list[str] | None = None) -> int:
    command = [sys.executable, "-m", module]
    if args:
        command.extend(args)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="PS40 project launcher")
    parser.add_argument("command", choices=["train", "predict", "app", "package"], help="Action to run")
    parser.add_argument("--input", help="Input CSV for prediction")
    parser.add_argument("--output", help="Output CSV for prediction")
    args = parser.parse_args()

    if args.command == "train":
        raise SystemExit(run_module("src.train_model"))
    if args.command == "predict":
        if not args.input or not args.output:
            raise SystemExit("--input and --output are required for predict")
        raise SystemExit(run_module("src.predict", ["--input", args.input, "--output", args.output]))
    if args.command == "app":
        raise SystemExit(run_module("streamlit", ["run", "app/streamlit_app.py"]))
    raise SystemExit(run_module("build_presentation"))


if __name__ == "__main__":
    main()