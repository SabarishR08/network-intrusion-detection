"""Project launcher for the PS40 Network Intrusion Detection repository."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src import __version__


PROJECT_ROOT = Path(__file__).resolve().parent


def run_module(module: str, args: list[str] | None = None) -> int:
    command = [sys.executable, "-m", module]
    if args:
        command.extend(args)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def run_evaluation() -> int:
    exit_code = run_module("src.evaluate_model")
    if exit_code != 0:
        return exit_code
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print("Evaluation Metrics:")
            if isinstance(metrics, dict):
                for key, val in metrics.items():
                    print(f"  {key}: {val}")
            else:
                print(metrics)
        except Exception:
            print(metrics_path.read_text(encoding="utf-8"))
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="PS40 project launcher")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "predict", "app", "package", "evaluate"],
        help="Action to run",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation and print metrics")
    parser.add_argument("--input", help="Input CSV for prediction")
    parser.add_argument("--output", help="Output CSV for prediction")
    args = parser.parse_args()

    if args.evaluate or args.command == "evaluate":
        raise SystemExit(run_evaluation())
    if not args.command:
        parser.error("the following arguments are required: command")
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
