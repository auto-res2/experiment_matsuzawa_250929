"""src/main.py
Command-line entry point.  Delegates to train.py (training) or evaluate.py
(evaluation) depending on arguments.  Ensures the CLI contract required by the
workflow is fully respected.
"""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description="COMMON CORE FOUNDATION entry-point")
    sub = parser.add_subparsers(dest="mode")

    # training modes -----------------------------------------------------------
    train_p = sub.add_parser("train", help="run training experiment")
    train_p.add_argument("--smoke-test", action="store_true", help="quick validation run")
    train_p.add_argument("--full-experiment", action="store_true", help="full benchmark run")
    train_p.add_argument("--variation", type=str, required=False, default="ASHA-baseline")
    train_p.add_argument("--config-path", type=str, default=None)

    # evaluation ---------------------------------------------------------------
    eval_p = sub.add_parser("evaluate", help="evaluate directory of runs")
    eval_p.add_argument("--results-dir", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "train":
        module = importlib.import_module("src.train")
        sys.argv = ["train"]  # clean argv for sub-module parser
        if args.smoke_test:
            sys.argv += ["--smoke-test"]
        if args.full_experiment:
            sys.argv += ["--full-experiment"]
        sys.argv += ["--variation", args.variation]
        if args.config_path:
            sys.argv += ["--config-path", args.config_path]
        module.main()
    elif args.mode == "evaluate":
        module = importlib.import_module("src.evaluate")
        sys.argv = ["evaluate", "--results-dir", args.results_dir]
        module.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
