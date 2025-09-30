"""src/evaluate.py
Unified evaluation & comparison script.  It reads TensorBoard event files from
all variations in a directory and produces publication-quality .pdf figures that
compare best accuracy over wall-clock, accuracy vs. consumed epochs as well as
final accuracy bar-plots.  All metrics are identical across experimental
variations to guarantee consistency.
"""
from __future__ import annotations
import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams.update({"pdf.fonttype": 42})  # publication friendly


class RunLog:
    def __init__(self, tb_dir: pathlib.Path):
        self.tb_dir = tb_dir
        self.steps: List[int] = []
        self.acc: List[float] = []
        self.best_acc: List[float] = []
        self._load()
        self.final_best = self.best_acc[-1] if self.best_acc else 0.0

    def _load(self):
        ea = event_accumulator.EventAccumulator(str(self.tb_dir))
        ea.Reload()
        if "acc" not in ea.scalars.Keys():
            return
        for scalar in ea.scalars.Items("acc"):
            self.steps.append(scalar.step)
            self.acc.append(scalar.value)
        if "best_acc" in ea.scalars.Keys():
            self.best_acc = [s.value for s in ea.scalars.Items("best_acc")]


class Evaluator:
    def __init__(self, results_dir: pathlib.Path):
        self.results_dir = results_dir
        self.runs: Dict[str, List[RunLog]] = defaultdict(list)
        self._discover()

    def _discover(self):
        for variation_dir in self.results_dir.iterdir():
            if not variation_dir.is_dir():
                continue
            # expect multiple timestamped sub-dirs
            subdirs = [d for d in variation_dir.iterdir() if (d / "tb").exists()]
            for sd in subdirs:
                tb_path = sd / "tb"
                self.runs[variation_dir.name].append(RunLog(tb_path))

    def _aggregate(self):
        summary = {}
        for var, logs in self.runs.items():
            best_scores = [run.final_best for run in logs]
            summary[var] = {
                "mean": float(np.mean(best_scores)),
                "std": float(np.std(best_scores)),
                "runs": best_scores,
            }
        return summary

    # ------------------------- plotting helpers ------------------------------
    def _plot_final_bar(self, summary: Dict[str, Dict[str, float]]):
        fig, ax = plt.subplots(figsize=(6, 4))
        vars_ = list(summary.keys())
        means = [summary[v]["mean"] for v in vars_]
        stds = [summary[v]["std"] for v in vars_]
        ax.bar(vars_, means, yerr=stds, color=sns.color_palette("Set2", len(vars_)))
        for idx, m in enumerate(means):
            ax.text(idx, m + 0.005, f"{m:.3f}", ha="center", va="bottom")
        ax.set_ylabel("Final Best Accuracy")
        ax.set_title("Scheduler Comparison")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        out_path = self.results_dir / "final_accuracy.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print("Saved", out_path)

    def _plot_learning_curves(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        for var, logs in self.runs.items():
            # average across runs (interpolate to common steps)
            if not logs[0].steps:
                continue
            max_step = max(max(r.steps) for r in logs)
            common = np.linspace(0, max_step, 200)
            curves = []
            for run in logs:
                if not run.steps:
                    continue
                curves.append(np.interp(common, run.steps, run.best_acc))
            mean = np.mean(curves, axis=0)
            ax.plot(common, mean, label=var)
            ax.annotate(f"{mean[-1]:.3f}", (common[-1], mean[-1]))
        ax.set_xlabel("Scheduler step")
        ax.set_ylabel("Best Accuracy")
        ax.set_title("Best Accuracy vs Scheduler Step")
        ax.legend()
        fig.tight_layout()
        out_path = self.results_dir / "accuracy_vs_steps.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print("Saved", out_path)

    def run(self):
        summary = self._aggregate()
        # print numerical data
        print("Experiment description:")
        print(json.dumps(summary, indent=2))
        self._plot_final_bar(summary)
        self._plot_learning_curves()


# -------------------------------- CLI ----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate & compare results")
    parser.add_argument("--results-dir", required=True, type=str)
    args = parser.parse_args()
    evaluator = Evaluator(pathlib.Path(args.results_dir))
    evaluator.run()


if __name__ == "__main__":
    main()
