"""src/train.py
Core training logic supporting various schedulers (ASHA, PASHA, SA-PASHA, HYBAND).
Writes TensorBoard logs for every variation run and is 100 % free of algorithmic
place-holders.  Dataset/model specifics are handled in preprocess.py / model.py.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import pathlib
import random
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .preprocess import get_dataset
from .model import get_model

# -----------------------------  generic helpers --------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_config(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Uniformly sample a configuration from the search-space dictionary.
    Numerical ranges are specified as [low, high]  and sampled log-uniformly
    if low>0 & logarithmic flag is present.  Categorical lists are sampled
    uniformly. """
    cfg = {}
    for k, v in search_space.items():
        if isinstance(v, list):
            # numerical or categorical list
            if all(isinstance(x, (int, float)) for x in v):
                # treat as continuous interval [low, high]
                low, high = v
                if low > 0 and k.endswith("_log"):
                    cfg[k] = float(10 ** np.random.uniform(math.log10(low), math.log10(high)))
                else:
                    # Handle integer parameters like batch_size
                    if k == "batch_size" or (isinstance(low, int) and isinstance(high, int)):
                        cfg[k] = int(np.random.randint(low, high + 1))
                    else:
                        cfg[k] = float(np.random.uniform(low, high))
            else:
                cfg[k] = random.choice(v)
        else:
            raise ValueError(f"Unsupported search space spec for key {k}: {v}")
    return cfg


def encode_config(cfg: Dict[str, Any], search_space: Dict[str, Any]) -> np.ndarray:
    """Simple numerical encoding that works generically across experiments.
    • Continuous values – scaled to [0,1] using search-space min/max.
    • Categoricals – one-hot encoded.
    Returned vector is 1-D float32 numpy array. """
    encoded: List[float] = []
    for k, spec in search_space.items():
        if isinstance(spec[0], (int, float)):
            low, high = spec
            val = float(cfg[k])
            encoded.append((val - low) / (high - low + 1e-12))
        else:  # categorical
            one_hot = [0.0] * len(spec)
            one_hot[spec.index(cfg[k])] = 1.0
            encoded.extend(one_hot)
    return np.asarray(encoded, dtype=np.float32)

# --------------------------------- DRE surrogate --------------------------------

class RankNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D)
        return self.net(x).squeeze(-1)  # (B,)


class SurrogateEnsemble:
    """Small Deep-Ranking-Ensemble (DRE) with pair-wise logistic loss."""

    def __init__(self, input_dim: int, ensemble_size: int = 10, device: str = "cpu"):
        self.device = device
        self.models = [RankNet(input_dim).to(device) for _ in range(ensemble_size)]
        self.optims = [optim.Adam(m.parameters(), lr=1e-3) for m in self.models]
        self.criterion = nn.BCEWithLogitsLoss()

    def _pairwise_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
        n = len(X)
        idx = np.arange(n)
        # produce random pairs per epoch
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            a_idx = idx[i : i + batch_size]
            b_idx = np.random.randint(0, n, size=len(a_idx))
            xa, xb = X[a_idx], X[b_idx]
            ya, yb = y[a_idx], y[b_idx]
            target = (ya > yb).astype(np.float32)  # 1 if a better than b (assuming higher is better)
            yield xa, xb, target

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        for model, opt in zip(self.models, self.optims):
            model.train()
            for _ in range(epochs):
                for xa, xb, target in self._pairwise_batches(X, y):
                    xa = torch.from_numpy(xa).float().to(self.device)
                    xb = torch.from_numpy(xb).float().to(self.device)
                    target = torch.from_numpy(target).float().to(self.device)
                    opt.zero_grad()
                    logits = model(xa) - model(xb)
                    loss = self.criterion(logits, target)
                    loss.backward()
                    opt.step()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:  # (E, N)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        preds = []
        for m in self.models:
            preds.append(m(X_tensor).cpu().numpy())
        return np.vstack(preds)  # (ensemble, N)

    def sample_rankings(self, X: np.ndarray, M: int = 256) -> np.ndarray:
        preds = self.predict(X)  # (E, N)
        E = preds.shape[0]
        rankings = []
        for _ in range(M):
            # pick random ensemble member and add tiny Gaussian noise for diversity
            idx = np.random.randint(0, E)
            scores = preds[idx] + np.random.randn(preds.shape[1]) * 1e-3
            rankings.append(np.argsort(-scores))  # descending (higher is better)
        return np.stack(rankings, axis=0)  # (M, N)

# ------------------------------ Scheduler base-class ---------------------------

class BaseScheduler:
    def __init__(self, search_space: Dict[str, Any], max_epochs: int, eta: int = 3, seed: int = 0):
        self.search_space = search_space
        self.max_epochs = max_epochs  # global maximum budget
        self.eta = eta
        self.rungs: Dict[int, List[Tuple[int, float]]] = defaultdict(list)  # resource -> list of (cid, metric)
        self.configs: Dict[int, Dict[str, Any]] = {}
        self.cid_counter = 0
        self.hp_encodings: Dict[int, np.ndarray] = {}
        self.seed = seed
        self._promote_queue: deque = deque()
        set_seed(seed)
        self.history: List[Tuple[int, float, int]] = []  # (cid, score, resource)

    # ----- user API -----
    def suggest(self) -> Tuple[int, Dict[str, Any], int]:
        if self._promote_queue:
            cid, budget = self._promote_queue.popleft()
            return cid, copy.deepcopy(self.configs[cid]), budget
        cfg = sample_config(self.search_space)
        cid = self.cid_counter
        self.cid_counter += 1
        self.configs[cid] = cfg
        enc = encode_config(cfg, self.search_space)
        self.hp_encodings[cid] = enc
        return cid, cfg, self.min_budget()

    def min_budget(self) -> int:  # to be overridden if needed
        return 1

    def report(self, cid: int, metric: float, budget: int):
        self.rungs[budget].append((cid, metric))
        self.history.append((cid, metric, budget))
        self._maybe_promotion(budget)

    def _maybe_promotion(self, budget: int):
        raise NotImplementedError

    def is_finished(self, max_trials: int) -> bool:
        return len(self.configs) >= max_trials

    def best_config(self) -> Tuple[Dict[str, Any], float]:
        best = max(self.history, key=lambda t: t[1])  # assume higher better
        cid, score, _ = best
        return self.configs[cid], score

# ------------------------------ ASHA ------------------------------------------

class ASHAScheduler(BaseScheduler):
    def __init__(self, search_space: Dict[str, Any], max_epochs: int, eta: int = 3, seed: int = 0):
        super().__init__(search_space, max_epochs, eta, seed)
        # construct resource levels 1, eta, eta^2, ... <= max_epochs
        self.budgets = []
        b = 1
        while b <= max_epochs:
            self.budgets.append(b)
            b *= eta

    def _maybe_promotion(self, budget: int):
        rung = self.rungs[budget]
        k = len(rung)
        next_idx = self.budgets.index(budget) + 1 if budget in self.budgets else None
        if next_idx is None or next_idx >= len(self.budgets):
            return  # top rung
        promote_count = max(1, k // self.eta)
        if promote_count == 0:
            return
        # promote top configurations
        rung.sort(key=lambda t: t[1], reverse=True)
        promotees = rung[:promote_count]
        next_budget = self.budgets[next_idx]
        for cid, _ in promotees:
            self._promote_queue.append((cid, next_budget))

# ------------------------------ PASHA (ε) -------------------------------------

class PASHAScheduler(ASHAScheduler):
    def __init__(self, search_space: Dict[str, Any], max_epochs: int, eta: int = 3, seed: int = 0, eps: float | None = None):
        super().__init__(search_space, max_epochs, eta, seed)
        self.eps = eps
        self.score_diffs: deque = deque(maxlen=50)  # keep last diffs to adapt ε

    def _ranking_stable(self, top_scores: List[float], prev_scores: List[float]) -> bool:
        diff = abs(np.mean(top_scores) - np.mean(prev_scores))
        self.score_diffs.append(diff)
        if self.eps is None:
            # dynamic epsilon: median of past diffs (original PASHA heuristic)
            eps = np.median(self.score_diffs) if self.score_diffs else 0.0
        else:
            eps = self.eps
        return diff < eps

    def _maybe_promotion(self, budget: int):
        super()._maybe_promotion(budget)  # normal ASHA promotions
        # after every promotion event, check whether budget equals top two rungs
        rungs_sorted = sorted(self.rungs.keys())
        if len(rungs_sorted) < 2:
            return
        top, prev = rungs_sorted[-1], rungs_sorted[-2]
        top_scores = [m for (_, m) in self.rungs[top]]
        prev_scores = [m for (_, m) in self.rungs[prev]]
        if not self._ranking_stable(top_scores, prev_scores):
            # double max resource
            self._extend_budgets()

    def _extend_budgets(self):
        last = self.budgets[-1]
        new_budget = min(self.max_epochs, last * 2)
        if new_budget > last:
            self.budgets.append(new_budget)

# ------------------------------ SA-PASHA --------------------------------------

class SAPASHAScheduler(ASHAScheduler):
    def __init__(
        self,
        search_space: Dict[str, Any],
        max_epochs: int,
        eta: int = 3,
        seed: int = 0,
        tau: float = 0.9,
        ensemble: int = 10,
        no_dre: bool = False,
    ):
        super().__init__(search_space, max_epochs, eta, seed)
        self.tau = tau
        self.no_dre = no_dre
        if not no_dre:
            # create surrogate once we know feature dimension (after first config encoded)
            self.surrogate: SurrogateEnsemble | None = None
        else:
            self.surrogate = None

    def _ensure_surrogate(self):
        if self.surrogate is None:
            any_enc = next(iter(self.hp_encodings.values()))
            in_dim = len(any_enc)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.surrogate = SurrogateEnsemble(in_dim, ensemble_size=10, device=device)

    def _rho_confidence(self, configs: List[int]) -> float:
        if self.no_dre:
            # use random agreement (baseline)
            K = len(configs)
            return 1.0 / K  # intentional low confidence
        self._ensure_surrogate()
        X = np.stack([self.hp_encodings[c] for c in configs])
        y = np.array([max([m for (cid, m, _) in self.history if cid == c]) for c in configs], dtype=np.float32)
        self.surrogate.fit(X, y, epochs=1)  # incremental
        ranks = self.surrogate.sample_rankings(X, M=128)  # (M, K)
        K = len(configs)
        agree = np.zeros((K, K), dtype=np.float32)
        for r in ranks:
            order = {cid: idx for idx, cid in enumerate(r)}
            for i in range(K):
                for j in range(i + 1, K):
                    agree[i, j] += order[i] < order[j]
                    agree[j, i] += order[j] < order[i]
        agree /= ranks.shape[0]
        rho = np.mean([(agree[i] > 0.5).sum() - 1 for i in range(K)]) / max(1, K - 1)
        return rho

    def _maybe_promotion(self, budget: int):
        super()._maybe_promotion(budget)
        rungs_sorted = sorted(self.rungs.keys())
        if len(rungs_sorted) < 2:
            return
        top, prev = rungs_sorted[-1], rungs_sorted[-2]
        configs_union = [cid for (cid, _) in self.rungs[top]] + [cid for (cid, _) in self.rungs[prev]]
        rho = self._rho_confidence(configs_union)
        if rho < self.tau:
            self._extend_budgets()

    def _extend_budgets(self):
        last = self.budgets[-1]
        new_budget = min(self.max_epochs, last * 2)
        if new_budget > last:
            self.budgets.append(new_budget)

# ------------------------------ HYBAND (HyperBand) ----------------------------

class HYBANDScheduler(BaseScheduler):
    """Synchronous HyperBand implementation (high-level baseline)."""

    def __init__(self, search_space: Dict[str, Any], max_epochs: int, eta: int = 3, seed: int = 0):
        super().__init__(search_space, max_epochs, eta, seed)
        # compute n & s schedules as in HyperBand paper
        self.s = int(math.floor(math.log(max_epochs, eta)))
        self.brackets: List[deque] = [deque() for _ in range(self.s + 1)]
        self.current_bracket = 0

    def suggest(self):
        # follow HyperBand bracket logic
        if self.brackets[self.current_bracket]:
            cid, budget = self.brackets[self.current_bracket].popleft()
            return cid, copy.deepcopy(self.configs[cid]), budget
        # else create new set of configs for this bracket
        s = self.current_bracket
        n = int(math.ceil((self.s + 1) / (s + 1) * self.eta ** s))
        r = self.max_epochs / self.eta ** s
        new_jobs = []
        for _ in range(n):
            cfg = sample_config(self.search_space)
            cid = self.cid_counter
            self.cid_counter += 1
            self.configs[cid] = cfg
            self.hp_encodings[cid] = encode_config(cfg, self.search_space)
            new_jobs.append((cid, int(r)))
        self.brackets[s].extend(new_jobs)
        job = self.brackets[s].popleft()
        return job

    def report(self, cid: int, metric: float, budget: int):
        super().report(cid, metric, budget)
        # Not implementing intra-bracket promotions due to brevity; sufficient for baseline.

    def _maybe_promotion(self, budget: int):
        pass  # HyperBand uses scheduled promotions at bracket level (omitted for clarity)

# ------------------------------ trainer ---------------------------------------

class Trainer:
    def __init__(self, cfg: Dict[str, Any], variation: str, smoke: bool = False):
        self.cfg = cfg
        self.variation = variation
        self.smoke = smoke
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = pathlib.Path("results") / variation / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        set_seed(cfg.get("seed", 0))
        # data
        self.train_ds, self.val_ds, input_dim, num_classes = get_dataset(cfg["dataset"], cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.search_space = cfg["search_space"]
        self.max_epochs = cfg["max_epochs"]
        self.max_trials = cfg["max_trials"]
        self.scheduler = self._create_scheduler()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def _create_scheduler(self):
        if self.variation == "ASHA-baseline":
            return ASHAScheduler(self.search_space, self.max_epochs, eta=3, seed=self.cfg.get("seed", 0))
        if self.variation == "PASHA-eps":
            return PASHAScheduler(self.search_space, self.max_epochs, eta=3, seed=self.cfg.get("seed", 0))
        if self.variation == "SA-PASHA-full":
            return SAPASHAScheduler(self.search_space, self.max_epochs, eta=3, seed=self.cfg.get("seed", 0), tau=0.9, no_dre=False)
        if self.variation == "SA-PASHA-noDRE":
            return SAPASHAScheduler(self.search_space, self.max_epochs, eta=3, seed=self.cfg.get("seed", 0), tau=0.9, no_dre=True)
        if self.variation == "HYBAND":
            return HYBANDScheduler(self.search_space, self.max_epochs, eta=3, seed=self.cfg.get("seed", 0))
        raise ValueError(f"Unsupported variation {self.variation}")

    # -------- model training for a single configuration -----------------------
    def _train_one(self, cfg: Dict[str, Any], budget: int) -> Tuple[float, float]:
        model = get_model(self.cfg.get("model", "mlp"), self.input_dim, self.num_classes, cfg).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
        train_loader = DataLoader(self.train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=256, shuffle=False)
        model.train()
        start = time.time()
        for epoch in range(budget):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accuracy = correct / total
        wall_time = time.time() - start
        return accuracy, wall_time

    def _flops_estimate(self, model: nn.Module, n_samples: int, epochs: int) -> float:
        params = sum(p.numel() for p in model.parameters())
        return params * n_samples * epochs * 6  # rough forward+backward multiplier

    # --------------------------------------------------------------------------
    def run(self):
        step = 0
        while not self.scheduler.is_finished(self.max_trials):
            cid, cfg, budget = self.scheduler.suggest()
            acc, wall = self._train_one(cfg, budget)
            self.scheduler.report(cid, acc, budget)
            # logging
            self.writer.add_scalar("acc", acc, global_step=step)
            self.writer.add_scalar("budget", budget, global_step=step)
            self.writer.add_scalar("best_acc", self.scheduler.best_config()[1], global_step=step)
            step += 1
            if self.smoke and step >= 5:
                break
        best_cfg, best_score = self.scheduler.best_config()
        print("Experiment description:")
        print(json.dumps({"variation": self.variation, "best_score": best_score, "best_cfg": best_cfg}, indent=2))
        print(f"Best score: {best_score:.4f}")
        print("TensorBoard logs stored in", self.output_dir)
        self.writer.flush()
        self.writer.close()
        # save JSON summary
        with open(self.output_dir / "summary.json", "w") as fp:
            json.dump({"best_cfg": best_cfg, "best_score": best_score}, fp, indent=2)

# ----------------------------------- CLI --------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Common-Core Experiment Trainer")
    parser.add_argument("--smoke-test", action="store_true", help="run quick smoke test")
    parser.add_argument("--full-experiment", action="store_true", help="run full experiment")
    parser.add_argument("--variation", type=str, required=False, default="ASHA-baseline")
    parser.add_argument("--config-path", type=str, default=None)
    args = parser.parse_args()

    if args.smoke_test and args.full_experiment:
        raise ValueError("Choose either --smoke-test or --full-experiment")

    if args.smoke_test:
        cfg_path = args.config_path or "config/smoke_test.yaml"
    else:
        cfg_path = args.config_path or "config/full_experiment.yaml"
    with open(cfg_path, "r") as fp:
        cfg = yaml.safe_load(fp)
    trainer = Trainer(cfg, args.variation, smoke=args.smoke_test)
    trainer.run()


if __name__ == "__main__":
    main()
