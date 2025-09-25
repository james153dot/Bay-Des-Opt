#!/usr/bin/env python3
"""
Bayesian Design Optimizer for stress/fatigue/thermal‑constrained problems.

Features
- Gaussian Process surrogate (scikit‑learn if available; NumPy fallback).
- Acquisition: EI, UCB, and **Constrained EI** (feasibility × EI) with
  separate GPs per constraint.
- Batch suggestions (qEI via greedy fantasization) and parallel evaluation.
- Reproducible seeding, checkpointing to JSON, resume.

Usage examples (see bottom of file):
  python bayes_optimize.py --dim 4 --parallel 8 --iters 60 \
      --acq cEI --objective min_mass --seed 42

Design API
- Implement `evaluate_design(x)` to call your solver/rig and return a dict:
    {"objective": float, "stress_max": float, "life": float, "temp_max": float}
- Define limits in `LIMITS` and objective in `objective_value(metrics)`.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple
import numpy as np

# Optional sklearn GP; fallback to minimal GP if unavailable
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

rng = np.random.default_rng(0)

# -------------------------- Problem setup ---------------------------
# Bounds for design variables (edit to your design space)
# Example: [thickness, rib_spacing, alloy_code, hole_diam] normalized to [0,1]
BOUNDS = np.array([
    [0.0, 1.0],  # x0
    [0.0, 1.0],  # x1
    [0.0, 1.0],  # x2
    [0.0, 1.0],  # x3
])

# Limits/specs (edit per program requirements)
LIMITS = {
    "stress_max": 300e6,  # Pa
    "temp_max":  120.0,   # °C
    "life_min":  1.0e6,   # cycles
}

# Objective: example = mass proxy (lower is better)
# Replace with your specific objective mapping from metrics

def objective_value(metrics: Dict[str, float]) -> float:
    # Example: penalize stress/temp violations softly (BO still uses hard constraints via feasibility models)
    mass = metrics.get("mass", 1.0)
    s = metrics.get("stress_max", 0.0)/LIMITS["stress_max"]
    t = metrics.get("temp_max", 0.0)/LIMITS["temp_max"]
    life = metrics.get("life", 1.0)/LIMITS["life_min"]
    penalty = 0.0
    penalty += 10.0*max(0.0, s-1.0)**2
    penalty += 10.0*max(0.0, t-1.0)**2
    penalty += 10.0*max(0.0, 1.0-life)**2
    return mass * (1.0 + penalty)

# --------------------- Mock evaluation (replace) --------------------
# Replace this with your solver hookup. Keep signature identical.
# It should be thread/process‑safe.

def evaluate_design(x: np.ndarray) -> Dict[str, float]:
    """Toy black‑box: multi‑modal, noisy physics‑like metrics for demo.
    Replace with calls to FEA/rig and real postprocessing.
    """
    # Denormalize if needed; here x is in [0,1]
    # Mass proxy: convex in thickness x0 and hole size x3
    mass = 5.0 + 3.0*x[0] + 2.0*(1.0 - x[3]) + 0.1*np.sum(x)
    # Stress rises with holes and spacing, drops with thickness; add curvature
    stress = 120e6 + 250e6*(0.6*x[1] + 0.7*x[3]) - 220e6*x[0] + 60e6*(x[2]-0.5)**2
    # Fatigue life increases with thickness and decreases with stress and temp
    life = 5e5 + 2.5e6*x[0] - 3e5*stress/300e6 - 5e3*(x[2]*100.0) + 2e4*np.sin(6*np.pi*x[1])
    # Temperature rises with x2 (material variant) and low x1 spacing
    temp = 50.0 + 90.0*(0.7*x[2] + 0.3*(1.0-x[1])) + 10.0*np.sin(4*np.pi*x[3])
    # Small heteroscedastic noise
    noise = rng.normal(0.0, 0.01)
    return {
        "objective": objective_value({"mass": mass, "stress_max": stress, "temp_max": temp, "life": life}),
        "mass": mass,
        "stress_max": stress,
        "temp_max": temp,
        "life": life,
        "noise": noise,
    }

# ---------------------------- Surrogates ----------------------------

class GPWrapper:
    def __init__(self, noise: float = 1e-6):
        self.noise = noise
        self._use_sk = _HAVE_SK
        if self._use_sk:
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(BOUNDS.shape[0]),
                                                  length_scale_bounds=(1e-2, 10.0),
                                                  nu=2.5) + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-9, 1e-3))
            self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
        else:
            self.X = None; self.y = None; self.L = None; self.alpha = None; self.l = 0.2; self.sigma2 = 1.0

    def _k(self, A, B):
        # Squared‑exp kernel for fallback
        d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
        return self.sigma2 * np.exp(-0.5 * d2 / (self.l**2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self._use_sk:
            self.gp.fit(X, y)
        else:
            self.X = np.array(X); self.y = np.array(y)
            K = self._k(self.X, self.X) + self.noise*np.eye(len(self.X))
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._use_sk:
            mu, std = self.gp.predict(X, return_std=True)
            return mu, std
        else:
            Kxs = self._k(X, self.X)
            v = np.linalg.solve(self.L, Kxs.T)
            mu = (v.T @ self.alpha)
            kxx = self._k(X, X)
            var = np.maximum(1e-12, np.diag(kxx) - np.sum(v*v, axis=0))
            return mu, np.sqrt(var)

# ----------------------- Acquisition functions ----------------------

def expected_improvement(mu, sigma, y_best, xi=0.0):
    from numpy import sqrt, maximum
    from scipy.stats import norm
    sigma = np.maximum(1e-12, sigma)
    imp = y_best - mu - xi  # minimization
    Z = imp / sigma
    return imp*norm.cdf(Z) + sigma*norm.pdf(Z)

def upper_confidence_bound(mu, sigma, beta=2.0):
    # minimization UCB
    return -(mu - beta*sigma)  # larger is better for acquisition maximization

# Probability of feasibility for constraints: c(x) <= 0

def prob_feasible(mu, sigma):
    from scipy.stats import norm
    sigma = np.maximum(1e-12, sigma)
    return norm.cdf(-mu / sigma)

# -------------------------- BO orchestration ------------------------

@dataclass
class BOState:
    X: np.ndarray
    y_obj: np.ndarray
    y_cons: Dict[str, np.ndarray]
    best_idx: int


def latin_hypercube(n_pts: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0, 1, n_pts + 1)
    u = rng.uniform(size=(n_pts, dim))
    a = cut[:n_pts, None]
    b = cut[1:n_pts+1, None]
    pts = a + (b - a)*u
    for j in range(dim):
        rng.shuffle(pts[:, j])
    return pts


def scale_to_bounds(X01: np.ndarray) -> np.ndarray:
    lo = BOUNDS[:, 0]; hi = BOUNDS[:, 1]
    return lo + X01*(hi - lo)


def checkpoint(path: str, state: BOState):
    data = {
        "X": state.X.tolist(),
        "y_obj": state.y_obj.tolist(),
        "y_cons": {k: v.tolist() for k, v in state.y_cons.items()},
        "best_idx": int(state.best_idx),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_checkpoint(path: str) -> BOState | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    X = np.array(d["X"], dtype=float)
    y_obj = np.array(d["y_obj"], dtype=float)
    y_cons = {k: np.array(v, dtype=float) for k, v in d["y_cons"].items()}
    return BOState(X=X, y_obj=y_obj, y_cons=y_cons, best_idx=int(d["best_idx"]))


def make_constraints(metrics: Dict[str, float]) -> Dict[str, float]:
    # c(x) <= 0 is feasible
    return {
        "c_stress": metrics["stress_max"] - LIMITS["stress_max"],
        "c_temp":   metrics["temp_max"] - LIMITS["temp_max"],
        "c_life":   (LIMITS["life_min"] - metrics["life"]),
    }


def suggest_batch(state: BOState, n_suggest: int, acq: str, beta_ucb: float = 2.0, xi: float = 0.0, n_starts: int = 64) -> np.ndarray:
    dim = BOUNDS.shape[0]
    # Fit surrogates
    gp_obj = GPWrapper(noise=1e-6)
    gp_obj.fit(state.X, state.y_obj)
    gp_cons = {}
    for cname, y in state.y_cons.items():
        gp = GPWrapper(noise=1e-6)
        gp.fit(state.X, y)
        gp_cons[cname] = gp

    def acq_score(Z01: np.ndarray) -> np.ndarray:
        Z = scale_to_bounds(Z01)
        mu, sd = gp_obj.predict(Z)
        if acq.lower() == "ei":
            base = expected_improvement(mu, sd, np.min(state.y_obj), xi=xi)
        elif acq.lower() == "ucb":
            base = upper_confidence_bound(mu, sd, beta=beta_ucb)
        else:  # constrained EI
            base = expected_improvement(mu, sd, np.min(state.y_obj), xi=xi)
            pfeas = 1.0
            for cname, gp in gp_cons.items():
                m, s = gp.predict(Z)
                pfeas = pfeas * prob_feasible(m, s)
            base = base * pfeas
        return base

    # Greedy fantasized qEI: select points one‑by‑one, updating with predictive mean
    batch = []
    for q in range(n_suggest):
        cand = latin_hypercube(n_starts, dim, rng)
        scores = acq_score(cand)
        z_best = cand[np.argmax(scores)]
        batch.append(z_best)
        # Fantasize by appending (mean) to dataset to encourage diversity
        Z = scale_to_bounds(z_best[None, :])
        mu_o, _ = gp_obj.predict(Z)
        state.X = np.vstack([state.X, Z])
        state.y_obj = np.concatenate([state.y_obj, mu_o])
        for cname, gp in gp_cons.items():
            mu_c, _ = gp.predict(Z)
            state.y_cons[cname] = np.concatenate([state.y_cons[cname], mu_c])
            gp.fit(state.X, state.y_cons[cname])
        gp_obj.fit(state.X, state.y_obj)
    return scale_to_bounds(np.array(batch))


def run_bo(dim: int, iters: int, parallel: int, acq: str, seed: int, ckpt: str | None) -> BOState:
    rng_np = np.random.default_rng(seed)
    # Init
    n_init = max(8, 2*dim)
    state = load_checkpoint(ckpt) if ckpt else None
    if state is None:
        X01 = latin_hypercube(n_init, dim, rng_np)
        X = scale_to_bounds(X01)
        y_obj_list = []
        y_cons = {"c_stress": [], "c_temp": [], "c_life": []}
        for x in X:
            m = evaluate_design(x)
            y_obj_list.append(m["objective"])
            c = make_constraints(m)
            for k in y_cons: y_cons[k].append(c[k])
        state = BOState(X=X, y_obj=np.array(y_obj_list), y_cons={k: np.array(v) for k, v in y_cons.items()}, best_idx=int(np.argmin(y_obj_list)))
        if ckpt: checkpoint(ckpt, state)

    # Main loop
    for it in range(iters):
        # Suggest a batch
        tmp_state = BOState(X=state.X.copy(), y_obj=state.y_obj.copy(), y_cons={k: v.copy() for k, v in state.y_cons.items()}, best_idx=state.best_idx)
        X_batch = suggest_batch(tmp_state, n_suggest=parallel, acq=acq)

        # Parallel evaluate (process pool optional; here simple loop to keep it portable)
        metrics = [evaluate_design(x) for x in X_batch]
        y_new = [m["objective"] for m in metrics]
        C_new = [make_constraints(m) for m in metrics]

        # Append
        state.X = np.vstack([state.X, X_batch])
        state.y_obj = np.concatenate([state.y_obj, np.array(y_new)])
        for k in state.y_cons:
            state.y_cons[k] = np.concatenate([state.y_cons[k], np.array([c[k] for c in C_new])])
        state.best_idx = int(np.argmin(state.y_obj))

        if ckpt: checkpoint(ckpt, state)
        best = state.X[state.best_idx]
        besty = state.y_obj[state.best_idx]
        feas = (
            (state.y_cons["c_stress"][state.best_idx] <= 0.0) and
            (state.y_cons["c_temp"][state.best_idx]   <= 0.0) and
            (state.y_cons["c_life"][state.best_idx]   <= 0.0)
        )
        print(f"Iter {it+1:03d}: best_obj={besty:.5g} feasible={feas} x={best}")

    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=BOUNDS.shape[0])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--parallel", type=int, default=8, help="batch size per BO iteration")
    ap.add_argument("--acq", type=str, default="cEI", choices=["ei", "ucb", "cEI"])  # constrained EI default
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="bo_ckpt.json")
    args = ap.parse_args()

    state = run_bo(dim=args.dim, iters=args.iters, parallel=args.parallel, acq=args.acq, seed=args.seed, ckpt=args.ckpt)
    x_best = state.X[state.best_idx]
    print("\n=== BO Summary ===")
    print("Best x:", x_best)
    print("Best obj:", state.y_obj[state.best_idx])
    print("Feasible?",
          state.y_cons["c_stress"][state.best_idx] <= 0.0 and
          state.y_cons["c_temp"][state.best_idx]   <= 0.0 and
          state.y_cons["c_life"][state.best_idx]   <= 0.0)
    # Exit non‑zero if the best is infeasible
    feas = (
        (state.y_cons["c_stress"][state.best_idx] <= 0.0) and
        (state.y_cons["c_temp"][state.best_idx]   <= 0.0) and
        (state.y_cons["c_life"][state.best_idx]   <= 0.0)
    )
    sys.exit(0 if feas else 2)
