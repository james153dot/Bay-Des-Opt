# Bayesian-Design-Optimizer: Stress-Fatigue-Thermal

This is a standalone Bayesian optimization driver for mechanical design under constraints. It uses Gaussian‑process surrogates and constrained Expected Improvement to propose parallel design batches. Plug in your solver/rig via evaluate_design(x); the optimizer handles exploration vs exploitation, feasibility modeling, checkpointing, and CI‑friendly exits.

# Metrics & Constraints

Return from your solver:

objective (float): lower is better (e.g., mass, cost, −reliability)

stress_max ≤ LIMITS["stress_max"]

temp_max ≤ LIMITS["temp_max"]

life ≥ LIMITS["life_min"]

# Quick Start
```
python bayes_optimize.py --dim 4 --iters 60 --parallel 8 --acq cEI --seed 1
```
Then edit:

BOUNDS for your design variables

LIMITS for your specs

evaluate_design(x) to call FEA/rig and postprocess metrics

# Parallel evaluation

--parallel k proposes k designs each iteration; evaluate in your cluster and push measurements back by returning from evaluate_design.

# Notes

Auto‑resumes from --ckpt JSON

Uses scikit‑learn if available, otherwise a minimal GP

Acquisition: EI, UCB, constrained‑EI (default)
