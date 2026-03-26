# Deep Symbolic Regression with Reinforcement Learning

**Authors:** Dequatre Julian, L'Herminé Camille, Jelassi Meriem

---

## Motivation

Discovering concise analytical equations from data is a central challenge for scientific discovery, yet most deep learning methods focus on black-box prediction rather than interpretable structure. Symbolic regression addresses this gap by searching directly over symbolic expressions, but the combinatorial search space becomes extremely large as the number of operators and variables grows.

Recent work on Deep Symbolic Regression (DSR) shows that this search can be naturally framed as an RL problem. We study symbolic regression as an RL task in which the agent learns to discover simple, accurate formulas for small benchmark datasets derived from physics, maximizing a reward that balances data fit and complexity.

---

## Architecture

### Environment
Each episode corresponds to constructing a single mathematical expression for a fixed dataset.

- **State Space:** Partially constructed expression in prefix notation + a permutation-invariant embedding of the dataset (DeepSets encoder).
- **Action Space:** Discrete grammar tokens — variables, operators (`+`, `-`, `×`, `÷`, `sin`, `cos`, `exp`, `log`), constants (`0.5`, `1.0`, `2.0`, `3.0`, `π`), and a learnable `const` placeholder.
- **Terminal Reward:** `R(e) = −NMSE(e) − α·complexity(e)` where complexity is expression length and `α = 0.01`.

### Agent
- **2-layer LSTM** (hidden dim 512) that autoregressively generates mathematical tokens.
- **Vectorized Batched Rollouts:** 256 expression trees are generated simultaneously via PyTorch tensor ops, bypassing Python loops entirely.
- **Risk-Seeking Policy Gradient (RSPG):** Training is done exclusively on the top 5% quantile of sampled expressions per batch.
- **Top-K Memory (Teacher Forcing):** The historical top-20 expressions are re-injected into every gradient update to prevent forgetting.
- **BFGS Constant Optimization:** Expressions with `const` tokens are post-processed by a SciPy BFGS solver (10 iterations) before reward assignment.

---

## Results

Benchmark: **PMLB Feynman Physics Suite** (119 tasks, 50 000 episodes each).

| Quality Tier | NMSE Range | Tasks | % | Cumulative |
|---|---|---|---|---|
| ✅ Excellent | < 0.01 | 9 | 7.6% | 7.6% |
| 🔵 Very Good | 0.01 – 0.05 | 10 | 8.4% | 16.0% |
| 🟡 Good | 0.05 – 0.15 | 8 | 6.7% | 22.7% |
| 🟠 Moderate | 0.15 – 0.50 | 39 | 32.8% | 55.5% |
| 🔴 Poor | ≥ 0.50 | 53 | 44.5% | — |

**Best recovered equations (NMSE < 1e-13):** `x0·x1`, `x0·x1·sin(x2)`, `x0·x1·x2·sin(x3)`.

---

## Installation

### With `uv` (recommended)
```bash
git clone <repo-url>
cd deep-symbolic-regression
uv sync          # reads pyproject.toml + uv.lock → exact reproducible environment
```

### With `pip`
```bash
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Training
```bash
# Full Feynman suite (50 000 episodes per task)
uv run -m src.dsr.training.train --suite pmlb_feynman_all --num_episodes 50000

# Quick test on a small subset
uv run -m src.dsr.training.train --suite pmlb_feynman_subset --num_episodes 1000

# Key hyperparameters
uv run -m src.dsr.training.train \
    --suite pmlb_feynman_all \
    --num_episodes 50000 \
    --learning_rate 0.000335 \
    --entropy_weight 0.017
```

Results are saved automatically to `results/results_<suite>_<timestamp>.csv`.

### Analysing Results
```bash
# Full report with quality tiers and cumulative success rates
uv run results/analyse_results.py
```

### Visualising an Equation
```bash
# Random Excellent / Very Good equation (default)
uv run results/visualize.py

# Specify a CSV file
uv run results/visualize.py --csv results/results_pmlb_feynman_all_50000.csv

# Visualize a specific task
uv run results/visualize.py --task feynman_I_12_1
```

---

## Project Structure

```
deep-symbolic-regression/
├── src/dsr/
│   ├── core/
│   │   ├── config.py          # All hyperparameters and grammar config
│   │   ├── grammar.py         # Token grammar and action space
│   │   ├── env.py             # RL environment (episode logic, reward)
│   │   ├── evaluator.py       # NMSE evaluation + BFGS constant optimization
│   │   └── expression.py      # Prefix ↔ infix conversion and AST utilities
│   ├── models/
│   │   └── policy.py          # LSTM policy + DeepSets dataset encoder
│   ├── analysis/
│   │   ├── memory.py          # Top-K experience replay memory (min-heap)
│   │   └── visualizer.py      # AST tree visualizer (networkx + matplotlib)
│   ├── data/
│   │   └── datasets.py        # PMLB dataset loader with local cache
│   └── training/
│       ├── train.py            # Entry point — CLI + training loop
│       ├── trainer.py          # Trainer class orchestrating all components
│       ├── rollout.py          # Episode collection (sequential + batched)
│       ├── risk_seeking_optimizer.py  # RSPG optimizer
│       └── policy_optimizer.py        # REINFORCE baseline optimizer
├── results/
│   ├── analyse_results.py     # Results analysis script
│   ├── visualize.py           # Equation tree visualizer script
│   └── results_*.csv          # Training outputs
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Pinned dependency lockfile
└── requirements.txt           # Pip-compatible requirements
```

---

## References

- Petersen et al., *Deep Symbolic Regression (DSR)*, ICLR 2021. [arXiv:1912.04871](https://arxiv.org/abs/1912.04871)
- Bastiani et al., *Complexity-Aware DSR with Robust Risk-Seeking Policy Gradients*, 2024. [arXiv:2406.06751](https://arxiv.org/abs/2406.06751)
- Udrescu & Tegmark, *AI Feynman: A Physics-Inspired Method for Symbolic Regression*. [Science Advances](https://www.science.org/doi/10.1126/sciadv.aay2631)
- Samuel Holt et al., *Deep Generative Symbolic Regression (DGSR)*, 2024. [arXiv:2311.14022](https://arxiv.org/abs/2311.14022)
