# Deep Symbolic Regression with Reinforcement Learning

**Authors:** Dequatre Julian, L'Herminé Camille, Jelassi Meriem  
**Institution:** CentraleSupélec  
**Code:** [github.com/julian-dqtr/deep-symbolic-regression](https://github.com/julian-dqtr/deep-symbolic-regression)

---

## Motivation

Discovering concise analytical equations from data is a central challenge for scientific discovery, yet most deep learning methods focus on black-box prediction rather than interpretable structure. Symbolic regression addresses this gap by searching directly over symbolic expressions, but the combinatorial search space becomes extremely large as the number of operators and variables grows.

We study symbolic regression as an RL task in which an agent learns to discover accurate formulas from data, framed as a sequential decision problem: each token generated is an action, the partially-built expression is the state, and the final −NMSE is the reward. Our central question is: **under what conditions can reinforcement learning be made effective for symbolic regression?**

---

## Architecture

### Environment

Each episode corresponds to constructing a single mathematical expression for a fixed dataset.

- **State:** Partially constructed expression in prefix notation + permutation-invariant dataset embedding (DeepSets encoder).
- **Action space:** Discrete grammar tokens, variables `{x0, …, x9}`, binary operators `{+, −, ×, ÷, pow}`, unary operators `{sin, cos, exp, log, sqrt}`, constants `{0.5, 1.0, 2.0, 3.0, π}`, and a learnable `const` placeholder.
- **Grammar-aware action mask:** At each step, a binary mask filters tokens that would make the expression impossible to complete within the length budget. A token is valid iff the resulting pending slot count stays non-negative AND the expression can still be completed within `max_length = 30`. This eliminates structurally invalid actions before the policy sees them.
- **Terminal reward:** `R(e) = −NMSE(e) − α·|e|`, with `α = 0.01`.

### Agent

- **LSTM policy:** 2-layer LSTM (hidden dim 256) autoregressively generating grammar tokens, conditioned on a DeepSets dataset embedding at each step.
- **DeepSets encoder:** Each `(x_i, y_i)` point is independently encoded by an MLP, then aggregated by sum-pooling, producing a permutation-invariant context vector.
- **Vectorised batched rollouts:** 256 expression trees generated simultaneously via PyTorch tensor operations, bypassing Python environment loops (~10× speed-up).
- **Risk-Seeking Policy Gradient (RSPG):** Only the top 5% quantile of sampled expressions per batch contributes to the gradient update.
- **Top-K memory replay:** The best 20 expressions ever found are stored in a min-heap and re-injected into every gradient update via teacher forcing (preventing catastrophic forgetting).
- **BFGS constant optimisation:** Expressions containing `const` tokens are post-processed by a SciPy L-BFGS-B solver (10 iterations) before reward assignment, decoupling discrete structure search from continuous parameter fitting.

### Original Contributions

- **Curriculum learning:** Maximum expression length is linearly ramped from 5 to 30 over the first 50% of training episodes, preventing early exploration of overly complex expressions.
- **DiverseTopKMemory:** Extends Top-K memory with a token-level Levenshtein edit-distance filter, a new expression is rejected if it is structurally too similar to an already stored one (threshold: 3 edits), preventing memory collapse onto near-identical expressions.
- **PrioritizedTopKMemory:** Expressions are replayed in order of "surprise", `|reward − baseline|^α`, focusing gradient updates on the most informative past experience (inspired by Schaul et al., 2016).
- **MDL reward:** A theoretically grounded Minimum Description Length alternative to the hand-tuned linear complexity penalty.

---

## Results

### Nguyen Benchmark (12 tasks, 3 000 episodes each)

| Quality | Tasks | Notes |
|---|---|---|
| Perfect (NMSE < 0.001) | 3 | Including `sqrt(x0)` and `x0^x1` exactly |
| Good (NMSE < 0.05) | 7 | |
| Poor | 2 | Deep compositions (`nguyen_5`, `nguyen_12`) |

**10/12 tasks** recovered with NMSE < 0.05.

### Feynman Benchmark (119 tasks, 50 000 episodes each)

| Quality Tier | NMSE Range | Tasks | % |
|---|---|---|---|
| Perfect | < 10⁻³ | 6 | 5.0% |
| Good | [10⁻³, 0.05) | 14 | 11.8% |
| Moderate | [0.05, 0.5) | 54 | 45.4% |
| Poor | ≥ 0.5 | 45 | 37.8% |

**20/119 tasks** recovered with NMSE < 0.05 (16.8% success rate).

Six tasks recovered at machine precision (NMSE ≈ 0): `I.12.1` (μNₙ), `I.12.5` (q²Eₓ), `I.25.13` (q/C), `I.29.4` (ω/c), `I.39.11` (1/(γ−1)·pr·V), `II.27.18` (nαE²ε).

### Key Findings

- **DeepSets is necessary:** Removing the dataset encoder causes the reward curve to stay flat throughout training, the policy cannot learn without dataset conditioning.
- **BFGS is the most impactful component:** Mean NMSE increases from 0.503 (full) to 0.562 (−BFGS), the largest delta of any ablated component.
- **Curriculum learning** is the clearest of the three original contributions: the −Curriculum variant achieves systematically lower rewards on harder tasks.
- **gplearn comparison:** Genetic programming (gplearn, population 1000, 20 generations) outperforms RSPG on 93/119 Feynman tasks (mean NMSE 0.183 vs 0.418), primarily due to broader exploration diversity. RSPG is competitive on moderate-complexity tasks requiring precise constant tuning.
- **Beam search and MCTS** do not consistently improve results and can degrade perfect solutions, the quality of the learned policy is the true bottleneck.

---

## Installation

### With `uv` (recommended)

```bash
git clone https://github.com/julian-dqtr/deep-symbolic-regression.git
cd deep-symbolic-regression
uv sync          # reads pyproject.toml + uv.lock → exact reproducible environment
```

### With `pip`

```bash
pip install torch numpy scipy matplotlib networkx gplearn optuna pmlb
pip install -e .
```

---

## Usage

All commands are run from the `src/` directory:

```bash
cd src
```

### Run unit tests (48 tests)

```bash
python -m dsr.tests.test_core
```

### Training

```bash
# Nguyen benchmark (fast validation, ~1h30)
python -m dsr.training.evaluate_expressions --suite nguyen --num_episodes 3000

# Full Feynman suite (119 tasks, 50 000 episodes, ~6h GPU)
python -m dsr.training.evaluate_expressions --suite pmlb_feynman_all --num_episodes 50000

# Feynman subset only (4 tasks, quick test)
python -m dsr.training.train --suite pmlb_feynman_subset --num_episodes 5000

# Key hyperparameters (Optuna-tuned)
python -m dsr.training.train \
    --suite pmlb_feynman_all \
    --num_episodes 50000 \
    --learning_rate 0.000335 \
    --entropy_weight 0.017
```

### Optimizer comparison (RSPG vs REINFORCE vs PPO)

```bash
python -m dsr.training.compare_optimizers \
    --suite pmlb_feynman_subset \
    --num_episodes 5000 \
    --num_seeds 3
```

### Ablation studies

```bash
# System components (Memory / BFGS / DeepSets)
python -m dsr.training.ablation_study \
    --suite pmlb_feynman_subset \
    --num_episodes 5000 --num_seeds 3

# Original contributions (Curriculum / DiverseMemory / PrioritizedReplay)
python -m dsr.training.ablation_contributions \
    --suite pmlb_feynman_subset \
    --num_episodes 5000 --num_seeds 3
```

### Baseline comparison (gplearn)

```bash
python -m dsr.baselines.baseline_gplearn \
    --suite pmlb_feynman_subset \
    --rspg_csv results/true_vs_recovered_<timestamp>.csv
```

### Post-training inference

```bash
# Beam search (width=50)
python -m dsr.training.beam_search \
    --suite pmlb_feynman_subset \
    --checkpoint_dir checkpoints \
    --beam_width 50

# MCTS (200 simulations)
python -m dsr.training.mcts \
    --suite pmlb_feynman_subset \
    --checkpoint_dir checkpoints \
    --num_simulations 200
```

### Qualitative analysis

```bash
python -m dsr.analysis.qualitative_analysis \
    --csv results/true_vs_recovered_<timestamp>.csv
```

Results are saved automatically to `results/`.

---

## Project Structure

```
deep-symbolic-regression/
├── src/dsr/
│   ├── core/
│   │   ├── config.py                  # Hyperparameters and grammar config
│   │   ├── grammar.py                 # Token vocabulary and arity rules
│   │   ├── factory.py                 # Grammar builder
│   │   ├── env.py                     # RL environment (step, mask, reward)
│   │   ├── evaluator.py               # Prefix evaluator + BFGS optimisation
│   │   ├── expression.py              # Prefix ↔ infix conversion, AST utilities
│   │   └── mdl_reward.py              # MDL reward (Rissanen 1978)
│   ├── models/
│   │   └── policy.py                  # LSTM policy + DeepSets encoder
│   ├── training/
│   │   ├── rollout.py                 # Episode collection (sequential + batched)
│   │   ├── policy_optimizer.py        # REINFORCE with EMA baseline
│   │   ├── ppo_optimizer.py           # PPO (clipped surrogate)
│   │   ├── risk_seeking_optimizer.py  # RSPG (elite-only gradient)
│   │   ├── trainer.py                 # Main training loop + checkpoint API
│   │   ├── train.py                   # CLI entry point
│   │   ├── evaluate_expressions.py    # True vs recovered table
│   │   ├── compare_optimizers.py      # RSPG vs REINFORCE vs PPO
│   │   ├── ablation_study.py          # Memory / BFGS / DeepSets ablation
│   │   ├── ablation_contributions.py  # Curriculum / Diverse / Prioritized ablation
│   │   ├── ablation_mdl.py            # Linear vs MDL reward ablation
│   │   ├── beam_search.py             # Beam search post-training inference
│   │   ├── mcts.py                    # AlphaZero-style MCTS inference
│   │   ├── multitask_trainer.py       # Shared policy across task pool
│   │   ├── warm_start.py              # Meta-learning by embedding similarity
│   │   ├── zero_shot_eval.py          # Zero-shot / few-shot generalisation
│   │   ├── tune_lstm.py               # Optuna LSTM architecture search
│   │   └── run_optuna.py              # Optuna lr / entropy search
│   ├── analysis/
│   │   ├── memory.py                  # TopK / Diverse / Prioritized memory
│   │   ├── qualitative_analysis.py    # Failure analysis and plots
│   │   └── visualizer.py             # AST tree + training curve plots
│   ├── data/
│   │   ├── datasets.py                # Nguyen + Feynman PMLB task suites
│   │   └── feynman_ground_truth.py    # Ground-truth expressions + difficulty
│   ├── baselines/
│   │   └── baseline_gplearn.py        # gplearn comparison
│   └── tests/
│       └── test_core.py               # 48 unit tests
├── pyproject.toml
├── uv.lock
└── requirements.txt
```

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| LSTM hidden dim | 256 | 2 layers |
| Batch size | 256 | expressions / step |
| RSPG quantile ε | 5% | top risk-seeking subset |
| Top-K memory | 20 | min-heap |
| Learning rate | 3.35×10⁻⁴ | Adam, Optuna-tuned |
| Entropy weight | 0.017 | exploration bonus, Optuna-tuned |
| BFGS iterations | 10 | per `const` placeholder |
| Complexity α | 0.01 | length penalty |
| Max expression length | 30 | tokens |
| Edit distance threshold | 3 | DiverseTopKMemory |

---

## References

- Petersen et al., *Deep Symbolic Regression*, ICLR 2021. [arXiv:1912.04871](https://arxiv.org/abs/1912.04871)
- Bastiani et al., *Complexity-Aware DSR with Robust Risk-Seeking Policy Gradients*, 2024. [arXiv:2406.06751](https://arxiv.org/abs/2406.06751)
- Udrescu & Tegmark, *AI Feynman: A Physics-Inspired Method for Symbolic Regression*, Science Advances 2020.
- Holt et al., *Deep Generative Symbolic Regression*, 2024. [arXiv:2401.00282](https://arxiv.org/abs/2401.00282)
- Schaul et al., *Prioritized Experience Replay*, ICLR 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)
- Rissanen, J., *Modeling by shortest data description*, Automatica 1978.
