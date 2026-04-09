"""
mdl_reward.py
=============
Minimum Description Length (MDL) reward for symbolic regression.

Motivation
----------
The current reward in env.py is:

    reward = -NMSE - alpha * len(tokens)        (alpha = 0.01, arbitrary)

This linear complexity penalty has no theoretical justification — alpha = 0.01
was chosen by hand and produces inconsistent trade-offs across different
expression lengths and dataset sizes.

MDL principle (Rissanen, 1978)
-------------------------------
The best model is the one that minimises the total description length:

    L_total = L(model) + L(data | model)

where:
  L(model)        = cost of encoding the expression structure (bits)
  L(data | model) = cost of encoding the residuals given the model (bits)

This gives a principled, dataset-aware measure of model quality that
automatically balances fit and complexity — no alpha to tune.

Implementation
--------------
We use a practical MDL approximation:

    L(model) = n_tokens * log2(vocab_size)
               + n_consts * bits_per_constant

      Each token in the prefix expression costs log2(|vocab|) bits.
      Each optimised constant costs `bits_per_constant` bits (default 32,
      matching single-precision float representation).

    L(data | model) = (n/2) * log2(2*pi*e*sigma^2)
      where sigma^2 = MSE (residual variance).
      This is the Shannon entropy of a Gaussian with variance = MSE,
      which gives the minimum code length for i.i.d. Gaussian residuals.

    MDL score (normalised, higher = better):
        score = -(L_model + L_data) / n

    Dividing by n makes the score comparable across dataset sizes.

Reward
------
    reward_mdl = -(L_model + L_data) / n

This is a drop-in replacement for the existing reward. It is on a different
scale than -NMSE, so comparisons with the baseline must account for that —
use the same reward for both variants in ablation studies.

Usage
-----
from dsr.core.mdl_reward import mdl_reward, MdlRewardConfig

# Default config
config = MdlRewardConfig()
reward = mdl_reward(tokens, y_pred, y_true, grammar, config)

# Or integrate into Trainer via use_mdl_reward=True flag
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MdlRewardConfig:
    """
    Parameters for the MDL reward computation.

    bits_per_constant : float
        Bits used to encode each optimised constant.
        32 = single-precision float (default).
        Increase for stricter penalty on constants.

    eps : float
        Minimum MSE to avoid log(0). Also avoids degenerate perfect fit.

    invalid_reward : float
        Reward returned when the expression is invalid (same as env.py default).

    normalise_by_n : bool
        Divide total description length by n (dataset size) so the reward
        is comparable across datasets of different sizes. Default True.
    """
    bits_per_constant: float = 32.0
    eps:               float = 1e-9
    invalid_reward:    float = -1.0
    normalise_by_n:    bool  = True


# ---------------------------------------------------------------------------
# MDL components
# ---------------------------------------------------------------------------

def _model_length(
    tokens: List[str],
    vocab_size: int,
    n_consts: int,
    bits_per_constant: float,
) -> float:
    """
    L(model): bits to encode the expression structure.

    Each token in the prefix sequence costs log2(vocab_size) bits.
    Each optimised constant costs bits_per_constant bits.

    Parameters
    ----------
    tokens            : prefix token sequence
    vocab_size        : total number of tokens in the grammar
    n_consts          : number of 'const' tokens in the expression
    bits_per_constant : bits per optimised constant value
    """
    # Structure cost: each token position from the grammar alphabet
    structure_bits = len(tokens) * math.log2(max(vocab_size, 2))

    # Constant cost: each optimised constant is encoded as a float
    constant_bits = n_consts * bits_per_constant

    return structure_bits + constant_bits


def _data_length(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float,
) -> float:
    """
    L(data | model): bits to encode the residuals given the model.

    Uses the Gaussian MDL formula:
        L = (n/2) * log2(2 * pi * e * sigma^2)
    where sigma^2 = MSE = mean((y - y_pred)^2).

    This is the Shannon entropy of a zero-mean Gaussian with variance = MSE,
    i.e. the minimum code length for i.i.d. Gaussian residuals.

    A perfect fit (MSE → 0) gives L → -inf, but we clip MSE at eps.
    """
    n   = len(y_true)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mse = max(mse, eps)

    # Gaussian code length in bits (nats → bits: divide by ln(2))
    data_bits = (n / 2.0) * math.log2(2.0 * math.pi * math.e * mse)
    return data_bits


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def mdl_reward(
    tokens:    List[str],
    y_pred:    np.ndarray,
    y_true:    np.ndarray,
    vocab_size: int,
    config:    MdlRewardConfig = None,
) -> float:
    """
    Compute the MDL reward for a symbolic expression.

    Parameters
    ----------
    tokens     : prefix token sequence of the expression
    y_pred     : model predictions on the training set
    y_true     : ground-truth targets
    vocab_size : size of the grammar vocabulary
    config     : MdlRewardConfig (uses defaults if None)

    Returns
    -------
    float : MDL reward (higher = better, always negative or zero)
            reward = -(L_model + L_data) / n  if normalise_by_n
            reward = -(L_model + L_data)       otherwise
    """
    if config is None:
        config = MdlRewardConfig()

    # Guard: invalid predictions
    if (not isinstance(y_pred, np.ndarray)
            or y_pred.shape != y_true.shape
            or not np.all(np.isfinite(y_pred))):
        return config.invalid_reward

    n        = len(y_true)
    n_consts = tokens.count("const")

    l_model = _model_length(tokens, vocab_size, n_consts, config.bits_per_constant)
    l_data  = _data_length(y_true, y_pred, config.eps)

    total = l_model + l_data

    if config.normalise_by_n:
        reward = -total / n
    else:
        reward = -total

    return float(reward)


def mdl_reward_from_mse(
    tokens:    List[str],
    mse:       float,
    n:         int,
    vocab_size: int,
    config:    MdlRewardConfig = None,
) -> float:
    """
    Convenience version when MSE is already computed (avoids re-computing
    y_pred - y_true). Used in trainer.py where the evaluator already
    returns nmse and we can recover mse = nmse * var(y).
    """
    if config is None:
        config = MdlRewardConfig()

    if not math.isfinite(mse) or mse < 0:
        return config.invalid_reward

    n_consts = tokens.count("const")
    l_model  = _model_length(tokens, vocab_size, n_consts, config.bits_per_constant)
    l_data   = _data_length_from_mse(mse, n, config.eps)

    total = l_model + l_data
    return float(-total / n if config.normalise_by_n else -total)


def _data_length_from_mse(mse: float, n: int, eps: float) -> float:
    mse = max(mse, eps)
    return (n / 2.0) * math.log2(2.0 * math.pi * math.e * mse)


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_rewards(
    tokens:    List[str],
    nmse:      float,
    mse:       float,
    n:         int,
    vocab_size: int,
    alpha:     float = 0.01,
    config:    MdlRewardConfig = None,
) -> dict:
    """
    Side-by-side comparison of the linear and MDL rewards for an expression.
    Useful for ablation analysis and debugging.

    Parameters
    ----------
    tokens     : prefix token sequence
    nmse       : normalised MSE (from evaluator)
    mse        : raw MSE (nmse * var(y))
    n          : dataset size
    vocab_size : grammar vocabulary size
    alpha      : linear complexity penalty coefficient (default 0.01)
    config     : MdlRewardConfig

    Returns
    -------
    dict with both reward values and their components
    """
    if config is None:
        config = MdlRewardConfig()

    linear_reward = -nmse - alpha * len(tokens)
    mdl_r         = mdl_reward_from_mse(tokens, mse, n, vocab_size, config)

    n_consts = tokens.count("const")
    l_model  = _model_length(tokens, vocab_size, n_consts, config.bits_per_constant)
    l_data   = _data_length_from_mse(mse, n, config.eps)

    return {
        "n_tokens":      len(tokens),
        "n_consts":      n_consts,
        "nmse":          nmse,
        "mse":           mse,
        "linear_reward": round(linear_reward, 6),
        "mdl_reward":    round(mdl_r, 6),
        "L_model_bits":  round(l_model, 2),
        "L_data_bits":   round(l_data, 2),
        "L_total_bits":  round(l_model + l_data, 2),
    }
