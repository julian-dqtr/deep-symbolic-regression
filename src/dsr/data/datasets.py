import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SymbolicTask:
    name: str
    num_variables: int
    target_function: Callable[[np.ndarray], np.ndarray] | None = None
    target_expression: str = ""
    x_range: Tuple[float, float] = (-1.0, 1.0)
    num_samples: int = 100
    X_data: np.ndarray | None = None
    y_data: np.ndarray | None = None

    def generate(self):
        if self.X_data is not None and self.y_data is not None:
            num_samples = min(self.num_samples, len(self.X_data))
            indices = np.random.choice(len(self.X_data), num_samples, replace=False)
            return self.X_data[indices], self.y_data[indices]

        if self.target_function is None:
            raise ValueError(f"No data or function provided for {self.name} task.")

        X = np.linspace(
            self.x_range[0],
            self.x_range[1],
            self.num_samples,
            dtype=np.float32,
        ).reshape(-1, 1)

        if self.num_variables > 1:
            # Each column is an independent uniform sample over x_range
            rng = np.random.default_rng(42)
            X = rng.uniform(
                self.x_range[0], self.x_range[1],
                (self.num_samples, self.num_variables),
            ).astype(np.float32)

        y = self.target_function(X).astype(np.float32)
        return X, y


# ---------------------------------------------------------------------------
# PMLB (Feynman) datasets
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dsr", "pmlb")


def fetch_pmlb_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X"], data["y"]

    url = (
        f"https://github.com/EpistasisLab/pmlb/raw/master/datasets"
        f"/{name}/{name}.tsv.gz"
    )
    print(f"  [cache miss] Downloading {name}...", flush=True)
    df = pd.read_csv(url, sep="\t", compression="gzip")
    df.dropna(inplace=True)
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    np.savez(cache_path, X=X, y=y)
    return X, y


def get_pmlb_task(name: str, num_samples: int = 100) -> SymbolicTask:
    X, y = fetch_pmlb_dataset(name)
    return SymbolicTask(
        name=name,
        num_variables=X.shape[1],
        target_expression=f"pmlb_{name}",
        num_samples=num_samples,
        X_data=X,
        y_data=y,
    )


# ---------------------------------------------------------------------------
# Nguyen benchmark (Uy et al., 2011)
#
# Standard symbolic regression benchmark used in DSR (Petersen et al., 2021)
# and most subsequent papers. Expressions range from simple polynomials to
# nested trigonometric functions, allowing validation of the system on known
# ground-truth formulas before moving to the harder Feynman benchmark.
#
# All tasks use x ∈ [-1, 1] except where noted.
# Multi-variable tasks (N-10 to N-12) use independent uniform samples.
#
# Reference
# ---------
# Uy, N.Q. et al. "Semantically-based crossover in genetic programming."
# Genetic Programming and Evolvable Machines, 12(3), 2011.
# ---------------------------------------------------------------------------

def _get_nguyen_tasks(num_samples: int = 100) -> List[SymbolicTask]:
    """
    Returns all 12 standard Nguyen tasks as SymbolicTask objects.

    Tasks N-1 to N-9 use one variable (x0).
    Tasks N-10 to N-12 use two variables (x0, x1).
    """
    tasks = [
        # --- Single variable ---
        SymbolicTask(
            name="nguyen_1",
            num_variables=1,
            target_function=lambda X: X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0],
            target_expression="x0^3 + x0^2 + x0",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_2",
            num_variables=1,
            target_function=lambda X: (
                X[:, 0] ** 4 + X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]
            ),
            target_expression="x0^4 + x0^3 + x0^2 + x0",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_3",
            num_variables=1,
            target_function=lambda X: (
                X[:, 0] ** 5 + X[:, 0] ** 4 + X[:, 0] ** 3
                + X[:, 0] ** 2 + X[:, 0]
            ),
            target_expression="x0^5 + x0^4 + x0^3 + x0^2 + x0",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_4",
            num_variables=1,
            target_function=lambda X: (
                X[:, 0] ** 6 + X[:, 0] ** 5 + X[:, 0] ** 4
                + X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]
            ),
            target_expression="x0^6 + x0^5 + x0^4 + x0^3 + x0^2 + x0",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_5",
            num_variables=1,
            target_function=lambda X: (
                np.sin(X[:, 0] ** 2) * np.cos(X[:, 0]) - 1.0
            ),
            target_expression="sin(x0^2) * cos(x0) - 1",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_6",
            num_variables=1,
            target_function=lambda X: (
                np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0] ** 2)
            ),
            target_expression="sin(x0) + sin(x0 + x0^2)",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_7",
            num_variables=1,
            target_function=lambda X: np.log(X[:, 0] + 1.0) + np.log(X[:, 0] ** 2 + 1.0),
            target_expression="log(x0 + 1) + log(x0^2 + 1)",
            x_range=(0.0, 2.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_8",
            num_variables=1,
            target_function=lambda X: np.sqrt(X[:, 0]),
            target_expression="sqrt(x0)",
            x_range=(0.0, 4.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_9",
            num_variables=1,
            target_function=lambda X: np.sin(X[:, 0]) + np.sin(X[:, 0] ** 2),
            target_expression="sin(x0) + sin(x0^2)",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        # --- Two variables ---
        SymbolicTask(
            name="nguyen_10",
            num_variables=2,
            target_function=lambda X: 2.0 * np.sin(X[:, 0]) * np.cos(X[:, 1]),
            target_expression="2*sin(x0)*cos(x1)",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_11",
            num_variables=2,
            target_function=lambda X: X[:, 0] ** X[:, 1],
            target_expression="x0^x1",
            x_range=(0.0, 1.0),   # x > 0 required for x^y
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="nguyen_12",
            num_variables=2,
            target_function=lambda X: (
                X[:, 0] ** 4 - X[:, 0] ** 3
                + 0.5 * X[:, 1] ** 2 - X[:, 1]
            ),
            target_expression="x0^4 - x0^3 + x0^2/2 - x1",
            x_range=(-1.0, 1.0),
            num_samples=num_samples,
        ),
    ]
    return tasks


def get_nguyen_task(name: str, num_samples: int = 100) -> SymbolicTask:
    """Return a single Nguyen task by name (e.g. 'nguyen_1')."""
    all_tasks = {t.name: t for t in _get_nguyen_tasks(num_samples=num_samples)}
    if name not in all_tasks:
        raise ValueError(
            f"Unknown Nguyen task: {name!r}. "
            f"Valid names: {sorted(all_tasks.keys())}"
        )
    return all_tasks[name]


# ---------------------------------------------------------------------------
# Task suite registry
# ---------------------------------------------------------------------------

def get_task_suite(name: str, num_samples: int = 100) -> List[SymbolicTask]:
    name = name.lower()

    # --- Nguyen suites ---
    if name == "nguyen":
        return _get_nguyen_tasks(num_samples=num_samples)

    if name == "nguyen_univariate":
        # N-1 to N-9: single variable — good for initial validation
        return [t for t in _get_nguyen_tasks(num_samples=num_samples)
                if t.num_variables == 1]

    if name == "nguyen_bivariate":
        # N-10 to N-12: two variables
        return [t for t in _get_nguyen_tasks(num_samples=num_samples)
                if t.num_variables == 2]

    # --- Feynman suites ---
    if name == "pmlb_feynman_subset":
        feynman_names = [
            "feynman_I_10_7", "feynman_I_11_19",
            "feynman_I_12_1", "feynman_I_12_11",
        ]
        return [get_pmlb_task(n, num_samples) for n in feynman_names]

    if name == "pmlb_feynman_all":
        feynman_names = [
            "feynman_I_10_7", "feynman_I_11_19", "feynman_I_12_1", "feynman_I_12_11", "feynman_I_12_2",
            "feynman_I_12_4", "feynman_I_12_5", "feynman_I_13_12", "feynman_I_13_4", "feynman_I_14_3",
            "feynman_I_14_4", "feynman_I_15_10", "feynman_I_15_3t", "feynman_I_15_3x", "feynman_I_16_6",
            "feynman_I_18_12", "feynman_I_18_14", "feynman_I_18_4", "feynman_I_24_6", "feynman_I_25_13",
            "feynman_I_26_2", "feynman_I_27_6", "feynman_I_29_16", "feynman_I_29_4", "feynman_I_30_3",
            "feynman_I_30_5", "feynman_I_32_17", "feynman_I_32_5", "feynman_I_34_1", "feynman_I_34_14",
            "feynman_I_34_27", "feynman_I_34_8", "feynman_I_37_4", "feynman_I_38_12", "feynman_I_39_1",
            "feynman_I_39_11", "feynman_I_39_22", "feynman_I_40_1", "feynman_I_41_16", "feynman_I_43_16",
            "feynman_I_43_31", "feynman_I_43_43", "feynman_I_44_4", "feynman_I_47_23", "feynman_I_48_2",
            "feynman_I_50_26", "feynman_I_6_2", "feynman_I_6_2a", "feynman_I_6_2b", "feynman_I_8_14",
            "feynman_I_9_18", "feynman_II_10_9", "feynman_II_11_20", "feynman_II_11_27", "feynman_II_11_28",
            "feynman_II_11_3", "feynman_II_13_17", "feynman_II_13_23", "feynman_II_13_34", "feynman_II_15_4",
            "feynman_II_15_5", "feynman_II_21_32", "feynman_II_24_17", "feynman_II_27_16", "feynman_II_27_18",
            "feynman_II_2_42", "feynman_II_34_11", "feynman_II_34_2", "feynman_II_34_29a", "feynman_II_34_29b",
            "feynman_II_34_2a", "feynman_II_35_18", "feynman_II_35_21", "feynman_II_36_38", "feynman_II_37_1",
            "feynman_II_38_14", "feynman_II_38_3", "feynman_II_3_24", "feynman_II_4_23", "feynman_II_6_11",
            "feynman_II_6_15a", "feynman_II_6_15b", "feynman_II_8_31", "feynman_II_8_7", "feynman_III_10_19",
            "feynman_III_12_43", "feynman_III_13_18", "feynman_III_14_14", "feynman_III_15_12", "feynman_III_15_14",
            "feynman_III_15_27", "feynman_III_17_37", "feynman_III_19_51", "feynman_III_21_20", "feynman_III_4_32",
            "feynman_III_4_33", "feynman_III_7_38", "feynman_III_8_54", "feynman_III_9_52", "feynman_test_1",
            "feynman_test_2", "feynman_test_3", "feynman_test_4", "feynman_test_5", "feynman_test_6",
            "feynman_test_7", "feynman_test_8", "feynman_test_9", "feynman_test_10", "feynman_test_11",
            "feynman_test_12", "feynman_test_13", "feynman_test_14", "feynman_test_15", "feynman_test_16",
            "feynman_test_17", "feynman_test_18", "feynman_test_19", "feynman_test_20",
        ]
        return [get_pmlb_task(n, num_samples) for n in feynman_names]

    raise ValueError(
        f"Unknown task suite: {name!r}. "
        f"Valid suites: nguyen, nguyen_univariate, nguyen_bivariate, "
        f"pmlb_feynman_subset, pmlb_feynman_all"
    )


# ---------------------------------------------------------------------------
# Task lookup by name
# ---------------------------------------------------------------------------

def get_task_by_name(task_name: str, num_samples: int = 100) -> SymbolicTask:
    """Return a task by its exact name, searching all available suites."""

    # Nguyen tasks — generated locally, no download required
    if task_name.startswith("nguyen_"):
        return get_nguyen_task(task_name, num_samples=num_samples)

    # Feynman tasks — downloaded from PMLB
    if task_name.startswith("feynman_"):
        return get_pmlb_task(task_name, num_samples=num_samples)

    raise ValueError(
        f"Unknown task name: {task_name!r}. "
        f"Expected a name starting with 'nguyen_' or 'feynman_'."
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def load_dataset(task_name: str, num_samples: int = 100) -> dict:
    """
    Load a task by name and return a dict with X, y, and metadata.
    Used by scripts that need a quick one-liner to get data.
    """
    task = get_task_by_name(task_name=task_name, num_samples=num_samples)
    X, y = task.generate()
    return {
        "name":          task.name,
        "X":             X,
        "y":             y,
        "num_variables": task.num_variables,
        "target":        task.target_expression,
        "task":          task,
    }