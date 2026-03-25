from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


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
            X = np.tile(X, (1, self.num_variables))

        y = self.target_function(X).astype(np.float32)
        return X, y

import pandas as pd

def fetch_pmlb_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    url = f"https://github.com/EpistasisLab/pmlb/raw/master/datasets/{name}/{name}.tsv.gz"
    df = pd.read_csv(url, sep='\t', compression='gzip')
    df.dropna(inplace=True)
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
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





def get_task_suite(name: str, num_samples: int = 100) -> List[SymbolicTask]:
    name = name.lower()

    if name == "pmlb_feynman_subset":
        feynman_names = ["feynman_I_10_7", "feynman_I_11_19", "feynman_I_12_1", "feynman_I_12_11"]
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
            "feynman_test_17", "feynman_test_18", "feynman_test_19", "feynman_test_20"
        ]
        return [get_pmlb_task(n, num_samples) for n in feynman_names]



    raise ValueError(f"Unknown task suite: {name}")


def get_task_by_name(task_name: str, num_samples: int = 100) -> SymbolicTask:
    """
    Find a task by its exact name across all available suites.
    """
    all_tasks = []

    for task in all_tasks:
        if task.name == task_name:
            return task

    if task_name.startswith("feynman_"):
        try:
            return get_pmlb_task(task_name, num_samples=num_samples)
        except Exception as e:
            pass # Suppress warning if it's completely unknown

    raise ValueError(f"Unknown task name: {task_name}")


def load_dataset(task_name: str, num_samples: int = 100):
    """
    Convenience wrapper used by scripts.
    Returns a dictionary ready for experiments.
    """
    task = get_task_by_name(task_name=task_name, num_samples=num_samples)
    X, y = task.generate()

    return {
        "name": task.name,
        "X": X,
        "y": y,
        "num_variables": task.num_variables,
        "target": task.target_expression,
        "task": task,
    }