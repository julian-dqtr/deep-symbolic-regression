from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class SymbolicTask:
    name: str
    num_variables: int
    target_function: Callable[[np.ndarray], np.ndarray]
    target_expression: str
    x_range: Tuple[float, float] = (-1.0, 1.0)
    num_samples: int = 100

    def generate(self):
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


def task_x(X):
    return X[:, 0]


def task_x_plus_1(X):
    return X[:, 0] + 1.0


def task_x_squared(X):
    return X[:, 0] * X[:, 0]


def task_sin_x(X):
    return np.sin(X[:, 0])


def task_cos_x(X):
    return np.cos(X[:, 0])


def task_exp_x(X):
    return np.exp(np.clip(X[:, 0], -5.0, 5.0))


def task_x_sin_x(X):
    return X[:, 0] * np.sin(X[:, 0])


def get_toy_tasks(num_samples: int = 100) -> List[SymbolicTask]:
    return [
        SymbolicTask(
            name="x",
            num_variables=1,
            target_function=task_x,
            target_expression="x0",
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="x_plus_1",
            num_variables=1,
            target_function=task_x_plus_1,
            target_expression="(x0 + 1.0)",
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="x_squared",
            num_variables=1,
            target_function=task_x_squared,
            target_expression="(x0 * x0)",
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="sin_x",
            num_variables=1,
            target_function=task_sin_x,
            target_expression="sin(x0)",
            num_samples=num_samples,
        ),
    ]


def get_nguyen_style_tasks(num_samples: int = 100) -> List[SymbolicTask]:
    return [
        SymbolicTask("nguyen_x", 1, task_x, "x0", num_samples=num_samples),
        SymbolicTask("nguyen_x_plus_1", 1, task_x_plus_1, "(x0 + 1.0)", num_samples=num_samples),
        SymbolicTask("nguyen_x_squared", 1, task_x_squared, "(x0 * x0)", num_samples=num_samples),
        SymbolicTask("nguyen_sin_x", 1, task_sin_x, "sin(x0)", num_samples=num_samples),
        SymbolicTask("nguyen_cos_x", 1, task_cos_x, "cos(x0)", num_samples=num_samples),
        SymbolicTask("nguyen_exp_x", 1, task_exp_x, "exp(x0)", num_samples=num_samples),
        SymbolicTask("nguyen_x_sin_x", 1, task_x_sin_x, "(x0 * sin(x0))", num_samples=num_samples),
    ]


def get_feynman_style_tasks(num_samples: int = 100) -> List[SymbolicTask]:
    def feynman_like_1(X):
        return X[:, 0] * X[:, 0]

    def feynman_like_2(X):
        return X[:, 0] * X[:, 0] + 1.0

    return [
        SymbolicTask(
            name="feynman_style_v_squared",
            num_variables=1,
            target_function=feynman_like_1,
            target_expression="(x0 * x0)",
            num_samples=num_samples,
        ),
        SymbolicTask(
            name="feynman_style_harmonic",
            num_variables=1,
            target_function=feynman_like_2,
            target_expression="((x0 * x0) + 1.0)",
            num_samples=num_samples,
        ),
    ]


def get_task_suite(name: str, num_samples: int = 100) -> List[SymbolicTask]:
    name = name.lower()

    if name == "toy":
        return get_toy_tasks(num_samples=num_samples)
    if name == "nguyen":
        return get_nguyen_style_tasks(num_samples=num_samples)
    if name == "feynman":
        return get_feynman_style_tasks(num_samples=num_samples)

    raise ValueError(f"Unknown task suite: {name}")


def get_task_by_name(task_name: str, num_samples: int = 100) -> SymbolicTask:
    """
    Find a task by its exact name across all available suites.
    """
    all_tasks = []
    all_tasks.extend(get_toy_tasks(num_samples=num_samples))
    all_tasks.extend(get_nguyen_style_tasks(num_samples=num_samples))
    all_tasks.extend(get_feynman_style_tasks(num_samples=num_samples))

    for task in all_tasks:
        if task.name == task_name:
            return task

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