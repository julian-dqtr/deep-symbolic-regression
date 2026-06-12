ENV_CONFIG = {
    "max_length": 30,
    "complexity_penalty": 0.01,
    "invalid_reward": -1.0,
    "use_eos": False,
    "numeric_epsilon": 1e-6,
}

GRAMMAR_CONFIG = {
    # Binary operators — pow (x^y) added for Nguyen-11 and polynomial tasks
    "binary_operators": ["+", "-", "*", "/", "pow"],
    # Unary operators — sqrt added for Nguyen-8
    # Both are protected: sqrt(|x|), pow(|a|, b) to avoid domain errors
    "unary_operators": ["sin", "cos", "exp", "log", "sqrt"],
    "constants": {
        "1.0": 1.0,
        "0.5": 0.5,
        "2.0": 2.0,
        "3.0": 3.0,
        "pi": 3.141592653589793,
        "const": 1.0,
    },
    "max_num_variables": 10,
}

MODEL_CONFIG = {
    "token_embedding_dim":   32,
    "hidden_dim":            256,
    "dataset_embedding_dim": 32,
    "num_lstm_layers":       2,
}

TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "entropy_weight": 0.05,
    "num_episodes": 10000,
    "batch_size": 256,
    "optimizer_name": "adam",
    "grad_clip_norm": 1.0,
}