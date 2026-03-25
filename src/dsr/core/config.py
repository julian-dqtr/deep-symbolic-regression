ENV_CONFIG = {
    "max_length": 30,
    "complexity_penalty": 0.01,
    "invalid_reward": -1.0,
    "use_eos": False,
    "numeric_epsilon": 1e-6,
}

GRAMMAR_CONFIG = {
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sin", "cos", "exp", "log"],
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
    "token_embedding_dim": 64,
    "hidden_dim": 512,
    "dataset_embedding_dim": 64,
    "num_lstm_layers": 2,
}

TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "entropy_weight": 0.05,
    "num_episodes": 10000,
    "batch_size": 256,
    "optimizer_name": "adam",
    "grad_clip_norm": 1.0,
}

# import math

# CONFIG = {
#     #  Environment Parameters 
#     "max_length": 30,          # Maximum length of the generated expression
#     "alpha": 0.01,             # Penalty coefficient for expression complexity
#     "dataset_name": "Feynman", # Or "Nguyen"
    
#     #  Neural Network (Agent) Parameters 
#     "embedding_dim": 64,       # Size of the token embeddings
#     "hidden_dim": 128,         # Size of the LSTM hidden state and cell state
#     "encoder_dim": 64,         # Size of the DeepSets dataset representation
#     "num_lstm_layers": 1,      # Number of recurrent layers for the LSTM
    
#     #  RL Training Parameters 
#     "learning_rate": 1e-3,     # Optimizer learning rate
#     "entropy_weight": 0.05,    # Helps the agent explore new expressions
#     "num_episodes": 10000,     # Total number of training loops
#     "batch_size": 256,         # Number of expressions generated before updating weights
# }

# # Dictionary of physical and mathematical constants 
# # These will be automatically converted to PyTorch tensors during evaluation
# CONSTANTS = {
#     "pi": math.pi,
#     "e": math.e,
#     "g": 9.81,            # Gravity (Earth)
#     "c": 299792458.0,     # Speed of light
#     "G": 6.67430e-11,     # Gravitational constant
#     "h": 6.62607015e-34,  # Planck constant
#     "k": 1.380649e-23     # Boltzmann constant
# }

# # The official vocabulary for the RL agent (Action Space)
# # The agent's neural network will output probabilities over this exact list
# ACTION_SPACE = [
#     # Basic math operators
#     "+", "-", "*", "/", "**",
#     # Functions
#     "sin", "cos", "exp", "log", "sqrt", "abs",
#     # Physical and math constants
#     "pi", "e", "g", "c", "G", "h", "k",
#     # Variables (will be dynamically expanded based on the dataset)
#     "x0", "x1", "x2", "x3"
# ]
