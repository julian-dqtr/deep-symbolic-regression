CONFIG = {
    #  Environment Parameters 
    "max_length": 30,          # Maximum length of the generated expression
    "alpha": 0.01,             # Penalty coefficient for expression complexity
    "dataset_name": "Feynman", # Or "Nguyen"
    
    #  Neural Network (Agent) Parameters 
    "embedding_dim": 64,       # Size of the token embeddings
    "hidden_dim": 128,         # Size of the LSTM hidden state and cell state
    "encoder_dim": 64,         # Size of the DeepSets dataset representation
    "num_lstm_layers": 1,      # Number of recurrent layers for the LSTM
    
    #  RL Training Parameters 
    "learning_rate": 1e-3,     # Optimizer learning rate
    "entropy_weight": 0.05,    # Helps the agent explore new expressions
    "num_episodes": 10000,     # Total number of training loops
    "batch_size": 256,         # Number of expressions generated before updating weights
}
