
import torch
import torch.nn as nn
from config import CONFIG 

#-------------------------------
# Camille: ENVIRONMENT AND DATA
#-------------------------------

class SymbolicEnv:
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.alpha = CONFIG["alpha"]
        self.max_length = CONFIG["max_length"]
        self.vocab = {} # TODO: Fill with variables, operators, and constants
        self.current_expression = []

    def reset(self):
        self.current_expression = []
        # TODO: Return initial state (empty seq, dataset embedding)
        pass

    def step(self, action):
        # TODO: Append action, check validity, calculate reward if done
        pass



#-------------------------------
# Meriem: RL AGENT AND POLICY
#-------------------------------

class DeepSetsEncoder(nn.Module):
    """
    Encodes the permutation-invariant dataset into a fixed-size embedding.
    """
    def __init__(self):
        super(DeepSetsEncoder, self).__init__()
        # TODO: Define point-wise MLP and aggregation function
        pass

    def forward(self, dataset_x, dataset_y):
        raise NotImplementedError

class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, CONFIG["embedding_dim"])
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=CONFIG["embedding_dim"], 
            hidden_size=CONFIG["hidden_dim"], 
            num_layers=CONFIG["num_lstm_layers"],
            batch_first=True
        )
        self.fc = nn.Linear(CONFIG["hidden_dim"], vocab_size)

    def forward(self, partial_expression, dataset_embedding, hidden_state=None, cell_state=None):
        """
        Note: LSTM requires passing both hidden_state and cell_state between steps.
        """
        raise NotImplementedError

class RLAgent:
    def __init__(self, policy_network):
        self.policy = policy_network
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=CONFIG["learning_rate"]
        )

    def select_action(self, state):
        pass

    def update_policy(self, rewards, log_probs, entropies):
        pass

#----------------------------------------------
# Julian: EVALUATION AND VISUALIZATION + OPTUNA
#----------------------------------------------

class Evaluator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def random_search_baseline(self, num_trials):
        pass

    def calculate_metrics(self, test_x, test_y):
        pass
