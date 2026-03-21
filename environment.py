import torch
import torch.nn as nn
import numpy as np
from config import CONFIG, ACTION_SPACE, CONSTANTS

class DeepSetsEncoder(nn.Module):
    def __init__(self, input_dim=2):
        super(DeepSetsEncoder, self).__init__()
        hidden_dim = CONFIG["encoder_dim"]
        output_dim = CONFIG["embedding_dim"]

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
      
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, y):
        points = torch.cat([x, y], dim=-1) 
        phi_out = self.phi(points)
        sum_out = torch.sum(phi_out, dim=0)
        return self.rho(sum_out)

class SymbolicEnv:
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.alpha = CONFIG["alpha"]
        self.max_length = CONFIG["max_length"]
        self.action_space = ACTION_SPACE
        self.current_expression = []
        
        self.arity = {
            "+": 2, "-": 2, "*": 2, "/": 2, "**": 2,
            "sin": 1, "cos": 1, "exp": 1, "log": 1, "sqrt": 1, "abs": 1,
            "pi": 0, "e": 0, "g": 0, "c": 0, "G": 0, "h": 0, "k": 0,
            "x0": 0, "x1": 0, "x2": 0, "x3": 0, "<EOS>": 0
        }
        
        # On passe input_dim = colonnes de X + 1 (pour Y)
        self.encoder = DeepSetsEncoder(input_dim=dataset_x.shape[1] + 1)
        self.reset()

    def reset(self):
        self.current_expression = []
        self.terminal = False
        with torch.no_grad():
            x_tensor = torch.tensor(self.dataset_x, dtype=torch.float32)
            y_tensor = torch.tensor(self.dataset_y, dtype=torch.float32).view(-1, 1)
            self.dataset_embedding = self.encoder(x_tensor, y_tensor)
        return self.current_expression, self.dataset_embedding

    def step(self, action_idx):
        token = self.action_space[action_idx]
        self.current_expression.append(token)
        
        # Condition de fin : soit le jeton <EOS>, soit longueur max
        done = (token == "<EOS>") or (len(self.current_expression) >= self.max_length)
        
        reward = 0
        if done:
            reward = self._calculate_reward()
            self.terminal = True
            
        return self.current_expression, reward, done

    def _evaluate_expression(self, expr, x_data):
        stack = []
        for token in reversed(expr):
            if token == "<EOS>": continue
            
            if token.startswith("x"):
                idx = int(token[1:])
                stack.append(x_data[:, idx])
            elif token in CONSTANTS:
                stack.append(np.full(len(x_data), CONSTANTS[token]))
            elif self.arity.get(token) == 1:
                if not stack: return np.zeros(len(x_data))
                a = stack.pop()
                if token == "sin": stack.append(np.sin(a))
                if token == "cos": stack.append(np.cos(a))
                if token == "exp": stack.append(np.exp(np.clip(a, -20, 20)))
                if token == "log": stack.append(np.log(np.abs(a) + 1e-9))
                if token == "sqrt": stack.append(np.sqrt(np.abs(a)))
                if token == "abs": stack.append(np.abs(a))
            elif self.arity.get(token) == 2:
                if len(stack) < 2: return np.zeros(len(x_data))
                a, b = stack.pop(), stack.pop()
                if token == "+": stack.append(a + b)
                if token == "-": stack.append(a - b)
                if token == "*": stack.append(a * b)
                if token == "/": stack.append(a / (b + 1e-9))
                if token == "**": stack.append(np.power(np.abs(a), np.clip(b, -5, 5)))
                
        return stack[0] if (stack and isinstance(stack[0], np.ndarray)) else np.zeros(len(x_data))

    def _calculate_reward(self):
        try:
            y_pred = self._evaluate_expression(self.current_expression, self.dataset_x)
            mse = np.mean((self.dataset_y - y_pred)**2)
            var_y = np.var(self.dataset_y)
            nmse = mse / var_y if var_y > 0 else mse
            complexity = len(self.current_expression)
            reward = -nmse - (self.alpha * complexity)
            return reward if np.isfinite(reward) else -1000.0
        except:
            return -1000.0