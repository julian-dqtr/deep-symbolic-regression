import torch
from evaluator import Evaluator

def run_tests():
    print("--- Starting Evaluator Tests ---\n")
    
    # 1. Initialize the evaluator with a single variable 'x0'
    evaluator = Evaluator(var_names=['x0'])
    
    # 2. Create dummy data (5 examples)
    # X_batch has the shape (batch_size, num_vars) -> (5, 1)
    X_batch = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    
    # Ground Truth: let's say the true formula is y = 2 * x0
    y_true = 2 * X_batch[:, 0]
    
    # Test 1: The agent finds the PERFECT equation
    tokens_perfect = ['2', '*', 'x0']
    res_perfect = evaluator.evaluate_episode(tokens_perfect, X_batch, y_true)
    print(f"Test 1 (Perfect equation '2*x0'):\n{res_perfect}\n")
    # Expected: is_valid=True, nmse=0.0 (or very close)
    
    # Test 2: The agent finds a WRONG but mathematically valid equation
    tokens_wrong = ['x0', '+', '1']
    res_wrong = evaluator.evaluate_episode(tokens_wrong, X_batch, y_true)
    print(f"Test 2 (Wrong equation 'x0+1'):\n{res_wrong}\n")
    # Expected: is_valid=True, nmse > 0
    
    # Test 3: The agent writes nonsense (Syntax Error)
    tokens_syntax = ['+', '*', 'x0']
    res_syntax = evaluator.evaluate_episode(tokens_syntax, X_batch, y_true)
    print(f"Test 3 (Syntax Error '+*x0'):\n{res_syntax}\n")
    # Expected: is_valid=False, nmse=1.0 (max penalty)
    
    # Test 4: The agent does a division by zero (1 / (x0 - x0))
    tokens_math_error = ['1', '/', '(', 'x0', '-', 'x0', ')']
    res_math = evaluator.evaluate_episode(tokens_math_error, X_batch, y_true)
    print(f"Test 4 (Division by zero):\n{res_math}\n")
    # Expected: is_valid=True (syntax is correct), nmse=1.0 (due to infinity penalty)

if __name__ == "__main__":
    run_tests()