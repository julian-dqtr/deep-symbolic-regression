import torch
from evaluator import Evaluator

def run_tests():
    print("--- Starting Evaluator Tests ---\n")
    
    # 1. Initialisation de l'évaluateur avec une seule variable 'x0'
    evaluator = Evaluator(var_names=['x0'])
    
    # 2. Création de fausses données (5 exemples)
    # X_batch a la forme (batch_size, num_vars) -> (5, 1)
    X_batch = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    
    # La vérité terrain (Ground Truth) : disons que la vraie formule est y = 2 * x0
    y_true = 2 * X_batch[:, 0]
    
    # Test 1 : L'agent trouve la formule PARFAITE
    tokens_perfect = ['2', '*', 'x0']
    res_perfect = evaluator.evaluate_episode(tokens_perfect, X_batch, y_true)
    print(f"Test 1 (Perfect equation '2*x0'):\n{res_perfect}\n")
    # Attendu : is_valid=True, nmse=0.0 (ou très proche)
    
    # Test 2 : L'agent trouve une formule FAUSSE mais mathématiquement valide
    tokens_wrong = ['x0', '+', '1']
    res_wrong = evaluator.evaluate_episode(tokens_wrong, X_batch, y_true)
    print(f"Test 2 (Wrong equation 'x0+1'):\n{res_wrong}\n")
    # Attendu : is_valid=True, nmse > 0
    
    # Test 3 : L'agent écrit n'importe quoi (Erreur de syntaxe)
    tokens_syntax = ['+', '*', 'x0']
    res_syntax = evaluator.evaluate_episode(tokens_syntax, X_batch, y_true)
    print(f"Test 3 (Syntax Error '+*x0'):\n{res_syntax}\n")
    # Attendu : is_valid=False, nmse=1.0 (pénalité max)
    
    # Test 4 : L'agent fait une division par zéro (1 / (x0 - x0))
    tokens_math_error = ['1', '/', '(', 'x0', '-', 'x0', ')']
    res_math = evaluator.evaluate_episode(tokens_math_error, X_batch, y_true)
    print(f"Test 4 (Division by zero):\n{res_math}\n")
    # Attendu : is_valid=False (ou valid mais nmse=1.0 à cause de l'infini)

if __name__ == "__main__":
    run_tests()