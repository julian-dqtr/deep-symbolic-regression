import numpy as np
import torch
from environment import SymbolicEnv

# 1. Création d'un dataset : y = sin(x)
x_train = np.linspace(-3, 3, 100).reshape(-1, 1)
y_train = np.sin(x_train).flatten()

# 2. Initialisation
env = SymbolicEnv(x_train, y_train)
expr, embedding = env.reset()
print(f"Dataset embedding shape: {embedding.shape}")

# 3. Test d'une équation : on prend les indices existants
try:
    # On cherche 'sin' et 'x0' (ou 'x' selon le config de Julian)
    # Si 'sin' n'est pas là, on prend l'indice 0 par défaut
    idx_func = env.action_space.index("sin") if "sin" in env.action_space else 0
    idx_var = env.action_space.index("x0") if "x0" in env.action_space else 0

    print(f"Test avec les jetons : {env.action_space[idx_func]} et {env.action_space[idx_var]}")

    env.step(idx_func)
    _, reward, done = env.step(idx_var) # On s'arrête ici (longueur max ou auto-fin)

    print(f"Expression : {env.current_expression}")
    print(f"Récompense obtenue : {reward}")
    print("--- TEST RÉUSSI ---")

except Exception as e:
    print(f"Erreur lors du test : {e}")