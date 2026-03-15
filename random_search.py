import torch
import random
from config import ACTION_SPACE, CONFIG
from evaluator import Evaluator

def generate_random_equation(max_length):
    """
    Generates a random sequence of tokens from the action space.
    The length is randomly chosen between 1 and max_length.
    """
    # Randomly pick how long the equation will be
    length = random.randint(1, max_length)
    
    # Randomly choose tokens with replacement to build the equation
    tokens = random.choices(ACTION_SPACE, k=length)
    return tokens

def run_random_search(num_iterations=10000):
    """
    Runs the Random Search baseline to compare against the RL agent.
    """
    print(f"--- Starting Random Search Baseline ({num_iterations} iterations) ---\n")
    
    # 1. Setup the evaluator for two variables (x0, x1)
    evaluator = Evaluator(var_names=['x0', 'x1'])
    
    # 2. Create a synthetic dataset to test the equations
    # X_batch shape: (100 samples, 2 variables)
    X_batch = torch.rand((100, 2)) * 10.0  # Random values between 0 and 10
    
    # Ground truth target: y = x0 * sin(x1)
    # This is the secret formula the random search is trying to guess
    y_true = X_batch[:, 0] * torch.sin(X_batch[:, 1])
    
    best_nmse = 1.0  # Start with the maximum error
    best_equation = None
    valid_count = 0
    
    # 3. Random search loop
    for i in range(num_iterations):
        # Generate a random equation using the config parameter
        tokens = generate_random_equation(CONFIG["max_length"])
        
        # Evaluate it
        result = evaluator.evaluate_episode(tokens, X_batch, y_true)
        
        # If the equation is mathematically valid (no syntax error)
        if result["is_valid"]:
            valid_count += 1
            current_nmse = result["nmse"].item()
            
            # Save it if it's the best one we have seen so far
            if current_nmse < best_nmse or best_equation is None:
                best_nmse = current_nmse
                best_equation = tokens
                
        # Print progress every 2000 iterations to show it's working
        if (i + 1) % 2000 == 0:
            print(f"Iteration {i+1}/{num_iterations} | Best NMSE: {best_nmse:.4f} | Valid equations so far: {valid_count}")
            
    # 4. Final Results Output
    print("\n--- Final Results ---")
    if best_equation:
        print(f"Best Equation Found: {''.join(best_equation)}")
        print(f"Best NMSE: {best_nmse:.6f}")
    else:
        print("No valid equation found.")
        
    print(f"Total Valid Equations: {valid_count} / {num_iterations}")
    
    # Calculate the percentage of mathematically valid random equations
    valid_percentage = (valid_count / num_iterations) * 100
    print(f"Validity Rate: {valid_percentage:.2f}%")

if __name__ == "__main__":
    # We run 10,000 episodes to match the RL agent's training config
    run_random_search(num_iterations=CONFIG["num_episodes"])