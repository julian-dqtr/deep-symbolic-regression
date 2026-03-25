import os
import sys
import ast
import glob
import pandas as pd

# Allow absolute imports from project root independently of cwd
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.dsr.analysis.visualizer import ASTVisualizer
from src.dsr.core.factory import build_grammar

def ast_to_prefix(node):
    if isinstance(node, ast.Expression):
        return ast_to_prefix(node.body)
    elif isinstance(node, ast.BinOp):
        op = type(node.op)
        op_map = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
        ans = [op_map.get(op, str(op))]
        ans.extend(ast_to_prefix(node.left))
        ans.extend(ast_to_prefix(node.right))
        return ans
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        ans = [func_name]
        ans.extend(ast_to_prefix(node.args[0]))
        return ans
    elif isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Constant):
        return [str(node.value)]
    else:
        return [str(node)]

def main():
    # Load newest CSV
    search_pattern = os.path.join(script_dir, "results_*.csv")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        print("No results CSV found.")
        return
        
    csv_path = max(list_of_files, key=os.path.getctime)
    print(f"Reading {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Filter for good expressions (NMSE < 0.1)
    good_df = df[df["best_train_nmse"] < 0.1].copy()
    if len(good_df) == 0:
        print("No good expressions found!")
        return
        
    # Sort by complexity (string length as proxy) to find the most "beautiful/complex"
    good_df["complexity"] = good_df["best_train_expr"].astype(str).apply(len)
    good_df = good_df.sort_values(by="complexity", ascending=False)
    
    best_row = good_df.iloc[0]
    expr_str = best_row["best_train_expr"]
    task_name = best_row["task_name"]
    nmse = best_row["best_train_nmse"]
    
    print(f"Top Complex Equation:")
    print(f"Task: {task_name}")
    print(f"NMSE: {nmse}")
    print(f"Infix: {expr_str}")
    
    grammar = build_grammar(num_variables=10)
    
    try:
        tree = ast.parse(expr_str, mode='eval')
        tokens = ast_to_prefix(tree)
        print("Prefix tokens:", tokens)
    except Exception as e:
        print("Error parsing expression:", e)
        return
        
    vis = ASTVisualizer()
    vis.draw_tree(
        tokens, 
        grammar, 
        title=f"{task_name} (NMSE: {nmse:.1e})\n{expr_str}", 
        filename=os.path.join(script_dir, "best_equation.png"), 
        show=True
    )
    print("Saved visualization to results/best_equation.png")

if __name__ == "__main__":
    main()
