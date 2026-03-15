import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp
import uuid
import re

def tokens_to_latex_title(tokens):
    """
    Converts a list of tokens into a clean LaTeX math string for the title.
    """
    # Join tokens into a single string
    expr_str = "".join(tokens)
    
    # 1. Clean multiplication (use space for implicit math notation)
    latex_str = expr_str.replace('*', ' ')
    
    # 2. LaTeX symbols for constants
    latex_str = latex_str.replace('pi', r'\pi')
    
    # 3. Add backslashes to standard math functions for proper LaTeX rendering
    functions = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tan']
    for func in functions:
        latex_str = re.sub(r'\b' + func + r'\b', r'\\' + func, latex_str)
        
    # 4. Handle power operator
    latex_str = latex_str.replace('**', '^')

    return f"${latex_str}$"

class ASTVisualizer:
    def __init__(self):
        """
        Initializes the AST visualizer with the professional Solarized Light color palette.
        """
        self.colors = {
            "operator": "#CB4B16",   # Solarized Orange
            "function": "#859900",   # Solarized Green
            "variable": "#268BD2",   # Solarized Blue
            "constant": "#6C71C4",   # Solarized Violet
            "default":  "#93A1A1"    # Solarized Gray
        }
        self.bg_color = "#FDF6E3"    # Solarized Base3 (Background)
        self.edge_color = "#657B83"  # Solarized Base00

    def _get_node_category(self, label):
        """
        Categorizes tokens based on their clean text label string.
        """
        if label in ['pi', 'e', 'g', 'c', 'G', 'h', 'k'] or label.replace('.','',1).isdigit():
            return "constant"
        if re.match(r'^x\d+$', label):
            return "variable"
        if label in ['+', '*', '**', '-', '/', 'Add', 'Mul', 'Pow']:
            return "operator"
        if label in ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tan']:
            return "function"
        return "default"

    def _add_nodes_edges(self, expr, graph, parent_id=None):
        """
        Recursively builds the DiGraph for the tree.
        """
        node_id = str(uuid.uuid4())
        
        if expr.is_Atom:
            label = str(expr)
        else:
            label = type(expr).__name__
        
        mapping = {'Add': '+', 'Mul': '*', 'Pow': '**'}
        clean_label = mapping.get(label, label)
        
        category = self._get_node_category(clean_label)
        node_color = self.colors.get(category, self.colors["default"])

        graph.add_node(node_id, label=clean_label, color=node_color)
        
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        for arg in expr.args:
            self._add_nodes_edges(arg, graph, node_id)
        return node_id

    def draw_tree(self, tokens, filename="abstract_syntax_tree.png"):
        """
        Generates and saves a hierarchical AST with Solarized theme and LaTeX title.
        """
        expr_str = "".join(tokens)
        latex_formula = tokens_to_latex_title(tokens)
        
        try:
            # evaluate=False keeps the structure exactly as entered
            sympy_expr = sp.parse_expr(expr_str, evaluate=False)
        except Exception:
            print(f"Error: Could not parse expression {expr_str}")
            return False

        G = nx.DiGraph()
        self._add_nodes_edges(sympy_expr, G)

        plt.figure(figsize=(16, 12))
        plt.gcf().set_facecolor(self.bg_color)
        
        try:
            # Forces a strict hierarchical tree layout using Graphviz
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
        except Exception:
            print("Warning: Graphviz/pydot not found. Falling back to spring layout.")
            pos = nx.spring_layout(G, k=1.0)

        # Draw nodes with black outlines
        node_colors = [node[1]['color'] for node in G.nodes(data=True)]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5500, edgecolors='black', linewidths=2.0)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=self.edge_color, width=3.0, arrows=True, arrowsize=25)
        
        # Draw text labels inside nodes
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, font_weight='bold')
        
        # --- Clean Legend (No AST title, larger text) ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Operators', markerfacecolor=self.colors["operator"], markersize=14, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Functions', markerfacecolor=self.colors["function"], markersize=14, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Variables', markerfacecolor=self.colors["variable"], markersize=14, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Constants', markerfacecolor=self.colors["constant"], markersize=14, markeredgecolor='black'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=18, frameon=True, facecolor=self.bg_color, title_fontsize=20)

        # --- LaTeX Title in Pure Black ---
        plt.title(f"Abstract Syntax Tree : {latex_formula}", fontsize=28, pad=45, fontweight='bold', color='#000000')
        
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', facecolor=self.bg_color)
        plt.close()
        print(f"Professional AST saved as {filename}")
        return True

if __name__ == "__main__":
    viz = ASTVisualizer()
    # Complex test equation: x0 * sin(x1 + pi) + g + k
    test_tokens = ['x0', '*', 'sin', '(', 'x1', '+', 'pi', ')', '+', 'g', '+', 'k']
    viz.draw_tree(test_tokens)