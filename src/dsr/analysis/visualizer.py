import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..core.expression import prefix_to_tree, safe_prefix_to_infix


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


class ASTVisualizer:
    def __init__(self):
        self.colors = {
            "binary": "#CB4B16",    # orange
            "unary": "#859900",     # green
            "variable": "#268BD2",  # blue
            "constant": "#6C71C4",  # violet
            "special": "#93A1A1",   # gray
            "default": "#93A1A1",
        }
        self.bg_color = "#FDF6E3"
        self.edge_color = "#657B83"

    def _node_kind(self, token: str, grammar) -> str:
        return grammar.kind.get(token, "default")

    def _add_tree_to_graph(self, node, grammar, graph, parent_id=None, node_counter=None):
        if node_counter is None:
            node_counter = {"value": 0}

        node_id = node_counter["value"]
        node_counter["value"] += 1

        kind = self._node_kind(node.token, grammar)
        color = self.colors.get(kind, self.colors["default"])

        graph.add_node(
            node_id,
            label=node.token,
            color=color,
            kind=kind,
        )

        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        for child in node.children:
            self._add_tree_to_graph(
                child,
                grammar,
                graph,
                parent_id=node_id,
                node_counter=node_counter,
            )

    def draw_tree(self, tokens, grammar, filename=None, title=None, show=True):
        try:
            tree = prefix_to_tree(tokens, grammar)
        except Exception:
            print("Could not parse prefix expression into a tree.")
            return False

        G = nx.DiGraph()
        self._add_tree_to_graph(tree, grammar, G)

        plt.figure(figsize=(12, 8))
        plt.gcf().set_facecolor(self.bg_color)

        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        node_colors = [G.nodes[n]["color"] for n in G.nodes]
        labels = {n: G.nodes[n]["label"] for n in G.nodes}

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=3000,
            edgecolors="black",
            linewidths=1.5,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=self.edge_color,
            width=2.0,
            arrows=True,
            arrowsize=20,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=12,
            font_weight="bold",
        )

        final_title = title if title is not None else safe_prefix_to_infix(tokens, grammar)
        plt.title(final_title, fontsize=16, pad=20, color="black")
        plt.axis("off")
        plt.tight_layout()

        if filename is not None:
            ensure_dir(os.path.dirname(filename))
            plt.savefig(filename, bbox_inches="tight", facecolor=self.bg_color)

        if show:
            plt.show()
        else:
            plt.close()

        return True


def plot_training_history(history, title="Training History", show=True, save_path=None):
    rewards = history.get("final_reward", [])
    losses = history.get("loss", [])
    entropies = history.get("entropy", [])
    lengths = history.get("episode_length", [])

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

    axes[0].plot(rewards)
    axes[0].set_ylabel("Final Reward")
    axes[0].set_title(title)

    axes[1].plot(losses)
    axes[1].set_ylabel("Loss")

    axes[2].plot(entropies)
    axes[2].set_ylabel("Entropy")

    axes[3].plot(lengths)
    axes[3].set_ylabel("Ep. Length")
    axes[3].set_xlabel("Episode")

    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_target_vs_prediction(grammar, evaluator, tokens, X, y, title=None, show=True, save_path=None):
    try:
        y_pred = evaluator._eval_prefix(tokens, X)
    except Exception:
        print("Could not evaluate expression for plotting.")
        return False

    X_plot = np.asarray(X[:, 0])
    y_plot = np.asarray(y)
    y_pred_plot = np.asarray(y_pred)

    order = np.argsort(X_plot)
    X_plot = X_plot[order]
    y_plot = y_plot[order]
    y_pred_plot = y_pred_plot[order]

    plt.figure(figsize=(8, 5))
    plt.plot(X_plot, y_plot, label="Target")
    plt.plot(X_plot, y_pred_plot, label="Prediction")
    plt.xlabel("x0")
    plt.ylabel("y")

    final_title = title if title is not None else safe_prefix_to_infix(tokens, grammar)
    plt.title(final_title)

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return True


def plot_method_comparison(X, y, predictions_dict, title, save_path=None, show=True):
    """
    predictions_dict = {
        "Random": y_pred_random,
        "PPO": y_pred_ppo,
        "Beam": y_pred_beam,
    }
    """
    X_plot = np.asarray(X[:, 0])
    y_plot = np.asarray(y)

    order = np.argsort(X_plot)
    X_plot = X_plot[order]
    y_plot = y_plot[order]

    plt.figure(figsize=(8, 5))
    plt.plot(X_plot, y_plot, label="Target", linewidth=2)

    for method_name, y_pred in predictions_dict.items():
        y_pred_plot = np.asarray(y_pred)[order]
        plt.plot(X_plot, y_pred_plot, label=method_name, linestyle="--")

    plt.xlabel("x0")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def save_summary_barplot(rows, save_path=None, show=True):
    tasks = [row["task"] for row in rows]
    random_rewards = [row["random_reward"] for row in rows]
    ppo_rewards = [row["ppo_reward"] for row in rows]
    beam_rewards = [row["beam_reward"] for row in rows]

    x = np.arange(len(tasks))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, random_rewards, width, label="Random Search")
    plt.bar(x, ppo_rewards, width, label="PPO Sampling")
    plt.bar(x + width, beam_rewards, width, label="PPO + Beam Reranked")

    plt.xticks(x, tasks)
    plt.ylabel("Best Reward")
    plt.title("Best Reward by Method and Task")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def print_top_candidates(candidates):
    if not candidates:
        print("No candidates to display.")
        return

    for rank, cand in enumerate(candidates, start=1):
        print(f"Rank {rank}")
        print(f"  Reward: {cand.get('reward')}")
        print(f"  NMSE: {cand.get('nmse')}")
        print(f"  Valid: {cand.get('is_valid')}")
        print(f"  Complexity: {cand.get('complexity')}")
        print(f"  Tokens: {cand.get('tokens')}")
        print(f"  Expression: {cand.get('infix')}")
        print("  " + "-" * 40)