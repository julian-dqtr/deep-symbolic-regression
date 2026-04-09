import os
import torch
import numpy as np

from ..core.config import TRAINING_CONFIG
from ..core.factory import build_grammar
from ..core.env import SymbolicRegressionEnv
from ..core.expression import safe_prefix_to_infix
from ..core.evaluator import PrefixEvaluator
from ..models.policy import SymbolicPolicy
from ..analysis.memory import TopKMemory, DiverseTopKMemory, PrioritizedTopKMemory
from .rollout import collect_episode, recompute_episode, collect_batched_episodes
from .policy_optimizer import ReinforceOptimizer
from .ppo_optimizer import PPOOptimizer
from .risk_seeking_optimizer import RiskSeekingOptimizer
from ..core.mdl_reward import mdl_reward_from_mse, MdlRewardConfig


def normalize_device(device: str) -> str:
    device = device.lower().strip()
    if device == "gpu":
        device = "cuda"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device!r}. Use 'cpu' or 'cuda'.")
    return device


# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------

def curriculum_max_length(
    episode_idx: int,
    num_episodes: int,
    max_length: int,
    start_length: int = 5,
    warmup_fraction: float = 0.5,
) -> int:
    """
    Linearly ramp max expression length from `start_length` to `max_length`
    over the first `warmup_fraction` of training, then keep it constant.

    This curriculum prevents the agent from wasting early episodes on
    oversized expressions it cannot yet evaluate meaningfully.
    """
    warmup_episodes = int(num_episodes * warmup_fraction)
    if warmup_episodes <= 0 or episode_idx >= warmup_episodes:
        return max_length
    progress = episode_idx / warmup_episodes
    current  = start_length + progress * (max_length - start_length)
    return max(start_length, min(max_length, int(current)))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        X,
        y,
        num_variables: int,
        device: str = "cpu",
        optimizer_name: str = "reinforce",
        # Curriculum learning
        use_curriculum: bool = False,
        curriculum_start_length: int = 5,
        curriculum_warmup_fraction: float = 0.5,
        # Diverse memory
        use_diverse_memory: bool = False,
        diverse_memory_min_edit_distance: int = 3,
        # MDL reward
        use_mdl_reward: bool = False,
        mdl_bits_per_constant: float = 32.0,
        # Prioritized replay
        use_prioritized_memory: bool = False,
        prioritized_memory_alpha: float = 1.0,
        prioritized_memory_max_replay: int = 20,
    ):
        self.device = normalize_device(device)

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

        self.grammar   = build_grammar(num_variables=num_variables)
        self.env       = SymbolicRegressionEnv(self.X, self.y, self.grammar)
        self.evaluator = PrefixEvaluator(self.grammar)

        self.policy = SymbolicPolicy(vocab_size=len(self.grammar)).to(self.device)

        self.tensor_X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        self.tensor_y = torch.tensor(self.y, dtype=torch.float32, device=self.device)
        self.policy.set_dataset_embedding(self.tensor_X, self.tensor_y)

        if optimizer_name.lower() == "ppo":
            self.rl_optimizer = PPOOptimizer(self.policy)
        elif optimizer_name.lower() == "rspg":
            self.rl_optimizer = RiskSeekingOptimizer(self.policy)
        else:
            self.rl_optimizer = ReinforceOptimizer(self.policy)

        self.optimizer_name  = optimizer_name.lower()
        self.num_episodes    = TRAINING_CONFIG["num_episodes"]
        self.batch_size      = TRAINING_CONFIG["batch_size"]
        self.max_length      = 30

        # Expose lr / entropy_weight as trainer-level attributes so callers
        # that override them (e.g. compare_optimizers.py) can do so uniformly.
        self.learning_rate   = TRAINING_CONFIG["learning_rate"]
        self.entropy_weight  = TRAINING_CONFIG["entropy_weight"]

        # Curriculum parameters
        self.use_curriculum             = use_curriculum
        self.curriculum_start_length    = curriculum_start_length
        self.curriculum_warmup_fraction = curriculum_warmup_fraction

        # Memory — three options, in priority order:
        #   1. Diverse + Prioritized  (both flags True → use PrioritizedTopKMemory)
        #   2. Diverse only
        #   3. Prioritized only
        #   4. Standard TopKMemory   (default)
        if use_diverse_memory and use_prioritized_memory:
            self.memory = PrioritizedTopKMemory(
                capacity=20,
                alpha=prioritized_memory_alpha,
                max_replay=prioritized_memory_max_replay,
            )
        elif use_prioritized_memory:
            self.memory = PrioritizedTopKMemory(
                capacity=20,
                alpha=prioritized_memory_alpha,
                max_replay=prioritized_memory_max_replay,
            )
        elif use_diverse_memory:
            self.memory = DiverseTopKMemory(
                capacity=20,
                min_edit_distance=diverse_memory_min_edit_distance,
            )
        else:
            self.memory = TopKMemory(capacity=20)

        # MDL reward config
        self.use_mdl_reward = use_mdl_reward
        self.mdl_config     = (
            MdlRewardConfig(bits_per_constant=mdl_bits_per_constant)
            if use_mdl_reward else None
        )

        self.use_prioritized_memory = use_prioritized_memory

        self.history = {
            "loss":           [],
            "policy_loss":    [],
            "value_loss":     [],
            "entropy":        [],
            "final_reward":   [],
            "episode_length": [],
            "max_length":     [],
        }

        self.best_reward  = float("-inf")
        self.best_episode = None

    # -----------------------------------------------------------------------
    # Checkpoint API
    # -----------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """
        Save the complete trainer state so training can be resumed or
        beam search can be run on the exact same policy.

        Saved:
          - policy weights + architecture metadata
          - optimizer state (Adam moments)
          - training history
          - best episode found so far
          - num_variables (needed to rebuild grammar on load)
          - X, y (small — 100 samples by default)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Serialise best episode — tensors must be detached
        best_ep_serialised = None
        if self.best_episode is not None:
            best_ep_serialised = {
                "tokens":       self.best_episode["tokens"],
                "final_reward": self.best_episode["final_reward"],
            }

        # Count variable tokens to recover num_variables on load
        num_variables = len(
            [t for t in self.grammar.tokens if t.kind == "variable"]
        )

        torch.save(
            {
                # Policy
                "policy_state_dict":      self.policy.state_dict(),
                "policy_vocab_size":      self.policy.vocab_size,
                "policy_hidden_dim":      self.policy.hidden_dim,
                "policy_num_lstm_layers": self.policy.num_lstm_layers,
                "policy_token_emb_dim":   self.policy.token_embedding_dim,
                "policy_dataset_emb_dim": self.policy.dataset_embedding_dim,
                # Cached dataset embedding (used by beam search / MCTS)
                "cached_dataset_embedding": (
                    self.policy.cached_dataset_embedding.cpu()
                    if self.policy.cached_dataset_embedding is not None
                    else None
                ),
                # Optimizer
                "optimizer_state_dict": self.rl_optimizer.optimizer.state_dict(),
                "optimizer_name":       self.optimizer_name,
                # Training state
                "num_variables": num_variables,
                "best_reward":   self.best_reward,
                "best_episode":  best_ep_serialised,
                "history":       self.history,
                # Data (small — 100 samples)
                "X": self.X,
                "y": self.y,
            },
            path,
        )
        print(f"Checkpoint saved → {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "Trainer":
        """
        Restore a Trainer from a checkpoint file.
        The returned Trainer has the policy in eval mode, ready for beam search.
        """
        state  = torch.load(path, map_location=device, weights_only=False)
        device = normalize_device(device)

        X             = state["X"]
        y             = state["y"]
        num_variables = state["num_variables"]

        trainer = cls(
            X=X, y=y,
            num_variables=num_variables,
            device=device,
            optimizer_name=state.get("optimizer_name", "rspg"),
        )

        trainer.policy.load_state_dict(state["policy_state_dict"])

        if state.get("cached_dataset_embedding") is not None:
            trainer.policy.cached_dataset_embedding = (
                state["cached_dataset_embedding"].to(device)
            )

        trainer.rl_optimizer.optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )

        trainer.best_reward  = state.get("best_reward", float("-inf"))
        trainer.best_episode = state.get("best_episode", None)
        trainer.history      = state.get("history", trainer.history)

        trainer.policy.eval()
        print(f"Checkpoint loaded ← {path}  (best_reward={trainer.best_reward:.4f})")
        return trainer

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def train(
        self,
        checkpoint_dir: str = None,
        checkpoint_every: int = 0,
        checkpoint_name: str = "policy_final",
    ):
        """
        Train the policy.

        Parameters
        ----------
        checkpoint_dir   : if set, save checkpoints to this directory.
        checkpoint_every : save a checkpoint every N episodes (0 = only at end).
        checkpoint_name  : base filename for the final checkpoint (no .pt extension).
                           Use task name to avoid overwriting between tasks.
        """
        episode_idx = 0

        while episode_idx < self.num_episodes:
            current_batch_size = min(self.batch_size, self.num_episodes - episode_idx)

            if self.use_curriculum:
                current_max_length = curriculum_max_length(
                    episode_idx=episode_idx,
                    num_episodes=self.num_episodes,
                    max_length=self.max_length,
                    start_length=self.curriculum_start_length,
                    warmup_fraction=self.curriculum_warmup_fraction,
                )
            else:
                current_max_length = self.max_length

            # ------------------------------------------------------------------
            # 1. Collect batch of episodes (rewards filled below)
            # ------------------------------------------------------------------
            batch_episodes = collect_batched_episodes(
                env_template=self.env,
                policy=self.policy,
                grammar=self.grammar,
                batch_size=current_batch_size,
                max_length=current_max_length,
                device=self.device,
            )

            # ------------------------------------------------------------------
            # 2. Evaluate each episode and assign rewards
            # ------------------------------------------------------------------
            for episode in batch_episodes:
                eval_result = self.evaluator.evaluate(
                    tokens=episode["tokens"], X=self.X, y=self.y
                )

                if eval_result["is_valid"]:
                    if self.use_mdl_reward:
                        var_y  = float(np.var(self.y)) if np.var(self.y) > 1e-9 else 1.0
                        mse    = eval_result["nmse"] * var_y
                        reward = mdl_reward_from_mse(
                            tokens=episode["tokens"],
                            mse=mse,
                            n=len(self.y),
                            vocab_size=len(self.grammar),
                            config=self.mdl_config,
                        )
                    else:
                        reward = -eval_result["nmse"] - 0.01 * len(episode["tokens"])
                else:
                    reward = -1.0

                L = len(episode["tokens"])
                episode["rewards"]      = [0.0] * L
                if L > 0:
                    episode["rewards"][-1] = reward
                episode["final_reward"] = reward

                infix = safe_prefix_to_infix(
                    episode["tokens"], self.grammar,
                    eval_result.get("optimized_constants", []),
                )

                self.memory.add(
                    tokens=episode["tokens"],
                    infix=infix,
                    reward=reward,
                    nmse=eval_result["nmse"],
                    complexity=L,
                    source="sampling",
                )

                if reward > self.best_reward:
                    self.best_reward  = reward
                    self.best_episode = episode
                    print(
                        f"  ★ New best (ep={episode_idx}, "
                        f"max_len={current_max_length}): "
                        f"{episode['tokens']}"
                    )

                episode_idx += 1

            # ------------------------------------------------------------------
            # 3. Memory replay — teacher-force historical best expressions
            # ------------------------------------------------------------------
            memory_episodes = []
            for item in self.memory.to_rows():
                try:
                    ep = recompute_episode(
                        env=self.env,
                        policy=self.policy,
                        grammar=self.grammar,
                        tokens=item["tokens"],
                        device=self.device,
                    )
                    memory_episodes.append(ep)
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # 4. Policy update
            # ------------------------------------------------------------------
            if self.optimizer_name == "rspg":
                stats = self.rl_optimizer.update(
                    batch_episodes, memory_episodes=memory_episodes
                )
            else:
                stats = self.rl_optimizer.update(batch_episodes)

            self.history["loss"].append(stats["loss"])
            self.history["policy_loss"].append(stats["policy_loss"])
            self.history["value_loss"].append(stats.get("value_loss", 0.0))
            self.history["entropy"].append(stats["entropy"])
            self.history["final_reward"].append(stats["final_reward"])
            self.history["max_length"].append(current_max_length)

            # ------------------------------------------------------------------
            # 5. Logging
            # ------------------------------------------------------------------
            if episode_idx % max(100, self.batch_size) == 0 or episode_idx == current_batch_size:
                mem_info = f"mem={len(self.memory)}"
                if self.use_prioritized_memory and hasattr(self.memory, "priority_stats"):
                    ps = self.memory.priority_stats()
                    if ps:
                        mem_info += (
                            f" | prio_max={ps['max_priority']:.3f}"
                            f" base={ps['baseline']:.3f}"
                        )

                reward_mode = "mdl" if self.use_mdl_reward else "lin"
                print(
                    f"[Ep {episode_idx:>6}/{self.num_episodes}] "
                    f"opt={self.optimizer_name} | "
                    f"{reward_mode} | "
                    f"max_len={current_max_length} | "
                    f"reward={stats['final_reward']:.4f} | "
                    f"best={self.best_reward:.4f} | "
                    f"entropy={stats['entropy']:.4f} | "
                    f"{mem_info}"
                )

            # ------------------------------------------------------------------
            # 6. Periodic checkpoint
            # ------------------------------------------------------------------
            if (
                checkpoint_dir
                and checkpoint_every > 0
                and episode_idx % checkpoint_every == 0
            ):
                path = os.path.join(checkpoint_dir, f"policy_ep{episode_idx}.pt")
                self.save_checkpoint(path)

        # Final checkpoint
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.save_checkpoint(
                os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
            )

        return {
            "history":      self.history,
            "best_reward":  self.best_reward,
            "best_episode": self.best_episode,
            "memory":       self.memory,
        }