import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..core.config import TRAINING_CONFIG


class RiskSeekingOptimizer:
    def __init__(self, policy, epsilon=0.05):
        self.policy  = policy
        self.epsilon = epsilon

        self.learning_rate  = TRAINING_CONFIG["learning_rate"]
        self.entropy_weight = TRAINING_CONFIG["entropy_weight"]
        self.grad_clip_norm = TRAINING_CONFIG["grad_clip_norm"]

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def compute_returns(self, rewards):
        final_reward = rewards[-1] if len(rewards) > 0 else 0.0
        returns = [final_reward for _ in rewards]
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, batch_episodes, memory_episodes=None):
        # Cast to float64 to avoid overflow in np.quantile / np.mean
        # when rewards contain very large negative values (e.g. -1.0 * batch_size)
        rewards   = np.array([ep["final_reward"] for ep in batch_episodes],
                             dtype=np.float64)
        rewards   = np.clip(rewards, -1e6, 1e6)   # guard against ±inf
        threshold = np.quantile(rewards, 1.0 - self.epsilon)

        elite_episodes = [ep for ep in batch_episodes
                          if ep["final_reward"] >= threshold]
        if not elite_episodes:
            elite_episodes = batch_episodes

        # Inject historical best equations from memory
        if memory_episodes:
            elite_episodes = elite_episodes + memory_episodes  # avoid mutating list

        all_log_probs = []
        all_entropies = []
        all_returns   = []

        for episode in elite_episodes:
            if len(episode["log_probs"]) == 0:
                continue
            device = episode["log_probs"][0].device

            log_probs = torch.stack(episode["log_probs"])
            entropies = torch.stack(episode["entropies"])
            returns   = self.compute_returns(episode["rewards"]).to(device)

            all_log_probs.append(log_probs)
            all_entropies.append(entropies)
            all_returns.append(returns)

        if not all_log_probs:
            return {
                "loss":         0.0,
                "policy_loss":  0.0,
                "entropy":      0.0,
                "final_reward": float(rewards.mean()),
            }

        log_probs = torch.cat(all_log_probs)
        entropies = torch.cat(all_entropies)
        returns   = torch.cat(all_returns)

        # Number of elite episodes actually used (batch elites + memory replays).
        # FIX: normalise by this count so the loss magnitude is independent of:
        #   - the batch size
        #   - the number of memory episodes injected (varies across updates)
        #   - the expression length (longer expressions accumulate more log-probs)
        # Without normalisation, a batch of 256 episodes produces a loss ~256×
        # larger than a batch of 32, making lr and entropy_weight non-portable
        # across different batch sizes and memory sizes.
        n_elite = len([ep for ep in elite_episodes
                        if len(ep["log_probs"]) > 0])
        n_elite = max(n_elite, 1)  # safety guard

        # Policy loss: maximise log-prob of elite trajectories
        # Normalised per episode so gradients are batch-size invariant
        policy_loss   = -log_probs.sum() / n_elite

        # Entropy bonus: encourage exploration
        # Also normalised per episode for the same reason
        entropy_bonus = entropies.sum() / n_elite

        loss = policy_loss - self.entropy_weight * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss":         float(loss.item()),
            "policy_loss":  float(policy_loss.item()),
            "entropy":      float(entropy_bonus.item()),
            "final_reward": float(rewards.mean()),  # mean of full batch
            "n_elite":      n_elite,                  # useful for debugging
        }