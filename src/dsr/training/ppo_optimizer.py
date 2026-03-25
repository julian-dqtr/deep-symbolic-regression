import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ..core.config import TRAINING_CONFIG


class PPOOptimizer:
    def __init__(self, policy, clip_epsilon=0.2, ppo_epochs=4, value_weight=0.5):
        self.policy = policy

        self.learning_rate = TRAINING_CONFIG["learning_rate"]
        self.entropy_weight = TRAINING_CONFIG["entropy_weight"]
        self.grad_clip_norm = TRAINING_CONFIG["grad_clip_norm"]

        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.value_weight = value_weight

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def compute_returns(self, rewards):
        final_reward = rewards[-1] if len(rewards) > 0 else 0.0
        returns = [final_reward for _ in rewards]
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, batch_episodes):
        valid_episodes = [ep for ep in batch_episodes if len(ep["log_probs"]) > 0]
        if not valid_episodes:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "final_reward": float(np.mean([ep["final_reward"] for ep in batch_episodes])),
            }

        device = valid_episodes[0]["log_probs"][0].device

        all_old_log_probs = []
        all_old_values = []
        all_action_ids = []
        all_returns = []
        all_observations = []

        for episode in valid_episodes:
            old_log_probs = torch.stack(episode["log_probs"]).detach()
            old_values = torch.stack(episode["values"]).detach()
            action_ids = torch.tensor(episode["action_ids"], dtype=torch.long, device=device)
            returns = self.compute_returns(episode["rewards"]).to(device)

            all_old_log_probs.append(old_log_probs)
            all_old_values.append(old_values)
            all_action_ids.append(action_ids)
            all_returns.append(returns)
            all_observations.extend(episode["observations"])

        old_log_probs_tensor = torch.cat(all_old_log_probs)
        old_values_tensor = torch.cat(all_old_values)
        action_ids_tensor = torch.cat(all_action_ids)
        returns_tensor = torch.cat(all_returns)

        advantages_tensor = returns_tensor - old_values_tensor
        advantages_tensor = advantages_tensor.detach()

        last_loss = None
        last_policy_loss = None
        last_value_loss = None
        last_entropy = None

        for _ in range(self.ppo_epochs):
            new_log_probs_list = []
            entropies_list = []
            values_list = []

            for obs, action_id in zip(all_observations, action_ids_tensor):
                logits, value = self.policy(
                    token_ids=obs["token_ids"],
                    pending_slots=obs["pending_slots"],
                    length=obs["length"],
                    action_mask=obs["action_mask"],
                )

                dist = Categorical(logits=logits)
                action = torch.tensor(action_id.item(), dtype=torch.long, device=device)

                new_log_probs_list.append(dist.log_prob(action))
                entropies_list.append(dist.entropy())
                values_list.append(value)

            new_log_probs = torch.stack(new_log_probs_list)
            entropies = torch.stack(entropies_list)
            new_values = torch.stack(values_list)

            ratios = torch.exp(new_log_probs - old_log_probs_tensor)
            clipped_ratios = torch.clamp(
                ratios,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            )

            surrogate_1 = ratios * advantages_tensor
            surrogate_2 = clipped_ratios * advantages_tensor

            policy_loss = -torch.min(surrogate_1, surrogate_2).sum() / len(valid_episodes)
            value_loss = nn.functional.mse_loss(new_values, returns_tensor, reduction="sum") / len(valid_episodes)
            entropy_bonus = entropies.sum() / len(valid_episodes)

            loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            last_loss = loss
            last_policy_loss = policy_loss
            last_value_loss = value_loss
            last_entropy = entropy_bonus

        return {
            "loss": float(last_loss.item()),
            "policy_loss": float(last_policy_loss.item()),
            "value_loss": float(last_value_loss.item()),
            "entropy": float(last_entropy.item()),
            "final_reward": float(np.mean([ep["final_reward"] for ep in batch_episodes])),
        }