import torch
from torch.distributions import Categorical


def tokens_to_ids(tokens, grammar):
    return [grammar.token_to_id[tok] for tok in tokens]


def collect_episode(env, policy, grammar, device="cpu"):
    obs = env.reset()

    trajectory = {
        "tokens": [],
        "action_ids": [],
        "log_probs": [],
        "entropies": [],
        "values": [],
        "rewards": [],
        "observations": [],
        "final_reward": 0.0,
    }

    done = False

    while not done:
        token_ids_list = tokens_to_ids(obs["tokens"], grammar)

        token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)
        action_mask = torch.tensor(env.valid_action_mask(), dtype=torch.float32, device=device)

        logits, value = policy(
            token_ids=token_ids,
            pending_slots=obs["pending_slots"],
            length=obs["length"],
            action_mask=action_mask,
        )

        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        trajectory["observations"].append(
            {
                "token_ids": token_ids.detach().clone(),
                "pending_slots": obs["pending_slots"],
                "length": obs["length"],
                "action_mask": action_mask.detach().clone(),
            }
        )

        step_out = env.step(action.item())
        obs = step_out.observation
        done = step_out.done

        chosen_token = grammar.id_to_token[action.item()]

        trajectory["tokens"].append(chosen_token)
        trajectory["action_ids"].append(action.item())
        trajectory["log_probs"].append(log_prob)
        trajectory["entropies"].append(entropy)
        trajectory["values"].append(value)
        trajectory["rewards"].append(step_out.reward)

    trajectory["final_reward"] = trajectory["rewards"][-1] if trajectory["rewards"] else 0.0
    return trajectory


def recompute_episode(env, policy, grammar, tokens, device="cpu"):
    """
    Teacher forces the policy network through a predefined sequence of tokens to recompute
    gradients, log probabilities, and entropies for historical expressions.
    """
    obs = env.reset()

    trajectory = {
        "tokens": [],
        "action_ids": [],
        "log_probs": [],
        "entropies": [],
        "values": [],
        "rewards": [],
        "observations": [],
        "final_reward": 0.0,
    }

    for token in tokens:
        action_id = grammar.token_to_id[token]

        token_ids_list = tokens_to_ids(obs["tokens"], grammar)
        token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)
        action_mask = torch.tensor(env.valid_action_mask(), dtype=torch.float32, device=device)

        logits, value = policy(
            token_ids=token_ids,
            pending_slots=obs["pending_slots"],
            length=obs["length"],
            action_mask=action_mask,
        )

        dist = Categorical(logits=logits)
        
        # Use the predefined action instead of sampling
        action = torch.tensor(action_id, device=device)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        trajectory["observations"].append(
            {
                "token_ids": token_ids.detach().clone(),
                "pending_slots": obs["pending_slots"],
                "length": obs["length"],
                "action_mask": action_mask.detach().clone(),
            }
        )

        step_out = env.step(action_id)
        obs = step_out.observation

        trajectory["tokens"].append(token)
        trajectory["action_ids"].append(action_id)
        trajectory["log_probs"].append(log_prob)
        trajectory["entropies"].append(entropy)
        trajectory["values"].append(value)
        trajectory["rewards"].append(step_out.reward)

    trajectory["final_reward"] = trajectory["rewards"][-1] if trajectory["rewards"] else 0.0
    return trajectory