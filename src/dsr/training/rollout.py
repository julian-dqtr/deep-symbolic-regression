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


def collect_batched_episodes(env_template, policy, grammar, batch_size, max_length=30, device="cpu"):
    """
    Generates a full batch of episodes simultaneously using PyTorch parallel tensor operations.
    Bypasses the slow Python sequential environment step-by-step.
    """
    B = batch_size
    vocab_size = len(grammar)
    
    pending_slots = torch.ones((B, 1), dtype=torch.long, device=device)
    lengths = torch.ones((B, 1), dtype=torch.long, device=device)
    
    tokens_batch = [[] for _ in range(B)]
    action_ids_batch = [[] for _ in range(B)]
    log_probs_batch = [[] for _ in range(B)]
    entropies_batch = [[] for _ in range(B)]
    values_batch = [[] for _ in range(B)]
    observations_batch = [[] for _ in range(B)]
    
    done = torch.zeros(B, dtype=torch.bool, device=device)
    arities = torch.tensor([grammar.arity[grammar.id_to_token[i]] for i in range(vocab_size)], device=device)
    
    current_sequences = torch.empty((B, 0), dtype=torch.long, device=device)
    
    while not done.all() and current_sequences.shape[1] < max_length:
        remaining = max_length - lengths
        new_pending = pending_slots - 1 + arities.unsqueeze(0)
        action_mask = (new_pending <= remaining).float()
        
        logits, value = policy(
            token_ids=current_sequences,
            pending_slots=pending_slots,
            length=lengths,
            action_mask=action_mask
        )
        
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        
        active = ~done
        
        for i in range(B):
            if active[i]:
                action_idx = actions[i].item()
                tokens_batch[i].append(grammar.id_to_token[action_idx])
                action_ids_batch[i].append(action_idx)
                log_probs_batch[i].append(log_probs[i])
                entropies_batch[i].append(entropies[i])
                values_batch[i].append(value[i])
                
                observations_batch[i].append({
                    "token_ids": current_sequences[i].clone() if current_sequences.shape[1] > 0 else torch.empty(0, dtype=torch.long, device=device),
                    "pending_slots": pending_slots[i, 0].item(),
                    "length": lengths[i, 0].item(),
                    "action_mask": action_mask[i].clone()
                })
                
                pending_slots[i, 0] = pending_slots[i, 0] - 1 + grammar.arity[grammar.id_to_token[action_idx]]
                lengths[i, 0] += 1
                
                if pending_slots[i, 0] == 0:
                    done[i] = True
                    
        current_sequences = torch.cat([current_sequences, actions.unsqueeze(1)], dim=1)

    trajectories = []
    for i in range(B):
        traj = {
            "tokens": tokens_batch[i],
            "action_ids": action_ids_batch[i],
            "log_probs": log_probs_batch[i],
            "entropies": entropies_batch[i],
            "values": values_batch[i],
            "observations": observations_batch[i],
            "rewards": [], 
            "final_reward": 0.0,
        }
        trajectories.append(traj)
        
    return trajectories