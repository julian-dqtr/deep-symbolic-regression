import torch
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokens_to_ids(tokens, grammar):
    return [grammar.token_to_id[tok] for tok in tokens]


# ---------------------------------------------------------------------------
# Sequential episode (uses real environment)
# ---------------------------------------------------------------------------

def collect_episode(env, policy, grammar, device="cpu"):
    """
    Collect one episode by stepping through the real SymbolicRegressionEnv.

    Rewards come directly from env.step(), so rewards[-1] is the true
    terminal reward (NMSE-based) and all intermediate rewards are 0.0.
    """
    obs = env.reset()

    trajectory = {
        "tokens":       [],
        "action_ids":   [],
        "log_probs":    [],
        "entropies":    [],
        "values":       [],
        "rewards":      [],
        "observations": [],
        "final_reward": 0.0,
    }

    done = False
    while not done:
        token_ids_list = tokens_to_ids(obs["tokens"], grammar)
        token_ids  = torch.tensor(token_ids_list, dtype=torch.long, device=device)
        action_mask = torch.tensor(
            env.valid_action_mask(), dtype=torch.float32, device=device
        )

        logits, value = policy(
            token_ids=token_ids,
            pending_slots=obs["pending_slots"],
            length=obs["length"],
            action_mask=action_mask,
        )

        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        trajectory["observations"].append({
            "token_ids":    token_ids.detach().clone(),
            "pending_slots": obs["pending_slots"],
            "length":        obs["length"],
            "action_mask":   action_mask.detach().clone(),
        })

        step_out = env.step(action.item())
        obs  = step_out.observation
        done = step_out.done

        trajectory["tokens"].append(grammar.id_to_token[action.item()])
        trajectory["action_ids"].append(action.item())
        trajectory["log_probs"].append(log_prob)
        trajectory["entropies"].append(entropy)
        trajectory["values"].append(value)
        trajectory["rewards"].append(step_out.reward)

    trajectory["final_reward"] = (
        trajectory["rewards"][-1] if trajectory["rewards"] else 0.0
    )
    return trajectory


# ---------------------------------------------------------------------------
# Teacher-forcing replay (uses real environment for reward)
# ---------------------------------------------------------------------------

def recompute_episode(env, policy, grammar, tokens, device="cpu"):
    """
    Teacher-force the policy through a predefined token sequence.

    Used for memory replay: we need differentiable log_probs for a historical
    expression. The environment is still stepped so that rewards[] reflects
    the true terminal reward for this expression on the current dataset.

    Parameters
    ----------
    tokens : List[str]
        The exact prefix token sequence to replay.
    """
    obs = env.reset()

    trajectory = {
        "tokens":       [],
        "action_ids":   [],
        "log_probs":    [],
        "entropies":    [],
        "values":       [],
        "rewards":      [],
        "observations": [],
        "final_reward": 0.0,
    }

    for token in tokens:
        action_id = grammar.token_to_id[token]

        token_ids_list = tokens_to_ids(obs["tokens"], grammar)
        token_ids   = torch.tensor(token_ids_list, dtype=torch.long, device=device)
        action_mask = torch.tensor(
            env.valid_action_mask(), dtype=torch.float32, device=device
        )

        logits, value = policy(
            token_ids=token_ids,
            pending_slots=obs["pending_slots"],
            length=obs["length"],
            action_mask=action_mask,
        )

        dist     = Categorical(logits=logits)
        action   = torch.tensor(action_id, device=device)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        trajectory["observations"].append({
            "token_ids":    token_ids.detach().clone(),
            "pending_slots": obs["pending_slots"],
            "length":        obs["length"],
            "action_mask":   action_mask.detach().clone(),
        })

        step_out = env.step(action_id)
        obs = step_out.observation

        trajectory["tokens"].append(token)
        trajectory["action_ids"].append(action_id)
        trajectory["log_probs"].append(log_prob)
        trajectory["entropies"].append(entropy)
        trajectory["values"].append(value)
        trajectory["rewards"].append(step_out.reward)

    trajectory["final_reward"] = (
        trajectory["rewards"][-1] if trajectory["rewards"] else 0.0
    )
    return trajectory


# ---------------------------------------------------------------------------
# Vectorised batch collection (fast — no env.step calls)
# ---------------------------------------------------------------------------

def collect_batched_episodes(
    env_template, policy, grammar, batch_size, max_length=30, device="cpu"
):
    """
    Generate a full batch of episodes in parallel using PyTorch tensor ops.

    Avoids the Python overhead of stepping through SymbolicRegressionEnv
    one episode at a time. Roughly 10× faster than a sequential loop for
    batch_size >= 64.

    IMPORTANT — reward contract
    ---------------------------
    This function does NOT evaluate expressions. The returned trajectories
    have ``rewards = []`` and ``final_reward = 0.0``. The caller (Trainer)
    is responsible for evaluating each episode's token sequence and filling:

        episode["rewards"]      = [0.0] * len(tokens)
        episode["rewards"][-1]  = computed_reward
        episode["final_reward"] = computed_reward

    This is intentional: expression evaluation (BFGS + NMSE) is done once
    per episode in Trainer.train(), avoiding redundant computation.

    Action mask
    -----------
    For each episode i and token j:
        new_pending[i, j] = pending_slots[i] - 1 + arity[j]
    Token j is valid iff:
        new_pending[i, j] >= 0   (doesn't underflow the slot counter)
        new_pending[i, j] <= remaining[i]  (can still be completed in time)

    Parameters
    ----------
    env_template : SymbolicRegressionEnv — only grammar/max_length are used;
                   this env is NOT stepped during batch collection.
    """
    B          = batch_size
    vocab_size = len(grammar)

    # State tensors — shape (B, 1) for easy broadcasting with (1, vocab_size)
    pending_slots    = torch.ones((B, 1), dtype=torch.long, device=device)
    lengths          = torch.zeros((B, 1), dtype=torch.long, device=device)
    done             = torch.zeros(B, dtype=torch.bool, device=device)

    # Precompute arity vector — shape (vocab_size,)
    arities = torch.tensor(
        [grammar.arity[grammar.id_to_token[i]] for i in range(vocab_size)],
        dtype=torch.long, device=device,
    )

    # Per-episode accumulators
    tokens_batch       = [[] for _ in range(B)]
    action_ids_batch   = [[] for _ in range(B)]
    log_probs_batch    = [[] for _ in range(B)]
    entropies_batch    = [[] for _ in range(B)]
    values_batch       = [[] for _ in range(B)]
    observations_batch = [[] for _ in range(B)]

    # Token-id sequence accumulated across steps — shape (B, step)
    current_sequences = torch.empty((B, 0), dtype=torch.long, device=device)

    while not done.all() and current_sequences.shape[1] < max_length:
        remaining = max_length - lengths                         # (B, 1)

        # new_pending[i, j] = pending_slots[i] - 1 + arity[j]
        new_pending = pending_slots - 1 + arities.unsqueeze(0)  # (B, vocab)

        # Valid iff slot counter stays non-negative AND completable in time
        action_mask = (
            (new_pending >= 0) & (new_pending <= remaining)
        ).float()                                                # (B, vocab)

        logits, value = policy(
            token_ids=current_sequences,
            pending_slots=pending_slots,
            length=lengths,
            action_mask=action_mask,
        )

        dist      = Categorical(logits=logits)
        actions   = dist.sample()          # (B,)
        log_probs = dist.log_prob(actions) # (B,)
        entropies = dist.entropy()         # (B,)

        active = ~done  # (B,)

        for i in range(B):
            if not active[i]:
                continue

            action_idx = actions[i].item()
            arity_i    = grammar.arity[grammar.id_to_token[action_idx]]

            tokens_batch[i].append(grammar.id_to_token[action_idx])
            action_ids_batch[i].append(action_idx)
            log_probs_batch[i].append(log_probs[i])
            entropies_batch[i].append(entropies[i])
            values_batch[i].append(value[i])
            observations_batch[i].append({
                "token_ids": (
                    current_sequences[i].clone()
                    if current_sequences.shape[1] > 0
                    else torch.empty(0, dtype=torch.long, device=device)
                ),
                "pending_slots": pending_slots[i, 0].item(),
                "length":        lengths[i, 0].item(),
                "action_mask":   action_mask[i].clone(),
            })

            pending_slots[i, 0] = pending_slots[i, 0] - 1 + arity_i
            lengths[i, 0]      += 1

            if pending_slots[i, 0] == 0:
                done[i] = True

        current_sequences = torch.cat(
            [current_sequences, actions.unsqueeze(1)], dim=1
        )

    # Build trajectory dicts.
    # rewards / final_reward are intentionally empty — Trainer fills them.
    trajectories = []
    for i in range(B):
        trajectories.append({
            "tokens":       tokens_batch[i],
            "action_ids":   action_ids_batch[i],
            "log_probs":    log_probs_batch[i],
            "entropies":    entropies_batch[i],
            "values":       values_batch[i],
            "observations": observations_batch[i],
            "rewards":      [],    # filled by Trainer after expression evaluation
            "final_reward": 0.0,   # filled by Trainer after expression evaluation
        })

    return trajectories