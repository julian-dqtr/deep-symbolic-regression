import torch

from ..core.expression import safe_prefix_to_infix


def _make_policy_inputs(tokens, grammar, X, y, pending_slots, device):
    token_ids = torch.tensor(
        [grammar.token_to_id[t] for t in tokens],
        dtype=torch.long,
        device=device,
    )

    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    return token_ids, x_tensor, y_tensor, pending_slots, len(tokens)


def _can_still_complete(pending_slots: int, current_length: int, max_length: int) -> bool:
    remaining_steps = max_length - current_length
    return pending_slots <= remaining_steps


def beam_search(
    policy,
    grammar,
    X,
    y,
    beam_width=10,
    max_length=30,
    temperature=1.0,
    device="cpu",
):
    """
    Structure-aware beam search over prefix expressions.

    Args:
        policy: trained policy network returning (logits, value)
        grammar: grammar object
        X, y: dataset
        beam_width: number of beams kept at each step
        max_length: max decoding length
        temperature: logits temperature (>1 = softer, <1 = sharper)
        device: cpu/cuda

    Returns:
        list of candidate dicts
    """
    policy.eval()

    beams = [
        {
            "tokens": [],
            "logprob": 0.0,
            "pending_slots": 1,
            "complete": False,
        }
    ]

    completed = []

    for _ in range(max_length):
        new_beams = []

        for beam in beams:
            if beam["complete"]:
                completed.append(beam)
                continue

            current_length = len(beam["tokens"])

            if beam["pending_slots"] <= 0:
                beam["complete"] = True
                completed.append(beam)
                continue

            if not _can_still_complete(
                pending_slots=beam["pending_slots"],
                current_length=current_length,
                max_length=max_length,
            ):
                continue

            token_ids, x_tensor, y_tensor, pending_slots, length = _make_policy_inputs(
                tokens=beam["tokens"],
                grammar=grammar,
                X=X,
                y=y,
                pending_slots=beam["pending_slots"],
                device=device,
            )

            action_mask = torch.ones(len(grammar), dtype=torch.float32, device=device)

            with torch.no_grad():
                policy_out = policy(
                    token_ids=token_ids,
                    x=x_tensor,
                    y=y_tensor,
                    pending_slots=pending_slots,
                    length=length,
                    action_mask=action_mask,
                )

                # Policy now returns (logits, value)
                if isinstance(policy_out, tuple):
                    logits, _ = policy_out
                else:
                    logits = policy_out

                logits = logits / temperature
                log_probs = torch.log_softmax(logits, dim=-1)

            top_k = min(max(beam_width * 3, 10), len(grammar))
            top_log_probs, top_indices = torch.topk(log_probs, k=top_k)

            for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                token = grammar.id_to_token[idx]
                new_tokens = beam["tokens"] + [token]
                new_pending_slots = beam["pending_slots"] - 1 + grammar.arity[token]
                new_length = len(new_tokens)

                if new_pending_slots < 0:
                    continue

                if not _can_still_complete(
                    pending_slots=new_pending_slots,
                    current_length=new_length,
                    max_length=max_length,
                ):
                    continue

                new_beam = {
                    "tokens": new_tokens,
                    "logprob": beam["logprob"] + lp,
                    "pending_slots": new_pending_slots,
                    "complete": new_pending_slots == 0,
                }
                new_beams.append(new_beam)

        if not new_beams:
            break

        def beam_score(b):
            completion_bonus = 1.0 if b["complete"] else 0.0
            length_penalty = 0.01 * len(b["tokens"])
            return b["logprob"] + completion_bonus - length_penalty

        new_beams = sorted(new_beams, key=beam_score, reverse=True)[:beam_width]
        beams = new_beams

        if all(b["complete"] for b in beams):
            completed.extend(beams)
            break

    final_candidates = completed if completed else beams

    final_candidates = sorted(
        final_candidates,
        key=lambda b: (b["complete"], b["logprob"] - 0.01 * len(b["tokens"])),
        reverse=True,
    )

    results = []
    for cand in final_candidates:
        results.append(
            {
                "tokens": cand["tokens"],
                "logprob": cand["logprob"],
                "infix": safe_prefix_to_infix(cand["tokens"], grammar),
                "pending_slots": cand["pending_slots"],
                "complete": cand["complete"],
            }
        )

    return results[:beam_width]