"""
memory.py
=========
Three memory buffer implementations for the RSPG symbolic regression agent.

TopKMemory (baseline)
    Stores the K expressions with the highest reward seen so far.
    Standard experience replay used in DSR (Petersen et al., 2021).

DiverseTopKMemory (original contribution)
    Extends TopKMemory with a structural diversity constraint: two expressions
    can coexist only if their token-level Levenshtein edit distance exceeds a
    configurable threshold. This prevents the memory from collapsing into K
    near-identical copies of the same structure (e.g. 20 variants of x0*x1
    with slightly different constants), which reduces the diversity of the
    replay signal and promotes policy collapse.

PrioritizedTopKMemory (original contribution, inspired by Schaul et al., 2016)
    Extends TopKMemory with priority-based replay ordering. Expressions are
    replayed in order of "surprise" — the absolute deviation of their reward
    from a running EMA baseline. High-priority expressions (very good or very
    unexpected) are replayed first, focusing gradient updates on the most
    informative past experience.

Reference
---------
Schaul, T. et al. "Prioritized Experience Replay." ICLR 2016.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass(order=True)
class MemoryItem:
    """
    One stored expression.

    Ordered by `reward` so that heapq operations (min-heap) always expose
    the worst item at position 0.
    """
    reward:     float
    expr_key:   str       = field(compare=False)
    tokens:     List[str] = field(compare=False)
    infix:      str       = field(compare=False)
    nmse:       float     = field(compare=False)
    complexity: int       = field(compare=False)
    source:     str       = field(compare=False, default="unknown")

    def __repr__(self) -> str:
        return (
            f"MemoryItem(reward={self.reward:.4f}, nmse={self.nmse:.4f}, "
            f"complexity={self.complexity}, infix={self.infix!r})"
        )


# ---------------------------------------------------------------------------
# Standard Top-K memory
# ---------------------------------------------------------------------------

class TopKMemory:
    """
    Keeps the K expressions with the highest reward.

    Internally uses a min-heap (Python's heapq) so that the worst item
    (lowest reward) is always at the root and can be evicted in O(log K).

    Complexity notes
    ----------------
    add()    : O(log K)  amortised — heap push/replace
    remove() : O(K)      — requires heap rebuild after linear-time filter.
               Acceptable for K ≤ 20; called only when updating a duplicate.
    topk()   : O(K log K) — sorts the heap once for display.
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._heap:  List[MemoryItem]       = []
        self._seen:  Dict[str, MemoryItem]  = {}

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return f"TopKMemory(capacity={self.capacity}, size={len(self)})"

    def add(
        self,
        tokens:     List[str],
        infix:      str,
        reward:     float,
        nmse:       float,
        complexity: int,
        source:     str = "unknown",
    ):
        """
        Insert (or update) an expression.

        If the expression (identified by its token sequence) is already
        stored, it is updated only if the new reward is strictly better.
        """
        expr_key = " ".join(tokens)

        if expr_key in self._seen:
            existing = self._seen[expr_key]
            if reward <= existing.reward:
                return          # not an improvement
            self.remove(expr_key)

        item = MemoryItem(
            reward=reward,
            expr_key=expr_key,
            tokens=list(tokens),
            infix=infix,
            nmse=float(nmse),
            complexity=int(complexity),
            source=source,
        )

        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
            self._seen[expr_key] = item
        elif reward > self._heap[0].reward:
            worst = heapq.heapreplace(self._heap, item)
            self._seen.pop(worst.expr_key, None)
            self._seen[expr_key] = item

    def remove(self, expr_key: str):
        """
        Remove an expression by key.

        O(K): filters the heap list then re-heapifies.
        Only called when updating a duplicate, so K ≤ 20 is fine.
        """
        if expr_key not in self._seen:
            return
        target      = self._seen.pop(expr_key)
        self._heap  = [it for it in self._heap if it.expr_key != target.expr_key]
        heapq.heapify(self._heap)

    def topk(self) -> List[MemoryItem]:
        """Return all stored items sorted by reward descending (best first)."""
        return sorted(self._heap, key=lambda x: x.reward, reverse=True)

    def to_rows(self) -> List[Dict]:
        return [
            {
                "reward":     item.reward,
                "nmse":       item.nmse,
                "complexity": item.complexity,
                "source":     item.source,
                "tokens":     item.tokens,
                "infix":      item.infix,
            }
            for item in self.topk()
        ]

    def pretty_print(self, title: Optional[str] = None):
        if title:
            print(f"\n{title}")
        rows = self.to_rows()
        if not rows:
            print("Memory is empty.")
            return
        for rank, row in enumerate(rows, start=1):
            print(f"Rank {rank}")
            print(f"  Reward:     {row['reward']:.6f}")
            print(f"  NMSE:       {row['nmse']:.6f}")
            print(f"  Complexity: {row['complexity']}")
            print(f"  Source:     {row['source']}")
            print(f"  Tokens:     {row['tokens']}")
            print(f"  Expression: {row['infix']}")
            print("  " + "-" * 40)


# ---------------------------------------------------------------------------
# Edit distance helper
# ---------------------------------------------------------------------------

def _edit_distance(a: List[str], b: List[str]) -> int:
    """
    Token-level Levenshtein edit distance.

    O(|a| × |b|) time, O(|b|) space (rolling-row DP).
    Acceptable for expression lengths ≤ 30 (max_length config).
    """
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j],      # deletion
                                   curr[j - 1],  # insertion
                                   prev[j - 1])  # substitution
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# Diverse Top-K memory (original contribution)
# ---------------------------------------------------------------------------

class DiverseTopKMemory:
    """
    Top-K memory with structural diversity enforcement.

    Two expressions can coexist only if their token-level edit distance
    exceeds `min_edit_distance`. When a candidate is too similar to an
    existing item, it replaces that item only if its reward is strictly
    better — otherwise it is rejected.

    This prevents the memory from filling with near-identical expressions
    (e.g. 20 variants of x0*x1 differing only by a constant), which would
    reduce the diversity of the gradient signal and promote policy collapse.

    Complexity
    ----------
    add()  : O(K × L²) — one edit-distance call per stored item (K ≤ 20,
             L ≤ 30 tokens), so at most 20 × 30² = 18,000 operations.
             This is negligible compared to BFGS evaluation.
    topk() : O(1) — items are kept sorted on insertion.

    Parameters
    ----------
    capacity          : maximum number of expressions stored.
    min_edit_distance : minimum token-level edit distance required between
                        any two stored expressions.
                        0  → same behaviour as TopKMemory (no constraint).
                        3  → good default: filters near-duplicates (same
                             structure, different constants) while allowing
                             genuinely different forms.
                        5+ → stricter; useful for large grammars.
    """

    def __init__(self, capacity: int = 20, min_edit_distance: int = 3):
        self.capacity          = capacity
        self.min_edit_distance = min_edit_distance
        self._items: List[MemoryItem] = []   # sorted descending by reward
        self._seen:  Dict[str, MemoryItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"DiverseTopKMemory(capacity={self.capacity}, "
            f"min_edit_distance={self.min_edit_distance}, size={len(self)})"
        )

    def _find_nearest(self, tokens: List[str]) -> Optional[MemoryItem]:
        """Return the stored item with minimum edit distance to `tokens`."""
        if not self._items:
            return None
        return min(self._items, key=lambda it: _edit_distance(tokens, it.tokens))

    def _min_distance_to_memory(self, tokens: List[str]) -> int:
        if not self._items:
            return 999
        return min(_edit_distance(tokens, it.tokens) for it in self._items)

    def add(
        self,
        tokens:     List[str],
        infix:      str,
        reward:     float,
        nmse:       float,
        complexity: int,
        source:     str = "unknown",
    ):
        expr_key = " ".join(tokens)

        # Exact duplicate: update if strictly better
        if expr_key in self._seen:
            if reward <= self._seen[expr_key].reward:
                return
            self._remove_by_key(expr_key)

        item = MemoryItem(
            reward=reward,
            expr_key=expr_key,
            tokens=list(tokens),
            infix=infix,
            nmse=float(nmse),
            complexity=int(complexity),
            source=source,
        )

        # Diversity check: is there a too-similar item in memory?
        nearest = self._find_nearest(tokens)
        if (
            nearest is not None
            and _edit_distance(tokens, nearest.tokens) < self.min_edit_distance
        ):
            # Too similar — replace the neighbour only if strictly better
            if reward > nearest.reward:
                self._remove_by_key(nearest.expr_key)
                # Fall through to insertion below
            else:
                return   # reject: not diverse enough AND not better

        # Insert
        if len(self._items) < self.capacity:
            self._items.append(item)
            self._seen[expr_key] = item
            self._items.sort(key=lambda x: x.reward, reverse=True)
        else:
            # Evict the worst item if the new one is better
            worst = self._items[-1]
            if reward > worst.reward:
                self._remove_by_key(worst.expr_key)
                self._items.append(item)
                self._seen[expr_key] = item
                self._items.sort(key=lambda x: x.reward, reverse=True)

    def _remove_by_key(self, expr_key: str):
        if expr_key not in self._seen:
            return
        self._seen.pop(expr_key)
        self._items = [it for it in self._items if it.expr_key != expr_key]

    def topk(self) -> List[MemoryItem]:
        return list(self._items)   # already sorted descending

    def to_rows(self) -> List[Dict]:
        return [
            {
                "reward":     item.reward,
                "nmse":       item.nmse,
                "complexity": item.complexity,
                "source":     item.source,
                "tokens":     item.tokens,
                "infix":      item.infix,
            }
            for item in self.topk()
        ]

    def pretty_print(self, title: Optional[str] = None):
        if title:
            print(f"\n{title}")
        rows = self.to_rows()
        if not rows:
            print("Memory is empty.")
            return
        for rank, row in enumerate(rows, start=1):
            print(f"Rank {rank}")
            print(f"  Reward:     {row['reward']:.6f}")
            print(f"  NMSE:       {row['nmse']:.6f}")
            print(f"  Complexity: {row['complexity']}")
            print(f"  Source:     {row['source']}")
            print(f"  Tokens:     {row['tokens']}")
            print(f"  Expression: {row['infix']}")
            print("  " + "-" * 40)

    def diversity_stats(self) -> Dict:
        """
        Pairwise edit-distance statistics across stored expressions.
        Useful for logging and ablation analysis.
        """
        items = self._items
        n = len(items)
        if n < 2:
            return {"mean_pairwise_distance": 0.0, "min_pairwise_distance": 0}
        distances = [
            _edit_distance(items[i].tokens, items[j].tokens)
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return {
            "mean_pairwise_distance": sum(distances) / len(distances),
            "min_pairwise_distance":  min(distances),
            "max_pairwise_distance":  max(distances),
            "n_items":                n,
        }


# ---------------------------------------------------------------------------
# Prioritized Top-K memory (original contribution, Schaul et al., 2016)
# ---------------------------------------------------------------------------

class PrioritizedTopKMemory(TopKMemory):
    """
    Top-K memory with priority-based replay ordering.

    Extends TopKMemory — same add/remove/topk interface — but assigns a
    priority score to each expression and orders to_rows() by priority
    descending (most surprising first).

    Priority score = |reward - baseline|^alpha
      where baseline is an exponential moving average of all rewards seen.

    A high-priority expression is one whose reward was very different from
    what the policy expected — either much better or much worse. Replaying
    these in priority order focuses gradient updates on the most informative
    past experience, analogous to Prioritized Experience Replay in standard RL.

    Reference
    ---------
    Schaul, T. et al. "Prioritized Experience Replay." ICLR 2016.

    Parameters
    ----------
    capacity           : max expressions stored (same as TopKMemory).
    alpha              : prioritisation exponent in [0, 1].
                         0.0 = uniform (identical to TopKMemory).
                         1.0 = full priority (default).
    baseline_momentum  : EMA momentum for the baseline (0.9 default).
    max_replay         : max items returned by to_rows(). None = all.

    Note on combined use with DiverseTopKMemory
    -------------------------------------------
    When both diversity and prioritization are desired, Trainer uses
    PrioritizedTopKMemory (which handles capacity and priority) and the
    diversity constraint is NOT enforced — the two cannot be cleanly
    composed without duplicating the priority bookkeeping. A future
    improvement would be a PrioritizedDiverseTopKMemory class that
    inherits from DiverseTopKMemory instead.
    """

    def __init__(
        self,
        capacity:          int            = 20,
        alpha:             float          = 1.0,
        baseline_momentum: float          = 0.9,
        max_replay:        Optional[int]  = None,
    ):
        super().__init__(capacity=capacity)
        self.alpha             = alpha
        self.baseline_momentum = baseline_momentum
        self.max_replay        = max_replay
        self._baseline:   float                 = 0.0
        self._priorities: Dict[str, float]      = {}

    def __repr__(self) -> str:
        return (
            f"PrioritizedTopKMemory(capacity={self.capacity}, "
            f"alpha={self.alpha}, size={len(self)})"
        )

    def _update_baseline(self, reward: float):
        self._baseline = (
            self.baseline_momentum * self._baseline
            + (1.0 - self.baseline_momentum) * reward
        )

    def _compute_priority(self, reward: float) -> float:
        """Priority = max(|reward - baseline|^alpha, 1e-6)."""
        surprise = abs(reward - self._baseline)
        return max(surprise ** self.alpha, 1e-6)

    def add(
        self,
        tokens:     List[str],
        infix:      str,
        reward:     float,
        nmse:       float,
        complexity: int,
        source:     str = "unknown",
    ):
        self._update_baseline(reward)
        priority = self._compute_priority(reward)
        expr_key = " ".join(tokens)

        # Update priority if expression already exists (take the max)
        if expr_key in self._seen:
            self._priorities[expr_key] = max(
                self._priorities.get(expr_key, 0.0), priority
            )

        super().add(tokens, infix, reward, nmse, complexity, source)

        if expr_key in self._seen:
            self._priorities[expr_key] = priority

    def remove(self, expr_key: str):
        self._priorities.pop(expr_key, None)
        super().remove(expr_key)

    def to_rows_prioritized(self) -> List[Dict]:
        """
        Returns items sorted by priority descending (most surprising first),
        up to max_replay items. Each row includes a 'priority' field.
        """
        rows = [
            {
                "reward":     item.reward,
                "nmse":       item.nmse,
                "complexity": item.complexity,
                "source":     item.source,
                "tokens":     item.tokens,
                "infix":      item.infix,
                "priority":   self._priorities.get(item.expr_key, 1e-6),
            }
            for item in self.topk()
        ]
        rows.sort(key=lambda r: r["priority"], reverse=True)
        if self.max_replay is not None:
            rows = rows[: self.max_replay]
        return rows

    def to_rows(self) -> List[Dict]:
        """
        Override: returns rows sorted by priority (most surprising first).
        Strips the 'priority' field to keep the same interface as TopKMemory.
        """
        rows = self.to_rows_prioritized()
        for r in rows:
            r.pop("priority", None)
        return rows

    def priority_stats(self) -> Dict:
        """Diagnostic: priority distribution across stored expressions."""
        if not self._priorities:
            return {}
        pvals = list(self._priorities.values())
        return {
            "baseline":      round(self._baseline, 4),
            "mean_priority": round(sum(pvals) / len(pvals), 4),
            "max_priority":  round(max(pvals), 4),
            "min_priority":  round(min(pvals), 4),
            "n_items":       len(pvals),
        }