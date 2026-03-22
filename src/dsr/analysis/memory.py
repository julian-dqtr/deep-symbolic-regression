import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass(order=True)
class MemoryItem:
    reward: float
    expr_key: str = field(compare=False)
    tokens: List[str] = field(compare=False)
    infix: str = field(compare=False)
    nmse: float = field(compare=False)
    complexity: int = field(compare=False)
    source: str = field(compare=False, default="unknown")


class TopKMemory:
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._heap: List[MemoryItem] = []
        self._seen: Dict[str, MemoryItem] = {}

    def __len__(self):
        return len(self._heap)

    def add(
        self,
        tokens: List[str],
        infix: str,
        reward: float,
        nmse: float,
        complexity: int,
        source: str = "unknown",
    ):
        expr_key = " ".join(tokens)

        if expr_key in self._seen:
            existing = self._seen[expr_key]
            if reward <= existing.reward:
                return
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
        else:
            if reward > self._heap[0].reward:
                worst = heapq.heapreplace(self._heap, item)
                self._seen.pop(worst.expr_key, None)
                self._seen[expr_key] = item

    def remove(self, expr_key: str):
        if expr_key not in self._seen:
            return

        target = self._seen.pop(expr_key)
        self._heap = [item for item in self._heap if item.expr_key != target.expr_key]
        heapq.heapify(self._heap)

    def topk(self) -> List[MemoryItem]:
        return sorted(self._heap, key=lambda x: x.reward, reverse=True)

    def to_rows(self) -> List[Dict]:
        rows = []
        for item in self.topk():
            rows.append(
                {
                    "reward": item.reward,
                    "nmse": item.nmse,
                    "complexity": item.complexity,
                    "source": item.source,
                    "tokens": item.tokens,
                    "infix": item.infix,
                }
            )
        return rows

    def pretty_print(self, title: Optional[str] = None):
        if title:
            print(f"\n{title}")

        rows = self.to_rows()
        if not rows:
            print("Memory is empty.")
            return

        for rank, row in enumerate(rows, start=1):
            print(f"Rank {rank}")
            print(f"  Reward: {row['reward']:.6f}")
            print(f"  NMSE: {row['nmse']:.6f}")
            print(f"  Complexity: {row['complexity']}")
            print(f"  Source: {row['source']}")
            print(f"  Tokens: {row['tokens']}")
            print(f"  Expression: {row['infix']}")
            print("  " + "-" * 40)