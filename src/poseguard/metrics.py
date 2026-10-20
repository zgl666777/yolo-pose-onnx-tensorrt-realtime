from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Distribution:
    samples: deque = field(default_factory=lambda: deque(maxlen=20000))
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0

    def add(self, value: float):
        self.samples.append(float(value))
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)

    def report(self):
        return {"count": self.count, "mean": self.total / max(self.count, 1),
                "p50": float(np.percentile(self.samples, 50)) if self.samples else None,
                "p95": float(np.percentile(self.samples, 95)) if self.samples else None,
                "max": self.maximum, "percentile_sample_count": len(self.samples)}


class Metrics:
    def __init__(self):
        self.distributions: dict[str, Distribution] = {}

    def add(self, **values):
        for name, value in values.items():
            self.distributions.setdefault(name, Distribution()).add(value)

    def report(self):
        return {name: dist.report() for name, dist in self.distributions.items()}

