"""Base classes for benchmark loaders."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class BenchmarkProblem:
    """Unified representation of a benchmark optimization problem.

    The problem is in canonical conic form:
        minimize    (1/2) x^T P x + q^T x
        subject to  A x + s = b
                    s ∈ K

    where K is a Cartesian product of cones.
    """

    name: str
    source: str  # e.g., "maros_meszaros", "smp", "qplib"

    # Dimensions
    n_vars: int
    n_constraints: int

    # Objective: min 0.5 x'Px + q'x
    P: sparse.csc_matrix | None  # None for LP
    q: np.ndarray

    # Constraints: Ax + s = b, s in K
    A: sparse.csc_matrix
    b: np.ndarray

    # Cone specification: list of (type, dim)
    # Types: "zero", "nonneg", "soc", "psd", "exp"
    cones: list[tuple[str, int]]

    # Metadata
    known_optimal: float | None = None
    problem_class: str = "QP"  # LP, QP, SOCP, SDP, MIP, etc.
    has_integer_vars: bool = False

    # Additional metadata
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate problem dimensions."""
        if self.q.shape[0] != self.n_vars:
            msg = f"q has length {self.q.shape[0]}, expected {self.n_vars}"
            raise ValueError(msg)
        if self.A.shape != (self.n_constraints, self.n_vars):
            msg = f"A has shape {self.A.shape}, expected ({self.n_constraints}, {self.n_vars})"
            raise ValueError(msg)
        if self.b.shape[0] != self.n_constraints:
            msg = f"b has length {self.b.shape[0]}, expected {self.n_constraints}"
            raise ValueError(msg)
        if self.P is not None and self.P.shape != (self.n_vars, self.n_vars):
            msg = f"P has shape {self.P.shape}, expected ({self.n_vars}, {self.n_vars})"
            raise ValueError(msg)

        # Validate cone dimensions
        total_cone_dim = sum(dim for _, dim in self.cones)
        if total_cone_dim != self.n_constraints:
            msg = f"Cone dimensions sum to {total_cone_dim}, expected {self.n_constraints}"
            raise ValueError(msg)


class BenchmarkLoader(ABC):
    """Abstract base class for benchmark suite loaders."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the loader.

        Args:
            cache_dir: Directory to cache downloaded benchmark data.
                Defaults to ~/.cache/cvxbench/<suite_name>
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "cvxbench" / self.name()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def name(self) -> str:
        """Return the name of this benchmark suite."""
        ...

    @abstractmethod
    def list_problems(self) -> list[str]:
        """Return list of available problem names."""
        ...

    @abstractmethod
    def load_problem(self, name: str) -> BenchmarkProblem:
        """Load a single problem by name."""
        ...

    def problem_count(self) -> int:
        """Return the number of problems in this suite."""
        return len(self.list_problems())

    def iterate_problems(
        self,
        sample_rate: float = 1.0,
        seed: int | None = None,
    ) -> Iterator[BenchmarkProblem]:
        """Iterate over problems with optional sampling.

        Args:
            sample_rate: Fraction of problems to include (0.0 to 1.0).
            seed: Random seed for reproducible sampling.

        Yields:
            BenchmarkProblem instances.
        """
        problems = self.list_problems()

        if sample_rate < 1.0:
            if seed is not None:
                random.seed(seed)
            k = max(1, int(len(problems) * sample_rate))
            problems = random.sample(problems, k)

        for name in problems:
            try:
                yield self.load_problem(name)
            except Exception as e:
                # Log and skip problematic instances
                print(f"Warning: Failed to load {name}: {e}")
                continue

    def ensure_downloaded(self) -> None:
        """Ensure benchmark data is downloaded to cache.

        Subclasses should override this to download data if needed.
        """
        pass

    def estimated_size_mb(self) -> float:
        """Return estimated size of the benchmark data in MB.

        Subclasses should override this with accurate estimates.
        """
        return 100.0  # Conservative default
