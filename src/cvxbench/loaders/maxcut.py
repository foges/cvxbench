"""Generator for Max-Cut SDP relaxation benchmark problems.

Max-Cut SDP relaxation is a classic problem where first-order methods
like SCS significantly outperform interior-point methods like Clarabel.

The Max-Cut SDP relaxation is:
    maximize    (1/4) * <L, X>
    subject to  diag(X) = 1
                X >= 0  (PSD)

where L is the graph Laplacian.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from cvxbench.loaders.base import BenchmarkLoader, BenchmarkProblem

# Pre-defined problem configurations: (nodes, edge_density, seed)
MAXCUT_PROBLEMS = {
    # Small (quick tests)
    "maxcut_50_01": (50, 0.1, 1),
    "maxcut_50_03": (50, 0.3, 2),
    "maxcut_100_01": (100, 0.1, 3),
    "maxcut_100_03": (100, 0.3, 4),
    # Medium
    "maxcut_200_01": (200, 0.1, 5),
    "maxcut_200_03": (200, 0.3, 6),
    "maxcut_300_01": (300, 0.1, 7),
    "maxcut_500_01": (500, 0.1, 8),
    # Large (SCS dominates here)
    "maxcut_500_03": (500, 0.3, 9),
    "maxcut_750_01": (750, 0.1, 10),
    "maxcut_1000_01": (1000, 0.1, 11),
    "maxcut_1000_03": (1000, 0.3, 12),
    # Very large (only SCS can solve in reasonable time)
    "maxcut_1500_01": (1500, 0.1, 13),
    "maxcut_2000_01": (2000, 0.1, 14),
}


def generate_random_graph(n: int, edge_density: float, seed: int) -> np.ndarray:
    """Generate a random symmetric adjacency matrix.

    Args:
        n: Number of nodes.
        edge_density: Probability of edge between any two nodes.
        seed: Random seed for reproducibility.

    Returns:
        Symmetric adjacency matrix (n x n).
    """
    rng = np.random.default_rng(seed)
    # Generate upper triangular random matrix
    adj = (rng.random((n, n)) < edge_density).astype(float)
    adj = np.triu(adj, 1)
    # Make symmetric
    adj = adj + adj.T
    return adj


def maxcut_to_conic(n: int, edge_density: float, seed: int, name: str) -> BenchmarkProblem:
    """Generate Max-Cut SDP relaxation in conic form.

    The Max-Cut SDP is:
        maximize    (1/4) * <L, X>
        subject to  diag(X) = 1
                    X >= 0  (PSD)

    In conic form (minimize with equality + PSD cone):
        minimize    q^T x           (x = vec(X), q = -vec(L)/4)
        subject to  A x + s = b     (diagonal extraction: A selects diag, b = 1)
                    s in {0}^n      (equality constraint)
                    x in S_+^{n^2}  (X is PSD)

    Actually, we need to be careful. The standard conic form is:
        min q'x s.t. Ax + s = b, s in K

    For Max-Cut:
    - We want diag(X) = 1, which is an equality constraint
    - We want X >= 0 (PSD)

    We'll represent this as:
    - n_vars = n^2 (vectorized X)
    - Zero cone for diag(X) = 1 (n constraints)
    - PSD cone for X >= 0 (n^2 dimensional)

    Args:
        n: Number of graph nodes.
        edge_density: Edge probability.
        seed: Random seed.
        name: Problem name.

    Returns:
        BenchmarkProblem in conic form.
    """
    # Generate random graph
    adj = generate_random_graph(n, edge_density, seed)

    # Compute Laplacian: L = D - A where D = diag(sum(A))
    degrees = adj.sum(axis=1)
    laplacian = np.diag(degrees) - adj

    # Objective: minimize -<L/4, X> = minimize q'vec(X)
    # where q = -vec(L/4)
    # Using column-major (Fortran) vectorization to match CVXPY
    q = -laplacian.flatten(order="F") / 4

    n_vars = n * n  # Vectorized X

    # Constraints:
    # 1. diag(X) = 1  (n equality constraints via zero cone)
    # 2. X >= 0 (PSD)  (n^2 dimensional PSD cone)

    # Build A matrix for diagonal extraction
    # A[i, :] selects X[i,i] from vec(X)
    # In column-major order, X[i,i] is at position i*n + i
    rows = []
    cols = []
    data = []
    for i in range(n):
        rows.append(i)
        cols.append(i * n + i)  # Column-major index of X[i,i]
        data.append(1.0)

    # A has shape (n + n^2, n^2)
    # First n rows: diagonal extraction for zero cone
    # Next n^2 rows: identity for PSD cone (s = b - Ax, need s = vec(X))

    # For the PSD cone part: we want s = vec(X), so A = -I, b = 0
    # But in our form Ax + s = b, if we want s = x (the PSD variable equals x),
    # we need -x + s = 0, so A = -I, b = 0 for PSD part

    # Actually, let me reconsider the formulation.
    # The standard form is: min q'x s.t. Ax + s = b, s in K
    #
    # For Max-Cut we have decision variable X (n x n PSD matrix).
    # Let x = vec(X). We want:
    #   - diag(X) = 1  (n equalities)
    #   - X >= 0 (PSD)
    #
    # The tricky part is that x itself should be PSD, not a slack variable.
    # One way: use a zero cone for diag constraints and treat x as PSD directly.
    #
    # In CVXPY's canonicalization, the PSD constraint X >> 0 becomes:
    #   vec(X) in S_+
    #
    # So we can model as:
    #   min q'x
    #   s.t. A_eq @ x = b_eq  (diag(X) = 1)
    #        x in S_+^{n^2}   (X is PSD, vectorized)
    #
    # In conic form with slack:
    #   min q'x
    #   s.t. [A_eq; -I] @ x + [s_eq; s_psd] = [b_eq; 0]
    #        s_eq in {0}^n
    #        s_psd in S_+^{n^2}
    #
    # Wait, this doesn't quite work because s_psd = -x, so s_psd in S_+ means -x in S_+.
    #
    # Let me use the formulation from my sdplib loader instead.
    # The CVXPY runner handles: Ax + s = b, s in K
    # For PSD: s = b - Ax must be PSD
    #
    # If we want x = vec(X) to be PSD (not a slack), we need a different approach.
    #
    # Actually, looking at my runner.py build_cvxpy_problem:
    # - It creates a variable x (n_vars)
    # - For PSD cones, it does: s = b_block - A_block @ x, reshape s to matrix, S >> 0
    #
    # So the PSD constraint is on the slack s, not on x directly.
    #
    # For Max-Cut, we can reformulate:
    # - Let the decision be the n diagonal elements (since diag(X) = 1, they're fixed)
    # - Actually no, the full X is the decision variable
    #
    # Let me use a different approach: make x the n^2 dimensional vectorized X,
    # and use constraints to enforce structure.

    # Simpler approach: just use diagonal constraints as equality (zero cone)
    # and then add a PSD cone constraint on the full X.
    #
    # Constraint structure:
    # - Zero cone (dim n): diag(X) = 1
    # - PSD cone (dim n^2): X >= 0, represented as: 0 - (-I)x + s = 0, s in PSD
    #   i.e., Ix + s = 0 with s in PSD... but then s = -x which means -X in PSD, wrong.
    #
    # I need: s = X (not -X). So: -Ix + s = 0, i.e., A = -I, b = 0.
    # Then s = Ix = x, and s in PSD means x in PSD (i.e., X is PSD). Correct!

    # Build full A matrix
    n_eq = n  # Diagonal equality constraints
    n_psd = n * n  # PSD cone dimension
    total_constraints = n_eq + n_psd

    # Equality part: extract diagonal
    for i in range(n):
        rows.append(i)
        cols.append(i * n + i)
        data.append(1.0)

    # PSD part: -I (so that s = x)
    for i in range(n_psd):
        rows.append(n_eq + i)
        cols.append(i)
        data.append(-1.0)

    # Rebuild A (we added to rows/cols/data twice for diagonal, need to redo)
    rows = []
    cols = []
    data = []

    # Equality constraints: diag(X) = 1
    for i in range(n):
        rows.append(i)
        cols.append(i * n + i)
        data.append(1.0)

    # PSD constraint: -I @ x + s = 0, s in PSD
    for i in range(n_psd):
        rows.append(n_eq + i)
        cols.append(i)
        data.append(-1.0)

    A = sparse.csc_matrix((data, (rows, cols)), shape=(total_constraints, n_vars))

    # RHS b: [1, 1, ..., 1, 0, 0, ..., 0]
    b = np.zeros(total_constraints)
    b[:n_eq] = 1.0

    # Cones: zero cone (n), then PSD cone (n^2)
    cones = [("zero", n_eq), ("psd", n_psd)]

    # Compute edge count for metadata
    edge_count = int(adj.sum() / 2)

    return BenchmarkProblem(
        name=name,
        source="maxcut",
        n_vars=n_vars,
        n_constraints=total_constraints,
        P=None,  # Linear objective
        q=q,
        A=A,
        b=b,
        cones=cones,
        problem_class="SDP",
        metadata={"nodes": n, "edges": edge_count, "density": edge_density},
    )


class MaxCutLoader(BenchmarkLoader):
    """Loader for Max-Cut SDP relaxation benchmark problems.

    Max-Cut SDP is a problem class where first-order methods (SCS) significantly
    outperform interior-point methods (Clarabel) due to sparse constraint structure.
    """

    def name(self) -> str:
        """Return loader name."""
        return "maxcut"

    def list_problems(self) -> list[str]:
        """Return list of available problems."""
        return list(MAXCUT_PROBLEMS.keys())

    def load_problem(self, name: str) -> BenchmarkProblem:
        """Load a problem by name.

        Args:
            name: Problem name (e.g., "maxcut_100_01").

        Returns:
            BenchmarkProblem instance.
        """
        if name not in MAXCUT_PROBLEMS:
            msg = f"Unknown problem: {name}"
            raise ValueError(msg)

        n, density, seed = MAXCUT_PROBLEMS[name]
        return maxcut_to_conic(n, density, seed, name)

    def estimated_size_mb(self) -> float:
        """Estimate total download size in MB."""
        return 0.0  # Generated, not downloaded
