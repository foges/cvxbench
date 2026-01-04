"""Shared fixtures for cvxbench tests."""

import numpy as np
import pytest
from scipy import sparse

from cvxbench.loaders.base import BenchmarkProblem


@pytest.fixture
def simple_qp() -> BenchmarkProblem:
    """Simple 2-variable QP for testing.

    min 0.5 * (x1^2 + x2^2) + x1 + x2
    s.t. x1 + x2 >= 1
         x1 >= 0, x2 >= 0

    Optimal: x = (0.5, 0.5), obj = 1.25
    """
    n, m = 2, 3
    P = sparse.csc_matrix(np.eye(2))
    q = np.array([1.0, 1.0])
    # Constraints: -x1 - x2 <= -1, -x1 <= 0, -x2 <= 0
    A = sparse.csc_matrix([
        [-1.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])
    b = np.array([-1.0, 0.0, 0.0])
    cones = [("nonneg", 3)]

    return BenchmarkProblem(
        name="simple_qp",
        source="test",
        n_vars=n,
        n_constraints=m,
        P=P,
        q=q,
        A=A,
        b=b,
        cones=cones,
        known_optimal=1.25,
    )


@pytest.fixture
def simple_lp() -> BenchmarkProblem:
    """Simple 2-variable LP for testing.

    min x1 + 2*x2
    s.t. x1 + x2 = 1
         x1 >= 0, x2 >= 0

    Optimal: x = (1, 0), obj = 1
    """
    n, m = 2, 3
    P = None
    q = np.array([1.0, 2.0])
    A = sparse.csc_matrix([
        [1.0, 1.0],   # equality
        [-1.0, 0.0],  # x1 >= 0
        [0.0, -1.0],  # x2 >= 0
    ])
    b = np.array([1.0, 0.0, 0.0])
    cones = [("zero", 1), ("nonneg", 2)]

    return BenchmarkProblem(
        name="simple_lp",
        source="test",
        n_vars=n,
        n_constraints=m,
        P=P,
        q=q,
        A=A,
        b=b,
        cones=cones,
        known_optimal=1.0,
    )


@pytest.fixture
def equality_qp() -> BenchmarkProblem:
    """QP with equality constraints.

    min 0.5 * (x1^2 + x2^2)
    s.t. x1 + x2 = 1

    Optimal: x = (0.5, 0.5), obj = 0.25
    """
    n, m = 2, 1
    P = sparse.csc_matrix(np.eye(2))
    q = np.array([0.0, 0.0])
    A = sparse.csc_matrix([[1.0, 1.0]])
    b = np.array([1.0])
    cones = [("zero", 1)]

    return BenchmarkProblem(
        name="equality_qp",
        source="test",
        n_vars=n,
        n_constraints=m,
        P=P,
        q=q,
        A=A,
        b=b,
        cones=cones,
        known_optimal=0.25,
    )
