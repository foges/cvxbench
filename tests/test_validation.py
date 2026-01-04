"""Tests for solution validation."""

import numpy as np
import pytest

from cvxbench.validation import (
    compute_constraint_violation,
    compute_objective,
    compute_primal_residual,
    validate_solution,
)


class TestComputeObjective:
    def test_qp_objective(self, simple_qp):
        """Test QP objective computation."""
        x = np.array([0.5, 0.5])
        obj = compute_objective(simple_qp, x)
        # 0.5 * (0.25 + 0.25) + 0.5 + 0.5 = 0.25 + 1.0 = 1.25
        assert abs(obj - 1.25) < 1e-10

    def test_lp_objective(self, simple_lp):
        """Test LP objective (no quadratic term)."""
        x = np.array([1.0, 0.0])
        obj = compute_objective(simple_lp, x)
        assert abs(obj - 1.0) < 1e-10


class TestPrimalResidual:
    def test_equality_satisfied(self, equality_qp):
        """Primal residual is zero when equality is satisfied."""
        x = np.array([0.5, 0.5])  # x1 + x2 = 1
        residual = compute_primal_residual(equality_qp, x)
        assert residual < 1e-10

    def test_equality_violated(self, equality_qp):
        """Primal residual is nonzero when equality is violated."""
        x = np.array([0.3, 0.3])  # x1 + x2 = 0.6, should be 1
        residual = compute_primal_residual(equality_qp, x)
        assert abs(residual - 0.4) < 1e-10

    def test_inequality_no_residual(self, simple_qp):
        """Inequality constraints don't contribute to primal residual."""
        # simple_qp has only nonneg cone (inequalities)
        x = np.array([0.5, 0.5])
        residual = compute_primal_residual(simple_qp, x)
        assert residual == 0.0


class TestConstraintViolation:
    def test_feasible_point(self, simple_qp):
        """No violation for feasible point."""
        x = np.array([0.5, 0.5])  # satisfies x1+x2>=1, x>=0
        violation = compute_constraint_violation(simple_qp, x)
        assert violation < 1e-10

    def test_infeasible_point(self, simple_qp):
        """Violation detected for infeasible point."""
        x = np.array([0.3, 0.3])  # violates x1+x2>=1
        violation = compute_constraint_violation(simple_qp, x)
        # -0.3 - 0.3 - (-1) = 0.4 violation
        assert abs(violation - 0.4) < 1e-10

    def test_equality_violation(self, equality_qp):
        """Equality violation detected."""
        x = np.array([0.3, 0.3])  # x1+x2=0.6, should be 1
        violation = compute_constraint_violation(equality_qp, x)
        assert abs(violation - 0.4) < 1e-10


class TestValidateSolution:
    def test_optimal_solution(self, simple_qp):
        """Validate optimal solution."""
        x = np.array([0.5, 0.5])
        result = validate_solution(simple_qp, x, reference_obj=1.25)

        assert result.is_feasible
        assert result.constraint_violation < 1e-6
        assert abs(result.primal_obj - 1.25) < 1e-6
        assert result.obj_error is not None
        assert result.obj_error < 1e-6

    def test_infeasible_solution(self, simple_qp):
        """Detect infeasible solution."""
        x = np.array([0.2, 0.2])  # violates x1+x2>=1
        result = validate_solution(simple_qp, x)

        assert not result.is_feasible
        assert result.constraint_violation > 0.5

    def test_suboptimal_solution(self, simple_qp):
        """Detect suboptimal objective."""
        x = np.array([0.6, 0.6])  # feasible but not optimal
        result = validate_solution(simple_qp, x, reference_obj=1.25)

        assert result.is_feasible
        assert result.obj_error is not None
        assert result.obj_error > 0.1  # significantly different from optimal
