"""Tests for benchmark runner."""

import pytest

from cvxbench.runner import run_single


class TestRunSingle:
    def test_solve_simple_qp(self, simple_qp):
        """Solve simple QP with SCS."""
        result = run_single(simple_qp, "scs", run_index=0, timeout=30, verbose=False)

        assert result.problem == "simple_qp"
        assert result.solver == "scs"
        assert result.status in ("optimal", "optimal_inaccurate")
        assert result.time_sec > 0
        assert result.primal_obj is not None
        # Check solution is close to optimal
        assert abs(result.primal_obj - 1.25) < 0.1

    def test_solve_simple_lp(self, simple_lp):
        """Solve simple LP with SCS."""
        result = run_single(simple_lp, "scs", run_index=0, timeout=30, verbose=False)

        assert result.status in ("optimal", "optimal_inaccurate")
        assert result.primal_obj is not None
        assert abs(result.primal_obj - 1.0) < 0.1

    def test_solve_with_clarabel(self, simple_qp):
        """Solve with Clarabel solver."""
        result = run_single(simple_qp, "clarabel", run_index=0, timeout=30, verbose=False)

        assert result.solver == "clarabel"
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_solve_with_validation(self, simple_qp):
        """Solve with validation enabled."""
        result = run_single(
            simple_qp, "scs", run_index=0, timeout=30, verbose=False, validate=True
        )

        assert result.is_validated
        assert result.primal_residual is not None
        assert result.constraint_violation is not None
        # Should be feasible
        assert result.constraint_violation < 1e-4

    def test_timeout_respected(self, simple_qp):
        """Short timeout doesn't hang."""
        # This should complete quickly, not actually timeout
        result = run_single(simple_qp, "scs", run_index=0, timeout=1, verbose=False)
        assert result.time_sec < 1.0

    def test_run_index_preserved(self, simple_qp):
        """Run index is preserved in result."""
        result = run_single(simple_qp, "scs", run_index=5, timeout=30, verbose=False)
        assert result.run_index == 5


class TestSolverOptions:
    def test_different_solvers_same_result(self, simple_qp):
        """Different solvers get similar objective."""
        scs_result = run_single(simple_qp, "scs", 0, timeout=30, verbose=False)
        clarabel_result = run_single(simple_qp, "clarabel", 0, timeout=30, verbose=False)

        assert scs_result.primal_obj is not None
        assert clarabel_result.primal_obj is not None
        # Both should be close to optimal
        assert abs(scs_result.primal_obj - 1.25) < 0.1
        assert abs(clarabel_result.primal_obj - 1.25) < 0.1
