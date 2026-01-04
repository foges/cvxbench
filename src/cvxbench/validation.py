"""Solution validation for cvxbench.

Computes residuals, constraint violations, and validates solutions against
reference values for benchmarking solver correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvxbench.loaders.base import BenchmarkProblem


@dataclass
class ValidationResult:
    """Solution validation metrics."""

    primal_residual: float
    """||Ax + s - b|| for conic constraints."""

    dual_residual: float
    """||Px + q + A'y|| for QP dual feasibility."""

    constraint_violation: float
    """Maximum constraint violation (negative slack)."""

    gap: float
    """Relative duality gap: |p_obj - d_obj| / (1 + |p_obj|)."""

    primal_obj: float
    """Primal objective value."""

    dual_obj: float | None
    """Dual objective value (if available)."""

    obj_error: float | None
    """Relative error vs reference: |obj - ref| / (1 + |ref|)."""

    is_feasible: bool
    """True if primal_residual and constraint_violation < tolerance."""

    is_optimal: bool
    """True if feasible and gap < tolerance."""


def compute_primal_residual(
    problem: BenchmarkProblem,
    x: np.ndarray,
) -> float:
    """Compute primal residual for equality constraints only.

    For conic form: Ax + s = b, s in K
    For zero cone (equality): s = 0, so Ax = b
    For nonneg cone (inequality): s >= 0, so Ax <= b (no equality residual)

    Args:
        problem: The benchmark problem.
        x: Primal solution vector.

    Returns:
        Infinity norm of equality constraint residual.
    """
    if problem.A is None or problem.b is None or not problem.cones:
        return 0.0

    Ax = problem.A @ x
    max_residual = 0.0
    row_offset = 0

    for cone_type, cone_dim in problem.cones:
        if cone_type == "zero":
            # Only equality constraints contribute to primal residual
            block = Ax[row_offset : row_offset + cone_dim]
            b_block = problem.b[row_offset : row_offset + cone_dim]
            residual = np.max(np.abs(block - b_block))
            max_residual = max(max_residual, residual)
        # For nonneg, soc, etc., there's slack so no equality residual

        row_offset += cone_dim

    return float(max_residual)


def compute_dual_residual(
    problem: BenchmarkProblem,
    x: np.ndarray,
    y: np.ndarray | None,
) -> float:
    """Compute dual residual ||Px + q + A'y|| for QP.

    For QP: min 0.5 x'Px + q'x s.t. Ax <= b
    KKT stationarity: Px + q + A'y = 0

    Args:
        problem: The benchmark problem.
        x: Primal solution vector.
        y: Dual solution vector (Lagrange multipliers).

    Returns:
        Infinity norm of dual residual.
    """
    # Compute Px + q
    if problem.P is not None:
        grad = problem.P @ x + problem.q
    else:
        grad = problem.q.copy()

    # Add A'y if dual variables available
    if y is not None and problem.A is not None:
        grad = grad + problem.A.T @ y

    return float(np.linalg.norm(grad, ord=np.inf))


def compute_constraint_violation(
    problem: BenchmarkProblem,
    x: np.ndarray,
) -> float:
    """Compute maximum constraint violation.

    Checks each cone block:
    - zero cone: |Ax - b| (equality violation)
    - nonneg cone: max(Ax - b, 0) (inequality violation)
    - soc cone: ||u|| - t where s = (t, u) (SOC violation)

    Args:
        problem: The benchmark problem.
        x: Solution vector.

    Returns:
        Maximum constraint violation across all cones.
    """
    if problem.A is None or problem.b is None or not problem.cones:
        return 0.0

    Ax = problem.A @ x
    max_violation = 0.0
    row_offset = 0

    for cone_type, cone_dim in problem.cones:
        block = Ax[row_offset : row_offset + cone_dim]
        b_block = problem.b[row_offset : row_offset + cone_dim]

        if cone_type == "zero":
            # Equality: Ax = b, violation is |Ax - b|
            violation = np.max(np.abs(block - b_block))
        elif cone_type == "nonneg":
            # Inequality: Ax <= b, violation is max(Ax - b, 0)
            violation = np.max(np.maximum(block - b_block, 0))
        elif cone_type == "soc":
            # SOC: s = b - Ax should satisfy s[0] >= ||s[1:]||
            s = b_block - block
            if len(s) > 0:
                t = s[0]
                u_norm = np.linalg.norm(s[1:]) if len(s) > 1 else 0.0
                violation = max(u_norm - t, 0)
            else:
                violation = 0.0
        else:
            # Unknown cone type, skip
            violation = 0.0

        max_violation = max(max_violation, violation)
        row_offset += cone_dim

    return float(max_violation)


def compute_objective(
    problem: BenchmarkProblem,
    x: np.ndarray,
) -> float:
    """Compute objective value 0.5 x'Px + q'x.

    Args:
        problem: The benchmark problem.
        x: Solution vector.

    Returns:
        Objective value.
    """
    obj = float(problem.q @ x)
    if problem.P is not None:
        obj += 0.5 * float(x @ (problem.P @ x))
    return obj


def compute_duality_gap(
    primal_obj: float,
    dual_obj: float | None,
) -> float:
    """Compute relative duality gap.

    Args:
        primal_obj: Primal objective value.
        dual_obj: Dual objective value.

    Returns:
        Relative gap: |p - d| / (1 + |p|), or inf if dual not available.
    """
    if dual_obj is None:
        return float("inf")
    return abs(primal_obj - dual_obj) / (1.0 + abs(primal_obj))


def validate_solution(
    problem: BenchmarkProblem,
    x: np.ndarray,
    y: np.ndarray | None = None,
    s: np.ndarray | None = None,
    reference_obj: float | None = None,
    tol_feas: float = 1e-6,
    tol_gap: float = 1e-6,
    tol_obj: float = 1e-4,
) -> ValidationResult:
    """Validate a solution against the problem constraints.

    Args:
        problem: The benchmark problem.
        x: Primal solution vector.
        y: Dual solution vector (optional).
        s: Slack variables (optional).
        reference_obj: Known optimal objective for comparison.
        tol_feas: Tolerance for feasibility.
        tol_gap: Tolerance for duality gap.
        tol_obj: Tolerance for objective error vs reference.

    Returns:
        ValidationResult with all computed metrics.
    """
    # Compute residuals
    primal_res = compute_primal_residual(problem, x)
    dual_res = compute_dual_residual(problem, x, y)
    constraint_viol = compute_constraint_violation(problem, x)

    # Compute objectives
    primal_obj = compute_objective(problem, x)

    # Dual objective for QP: L(x,y) = 0.5 x'Px + q'x + y'(Ax - b)
    # At optimum with complementarity, dual_obj should equal primal_obj
    dual_obj = None
    if y is not None and problem.A is not None and problem.b is not None:
        Ax_minus_b = problem.A @ x - problem.b
        dual_obj = primal_obj + float(y @ Ax_minus_b)

    gap = compute_duality_gap(primal_obj, dual_obj)

    # Objective error vs reference
    obj_error = None
    if reference_obj is not None:
        obj_error = abs(primal_obj - reference_obj) / (1.0 + abs(reference_obj))

    # Feasibility and optimality checks
    is_feasible = primal_res < tol_feas and constraint_viol < tol_feas
    is_optimal = is_feasible and (gap < tol_gap or dual_obj is None)

    return ValidationResult(
        primal_residual=primal_res,
        dual_residual=dual_res,
        constraint_violation=constraint_viol,
        gap=gap,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        obj_error=obj_error,
        is_feasible=is_feasible,
        is_optimal=is_optimal,
    )


def format_validation_short(result: ValidationResult) -> str:
    """Format validation result as short string for table display.

    Returns something like "OK" or "!feas" or "~gap".
    """
    if result.is_optimal and result.is_feasible:
        if result.obj_error is not None and result.obj_error > 1e-4:
            return f"~obj({result.obj_error:.0e})"
        return "OK"

    issues = []
    if result.constraint_violation > 1e-6:
        issues.append(f"viol:{result.constraint_violation:.0e}")
    if result.primal_residual > 1e-6:
        issues.append(f"pres:{result.primal_residual:.0e}")
    if result.gap > 1e-6 and result.gap != float("inf"):
        issues.append(f"gap:{result.gap:.0e}")

    return " ".join(issues) if issues else "?"
