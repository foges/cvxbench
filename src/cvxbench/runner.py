"""Benchmark runner for cvxbench."""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from cvxbench.loaders.base import BenchmarkLoader, BenchmarkProblem
from cvxbench.results import SolveResult


def get_loader(suite_name: str) -> BenchmarkLoader:
    """Get a loader for the given suite name."""
    if suite_name == "maros":
        from cvxbench.loaders.maros_meszaros import MarosMeszarosLoader

        return MarosMeszarosLoader()
    elif suite_name == "smp":
        from cvxbench.loaders.smp import SMPLoader

        return SMPLoader()
    else:
        msg = f"Unknown suite: {suite_name}"
        raise ValueError(msg)


def get_solver_name(solver: str) -> str | object:
    """Map solver name to CVXPY solver constant or instance."""
    mapping: dict[str, str | object] = {
        "scs": cp.SCS,
        "ecos": cp.ECOS,
        "osqp": cp.OSQP,
        "clarabel": cp.CLARABEL,
        "glpk": cp.GLPK,
        "highs": cp.SCIPY,  # Uses scipy with HiGHS
    }

    # Special case: minix requires instantiation
    if solver == "minix":
        from minix.cvxpy_backend import MINIX
        return MINIX()

    return mapping.get(solver, solver.upper())


def run_benchmarks(
    solver_names: list[str],
    suite_names: list[str],
    sample_rate: float = 1.0,
    runs_per_problem: int = 1,
    timeout: int = 300,
    verbose: bool = False,
    seed: int | None = None,
    validate: bool = False,
) -> list[SolveResult]:
    """Run benchmarks and collect results.

    Args:
        solver_names: List of solver names (e.g., ["scs", "ecos"]).
        suite_names: List of suite names (e.g., ["maros"]).
        sample_rate: Fraction of problems to run (0.0 to 1.0).
        runs_per_problem: Number of times to solve each problem.
        timeout: Timeout per problem in seconds.
        verbose: Whether to print solver output.
        seed: Random seed for reproducible sampling.
        validate: Whether to validate solutions (compute residuals).

    Returns:
        List of SolveResult objects.
    """
    results: list[SolveResult] = []

    # Load all problems
    problems: list[BenchmarkProblem] = []
    for suite_name in suite_names:
        loader = get_loader(suite_name)
        for problem in loader.iterate_problems(sample_rate=sample_rate, seed=seed):
            problems.append(problem)

    total_runs = len(problems) * len(solver_names) * runs_per_problem

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=total_runs)

        for problem in problems:
            for solver_name in solver_names:
                for run_idx in range(runs_per_problem):
                    progress.update(
                        task,
                        description=f"{problem.name} ({solver_name})",
                    )

                    result = run_single(
                        problem=problem,
                        solver_name=solver_name,
                        run_index=run_idx,
                        timeout=timeout,
                        verbose=verbose,
                        validate=validate,
                    )
                    results.append(result)

                    progress.advance(task)

    return results


def run_single(
    problem: BenchmarkProblem,
    solver_name: str,
    run_index: int,
    timeout: int,
    verbose: bool,
    validate: bool = False,
) -> SolveResult:
    """Run a single benchmark problem with a single solver.

    Args:
        problem: The benchmark problem to solve.
        solver_name: Name of the solver to use.
        run_index: Index of this run (for repeated runs).
        timeout: Timeout in seconds.
        verbose: Whether to print solver output.
        validate: Whether to validate solutions (compute residuals).

    Returns:
        SolveResult with timing and solution information.
    """
    try:
        # Build CVXPY problem
        cvxpy_problem, x = build_cvxpy_problem(problem)

        # Get solver
        solver = get_solver_name(solver_name)

        # Solve with timing
        # Build solver options based on solver type
        solver_opts: dict[str, object] = {}
        if solver_name in ("scs", "ecos"):
            solver_opts["max_iters"] = 100000

        start_time = time.perf_counter()
        try:
            cvxpy_problem.solve(
                solver=solver,
                verbose=verbose,
                **solver_opts,
            )
            elapsed = time.perf_counter() - start_time
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return SolveResult(
                problem=problem.name,
                source=problem.source,
                solver=solver_name,
                run_index=run_index,
                status="error",
                time_sec=elapsed,
                iterations=None,
                primal_obj=None,
                dual_obj=None,
                gap=None,
                error_message=str(e),
            )

        # Extract results
        status = cvxpy_problem.status
        obj_val = cvxpy_problem.value if cvxpy_problem.value is not None else None

        # Try to get solver stats
        iterations = None
        try:
            if hasattr(cvxpy_problem, "solver_stats") and cvxpy_problem.solver_stats:
                stats = cvxpy_problem.solver_stats
                if hasattr(stats, "num_iters"):
                    iterations = stats.num_iters
        except Exception:
            pass

        # Validation (if requested and solution available)
        primal_residual = None
        dual_residual = None
        constraint_violation = None
        obj_error = None
        is_validated = False

        if validate and x.value is not None:
            from cvxbench.validation import validate_solution

            x_val = np.array(x.value).flatten()

            # Try to get dual variables from constraints
            y_val = None
            try:
                if cvxpy_problem.constraints:
                    duals = []
                    for c in cvxpy_problem.constraints:
                        if c.dual_value is not None:
                            d = np.array(c.dual_value).flatten()
                            duals.append(d)
                    if duals:
                        y_val = np.concatenate(duals)
            except Exception:
                pass

            validation = validate_solution(
                problem,
                x_val,
                y=y_val,
                reference_obj=problem.known_optimal,
            )

            primal_residual = validation.primal_residual
            dual_residual = validation.dual_residual
            constraint_violation = validation.constraint_violation
            obj_error = validation.obj_error
            is_validated = True

        return SolveResult(
            problem=problem.name,
            source=problem.source,
            solver=solver_name,
            run_index=run_index,
            status=status,
            time_sec=elapsed,
            iterations=iterations,
            primal_obj=obj_val,
            dual_obj=None,
            gap=None,
            error_message=None,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            constraint_violation=constraint_violation,
            obj_error=obj_error,
            is_validated=is_validated,
        )

    except Exception as e:
        return SolveResult(
            problem=problem.name,
            source=problem.source,
            solver=solver_name,
            run_index=run_index,
            status="error",
            time_sec=0.0,
            iterations=None,
            primal_obj=None,
            dual_obj=None,
            gap=None,
            error_message=str(e),
        )


def build_cvxpy_problem(problem: BenchmarkProblem) -> tuple[cp.Problem, cp.Variable]:
    """Convert a BenchmarkProblem to a CVXPY problem.

    The problem is in conic form:
        minimize    (1/2) x^T P x + q^T x
        subject to  A x + s = b
                    s ∈ K

    Args:
        problem: The benchmark problem.

    Returns:
        Tuple of (CVXPY Problem, variable x).
    """
    n = problem.n_vars

    # Create variable
    x = cp.Variable(n)

    # Build objective
    if problem.P is not None:
        # Quadratic objective: 0.5 x'Px + q'x
        objective = 0.5 * cp.quad_form(x, problem.P) + problem.q @ x
    else:
        # Linear objective: q'x
        objective = problem.q @ x

    # Build constraints from conic form
    # A x + s = b, s ∈ K
    # This means: for each cone block, we have constraints on A_i x
    constraints: list[cp.Constraint] = []

    row_offset = 0
    for cone_type, cone_dim in problem.cones:
        # Extract rows for this cone
        A_block = problem.A[row_offset : row_offset + cone_dim, :]
        b_block = problem.b[row_offset : row_offset + cone_dim]

        if cone_type == "zero":
            # Zero cone: A_i x = b_i
            constraints.append(A_block @ x == b_block)
        elif cone_type == "nonneg":
            # Nonnegative cone: A_i x + s = b_i, s >= 0
            # => A_i x <= b_i
            # But in standard conic form, s = b - Ax, s >= 0 means b - Ax >= 0
            constraints.append(A_block @ x <= b_block)
        elif cone_type == "soc":
            # SOC: (t, u) in SOC means t >= ||u||
            # For Ax + s = b with s in SOC:
            # s = b - Ax, and s[0] >= ||s[1:]||
            # This is: b[0] - (Ax)[0] >= ||b[1:] - (Ax)[1:]||
            s = b_block - A_block @ x
            t = s[0]
            u = s[1:]
            constraints.append(cp.norm(u, 2) <= t)
        elif cone_type == "psd":
            # PSD cone: s in S_+
            # s = b - Ax is a symmetric matrix (in svec form)
            # For simplicity, skip PSD constraints for now
            pass
        elif cone_type == "exp":
            # Exponential cone
            # For simplicity, skip exp constraints for now
            pass

        row_offset += cone_dim

    cvxpy_problem = cp.Problem(cp.Minimize(objective), constraints)
    return cvxpy_problem, x
