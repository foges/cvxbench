"""Result aggregation and display for cvxbench."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class SolveResult:
    """Result from solving a single benchmark problem."""

    problem: str
    source: str
    solver: str
    run_index: int
    status: str  # "optimal", "infeasible", "unbounded", "error", etc.
    time_sec: float
    iterations: int | None
    primal_obj: float | None
    dual_obj: float | None
    gap: float | None
    error_message: str | None = None

    # Validation fields (populated when validation is enabled)
    primal_residual: float | None = None
    dual_residual: float | None = None
    constraint_violation: float | None = None
    obj_error: float | None = None  # Error vs reference solution
    is_validated: bool = False

    @property
    def is_success(self) -> bool:
        """Check if the solve was successful."""
        return self.status in ("optimal", "optimal_inaccurate")

    @property
    def is_accurate(self) -> bool:
        """Check if solution passes validation (if validated)."""
        if not self.is_validated:
            return self.is_success
        # Check feasibility
        tol = 1e-6
        if self.primal_residual is not None and self.primal_residual > tol:
            return False
        if self.constraint_violation is not None and self.constraint_violation > tol:
            return False
        return self.is_success


def shifted_geometric_mean(times: list[float], shift: float = 1.0) -> float:
    """Compute shifted geometric mean.

    The shifted geometric mean is defined as:
        exp(mean(log(t + shift))) - shift

    This is more robust to outliers than the arithmetic mean,
    and the shift prevents issues with zero values.

    Args:
        times: List of solve times.
        shift: Shift parameter (default 1.0).

    Returns:
        Shifted geometric mean.
    """
    if not times:
        return 0.0
    shifted = [t + shift for t in times]
    log_mean = sum(math.log(t) for t in shifted) / len(shifted)
    return math.exp(log_mean) - shift


def compute_summary_stats(results: list[SolveResult], solver: str) -> dict[str, Any]:
    """Compute summary statistics for a solver.

    Args:
        results: All solve results.
        solver: Solver name to compute stats for.

    Returns:
        Dictionary of statistics.
    """
    solver_results = [r for r in results if r.solver == solver]
    if not solver_results:
        return {}

    successes = [r for r in solver_results if r.is_success]
    failures = [r for r in solver_results if not r.is_success]
    success_times = [r.time_sec for r in successes]

    stats: dict[str, Any] = {
        "total": len(solver_results),
        "success": len(successes),
        "failure": len(failures),
        "success_rate": len(successes) / len(solver_results) if solver_results else 0.0,
    }

    if success_times:
        stats["mean_time"] = np.mean(success_times)
        stats["median_time"] = np.median(success_times)
        stats["std_time"] = np.std(success_times)
        stats["min_time"] = np.min(success_times)
        stats["max_time"] = np.max(success_times)
        stats["geom_mean"] = shifted_geometric_mean(success_times)
    else:
        stats["mean_time"] = float("nan")
        stats["median_time"] = float("nan")
        stats["std_time"] = float("nan")
        stats["min_time"] = float("nan")
        stats["max_time"] = float("nan")
        stats["geom_mean"] = float("nan")

    # Validation statistics
    validated = [r for r in successes if r.is_validated]
    if validated:
        primal_residuals = [
            r.primal_residual for r in validated if r.primal_residual is not None
        ]
        constraint_viols = [
            r.constraint_violation for r in validated if r.constraint_violation is not None
        ]

        stats["validated"] = len(validated)
        stats["max_primal_res"] = max(primal_residuals) if primal_residuals else None
        stats["max_constraint_viol"] = max(constraint_viols) if constraint_viols else None

        # Count accurate solutions (pass validation)
        accurate = [r for r in validated if r.is_accurate]
        stats["accurate"] = len(accurate)
    else:
        stats["validated"] = 0
        stats["max_primal_res"] = None
        stats["max_constraint_viol"] = None
        stats["accurate"] = None

    return stats


def display_summary(results: list[SolveResult], console: Console) -> None:
    """Display a rich summary of benchmark results.

    Args:
        results: List of all solve results.
        console: Rich console for output.
    """
    from rich.table import Table

    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Get unique solvers
    solvers = sorted(set(r.solver for r in results))

    # Check if any results have validation
    has_validation = any(r.is_validated for r in results)

    # Summary table
    summary_table = Table(title="Benchmark Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Solver", style="cyan")
    summary_table.add_column("Success", justify="right")
    summary_table.add_column("Failed", justify="right", style="red")
    summary_table.add_column("Rate", justify="right")
    summary_table.add_column("Median (s)", justify="right")
    summary_table.add_column("Geom Mean (s)", justify="right")

    # Add validation columns if validation was performed
    if has_validation:
        summary_table.add_column("Accurate", justify="right")
        summary_table.add_column("MaxRes", justify="right")

    for solver in solvers:
        stats = compute_summary_stats(results, solver)
        if not stats:
            continue

        rate_str = f"{stats['success_rate'] * 100:.1f}%"
        if stats["success_rate"] >= 0.9:
            rate_style = "green"
        elif stats["success_rate"] >= 0.7:
            rate_style = "yellow"
        else:
            rate_style = "red"

        row = [
            solver,
            str(stats["success"]),
            str(stats["failure"]),
            f"[{rate_style}]{rate_str}[/{rate_style}]",
            f"{stats['median_time']:.4f}" if not math.isnan(stats["median_time"]) else "-",
            f"{stats['geom_mean']:.4f}" if not math.isnan(stats["geom_mean"]) else "-",
        ]

        if has_validation:
            # Accurate count
            if stats.get("accurate") is not None:
                acc_str = f"{stats['accurate']}/{stats['validated']}"
                if stats["accurate"] == stats["validated"]:
                    acc_str = f"[green]{acc_str}[/green]"
                elif stats["accurate"] < stats["validated"]:
                    acc_str = f"[yellow]{acc_str}[/yellow]"
                row.append(acc_str)
            else:
                row.append("-")

            # Max residual
            max_res = stats.get("max_primal_res")
            if max_res is not None:
                if max_res < 1e-8:
                    res_str = f"[green]{max_res:.0e}[/green]"
                elif max_res < 1e-4:
                    res_str = f"[yellow]{max_res:.0e}[/yellow]"
                else:
                    res_str = f"[red]{max_res:.0e}[/red]"
                row.append(res_str)
            else:
                row.append("-")

        summary_table.add_row(*row)

    console.print()
    console.print(summary_table)

    # Failure details if any
    failures = [r for r in results if not r.is_success]
    if failures:
        console.print()
        failure_table = Table(
            title=f"Failures ({len(failures)} total)", show_header=True, header_style="bold red"
        )
        failure_table.add_column("Problem", style="yellow")
        failure_table.add_column("Solver")
        failure_table.add_column("Status")
        failure_table.add_column("Error")

        # Show at most 10 failures
        for r in failures[:10]:
            if r.error_message and len(r.error_message) > 50:
                error_msg = r.error_message[:50] + "..."
            else:
                error_msg = r.error_message or "—"
            failure_table.add_row(r.problem, r.solver, r.status, error_msg)

        if len(failures) > 10:
            failure_table.add_row("...", "...", "...", f"({len(failures) - 10} more)")

        console.print(failure_table)

    # Performance comparison if multiple solvers
    if len(solvers) > 1:
        console.print()
        _display_performance_comparison(results, solvers, console)


def _display_performance_comparison(
    results: list[SolveResult],
    solvers: list[str],
    console: Console,
) -> None:
    """Display performance comparison between solvers.

    Args:
        results: All solve results.
        solvers: List of solver names.
        console: Rich console for output.
    """
    from rich.table import Table

    # Find problems solved by all solvers
    problems = set(r.problem for r in results)
    common_problems = set()

    for problem in problems:
        problem_results = [r for r in results if r.problem == problem and r.is_success]
        solvers_solved = set(r.solver for r in problem_results)
        if solvers_solved == set(solvers):
            common_problems.add(problem)

    if not common_problems:
        console.print("[yellow]No problems solved by all solvers for comparison[/yellow]")
        return

    # Compute relative performance
    table = Table(
        title=f"Performance Comparison ({len(common_problems)} common problems)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Solver", style="cyan")
    table.add_column("Geom Mean (s)", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Relative", justify="right")

    # Collect times for common problems
    solver_times: dict[str, list[float]] = {s: [] for s in solvers}
    wins: dict[str, int] = {s: 0 for s in solvers}

    for problem in common_problems:
        problem_times = {}
        for solver in solvers:
            r = next(
                (
                    r for r in results
                    if r.problem == problem and r.solver == solver and r.is_success
                ),
                None,
            )
            if r:
                problem_times[solver] = r.time_sec
                solver_times[solver].append(r.time_sec)

        if problem_times:
            fastest = min(problem_times.values())
            for solver, t in problem_times.items():
                if abs(t - fastest) < 1e-9:
                    wins[solver] += 1

    # Compute geometric means
    geom_means = {s: shifted_geometric_mean(times) for s, times in solver_times.items()}
    min_geom = min(geom_means.values()) if geom_means else 1.0

    for solver in solvers:
        gm = geom_means.get(solver, float("nan"))
        relative = gm / min_geom if min_geom > 0 else float("nan")
        relative_str = f"{relative:.2f}x" if not math.isnan(relative) else "—"

        # Highlight best
        if abs(gm - min_geom) < 1e-9:
            relative_str = f"[bold green]{relative_str}[/bold green]"

        table.add_row(
            solver,
            f"{gm:.4f}" if not math.isnan(gm) else "—",
            str(wins.get(solver, 0)),
            relative_str,
        )

    console.print(table)
