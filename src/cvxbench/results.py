"""Result aggregation and display for cvxbench."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

    @property
    def is_success(self) -> bool:
        """Check if the solve was successful."""
        return self.status in ("optimal", "optimal_inaccurate")


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


def compute_summary_stats(results: list[SolveResult], solver: str) -> dict:
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

    stats = {
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

    return stats


def display_summary(results: list[SolveResult], console: Console) -> None:
    """Display a rich summary of benchmark results.

    Args:
        results: List of all solve results.
        console: Rich console for output.
    """
    from rich.panel import Panel
    from rich.table import Table

    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Get unique solvers
    solvers = sorted(set(r.solver for r in results))

    # Summary table
    summary_table = Table(title="Benchmark Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Solver", style="cyan")
    summary_table.add_column("Success", justify="right")
    summary_table.add_column("Failed", justify="right", style="red")
    summary_table.add_column("Rate", justify="right")
    summary_table.add_column("Median (s)", justify="right")
    summary_table.add_column("Geom Mean (s)", justify="right")
    summary_table.add_column("Min (s)", justify="right")
    summary_table.add_column("Max (s)", justify="right")

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

        summary_table.add_row(
            solver,
            str(stats["success"]),
            str(stats["failure"]),
            f"[{rate_style}]{rate_str}[/{rate_style}]",
            f"{stats['median_time']:.4f}" if not math.isnan(stats["median_time"]) else "—",
            f"{stats['geom_mean']:.4f}" if not math.isnan(stats["geom_mean"]) else "—",
            f"{stats['min_time']:.4f}" if not math.isnan(stats["min_time"]) else "—",
            f"{stats['max_time']:.4f}" if not math.isnan(stats["max_time"]) else "—",
        )

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
            error_msg = r.error_message[:50] + "..." if r.error_message and len(r.error_message) > 50 else (r.error_message or "—")
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
                (r for r in results if r.problem == problem and r.solver == solver and r.is_success),
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
