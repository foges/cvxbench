"""Baseline management for regression tracking.

Save benchmark results as baselines and compare new runs against them
to detect performance regressions or improvements.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

from cvxbench.results import SolveResult

# Default location for baselines
DEFAULT_BASELINE_DIR = Path.home() / ".cache" / "cvxbench" / "baselines"


@dataclass
class BaselineEntry:
    """Single problem result in a baseline."""

    problem: str
    source: str
    solver: str
    time_sec: float
    primal_obj: float | None
    status: str
    primal_residual: float | None = None
    constraint_violation: float | None = None


@dataclass
class Baseline:
    """Complete baseline for comparison."""

    name: str
    created: str
    solver: str
    entries: dict[str, BaselineEntry] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def add_entry(self, result: SolveResult) -> None:
        """Add a result to the baseline."""
        key = f"{result.source}:{result.problem}"
        self.entries[key] = BaselineEntry(
            problem=result.problem,
            source=result.source,
            solver=result.solver,
            time_sec=result.time_sec,
            primal_obj=result.primal_obj,
            status=result.status,
            primal_residual=result.primal_residual,
            constraint_violation=result.constraint_violation,
        )


@dataclass
class ProblemComparison:
    """Comparison result for a single problem."""

    problem: str
    source: str
    baseline_time: float
    current_time: float
    time_ratio: float  # current / baseline
    baseline_status: str
    current_status: str
    baseline_obj: float | None
    current_obj: float | None
    obj_match: bool
    status_change: str  # "same", "improved", "regressed"


@dataclass
class ComparisonReport:
    """Summary of baseline comparison."""

    baseline_name: str
    n_problems: int
    n_faster: int
    n_slower: int
    n_same: int
    n_status_improved: int
    n_status_regressed: int
    time_ratio_geom: float  # Geometric mean of time ratios
    details: list[ProblemComparison] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """Check if there are any status regressions."""
        return self.n_status_regressed > 0


def save_baseline(
    results: list[SolveResult],
    name: str,
    path: Path | None = None,
    solver: str | None = None,
) -> Path:
    """Save results as a baseline.

    Args:
        results: List of solve results.
        name: Name for this baseline.
        path: Output path (default: ~/.cache/cvxbench/baselines/{name}.json).
        solver: Solver name (inferred from results if not specified).

    Returns:
        Path where baseline was saved.
    """
    if path is None:
        DEFAULT_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        path = DEFAULT_BASELINE_DIR / f"{name}.json"

    # Infer solver from results
    if solver is None:
        solvers = set(r.solver for r in results)
        solver = ",".join(sorted(solvers))

    baseline = Baseline(
        name=name,
        created=datetime.now().isoformat(),
        solver=solver,
    )

    for result in results:
        if result.is_success:  # Only store successful results
            baseline.add_entry(result)

    # Serialize
    data = {
        "name": baseline.name,
        "created": baseline.created,
        "solver": baseline.solver,
        "metadata": baseline.metadata,
        "entries": {k: asdict(v) for k, v in baseline.entries.items()},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

    return path


def load_baseline(path_or_name: Path | str) -> Baseline:
    """Load a baseline from file.

    Args:
        path_or_name: Path to baseline file, or name to look up in default dir.

    Returns:
        Loaded baseline.
    """
    if isinstance(path_or_name, str):
        # Try as name first
        default_path = DEFAULT_BASELINE_DIR / f"{path_or_name}.json"
        if default_path.exists():
            path = default_path
        else:
            path = Path(path_or_name)
    else:
        path = path_or_name

    if not path.exists():
        msg = f"Baseline not found: {path}"
        raise FileNotFoundError(msg)

    data = json.loads(path.read_text())

    baseline = Baseline(
        name=data["name"],
        created=data["created"],
        solver=data["solver"],
        metadata=data.get("metadata", {}),
    )

    for key, entry_data in data["entries"].items():
        baseline.entries[key] = BaselineEntry(**entry_data)

    return baseline


def list_baselines() -> list[tuple[str, str, int]]:
    """List available baselines.

    Returns:
        List of (name, created, n_entries) tuples.
    """
    if not DEFAULT_BASELINE_DIR.exists():
        return []

    baselines = []
    for path in DEFAULT_BASELINE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            baselines.append((
                data["name"],
                data["created"],
                len(data.get("entries", {})),
            ))
        except Exception:
            pass

    return sorted(baselines, key=lambda x: x[1], reverse=True)


def compare_to_baseline(
    results: list[SolveResult],
    baseline: Baseline,
    time_threshold: float = 0.1,  # 10% time change
) -> ComparisonReport:
    """Compare current results to a baseline.

    Args:
        results: Current benchmark results.
        baseline: Baseline to compare against.
        time_threshold: Threshold for considering time change significant.

    Returns:
        Comparison report.
    """
    import math

    comparisons = []
    time_ratios = []

    n_faster = 0
    n_slower = 0
    n_same = 0
    n_status_improved = 0
    n_status_regressed = 0

    for result in results:
        key = f"{result.source}:{result.problem}"
        if key not in baseline.entries:
            continue

        entry = baseline.entries[key]

        # Time comparison
        if entry.time_sec > 0:
            time_ratio = result.time_sec / entry.time_sec
        else:
            time_ratio = 1.0

        if time_ratio < 1 - time_threshold:
            n_faster += 1
        elif time_ratio > 1 + time_threshold:
            n_slower += 1
        else:
            n_same += 1

        # Status comparison
        baseline_success = entry.status in ("optimal", "optimal_inaccurate")
        current_success = result.is_success

        if current_success and not baseline_success:
            status_change = "improved"
            n_status_improved += 1
        elif not current_success and baseline_success:
            status_change = "regressed"
            n_status_regressed += 1
        else:
            status_change = "same"

        # Objective comparison
        obj_match = True
        if entry.primal_obj is not None and result.primal_obj is not None:
            rel_diff = abs(entry.primal_obj - result.primal_obj) / (1 + abs(entry.primal_obj))
            obj_match = rel_diff < 1e-4

        time_ratios.append(time_ratio)

        comparisons.append(ProblemComparison(
            problem=result.problem,
            source=result.source,
            baseline_time=entry.time_sec,
            current_time=result.time_sec,
            time_ratio=time_ratio,
            baseline_status=entry.status,
            current_status=result.status,
            baseline_obj=entry.primal_obj,
            current_obj=result.primal_obj,
            obj_match=obj_match,
            status_change=status_change,
        ))

    # Geometric mean of time ratios
    if time_ratios:
        log_sum = sum(math.log(r) for r in time_ratios)
        time_ratio_geom = math.exp(log_sum / len(time_ratios))
    else:
        time_ratio_geom = 1.0

    return ComparisonReport(
        baseline_name=baseline.name,
        n_problems=len(comparisons),
        n_faster=n_faster,
        n_slower=n_slower,
        n_same=n_same,
        n_status_improved=n_status_improved,
        n_status_regressed=n_status_regressed,
        time_ratio_geom=time_ratio_geom,
        details=comparisons,
    )


def display_comparison(report: ComparisonReport, console: Console) -> None:
    """Display baseline comparison as rich table.

    Args:
        report: Comparison report.
        console: Rich console for output.
    """
    from rich.table import Table

    # Summary
    console.print()
    console.print(f"[bold]Baseline Comparison: {report.baseline_name}[/bold]")
    console.print(f"Problems compared: {report.n_problems}")
    console.print()

    # Status summary
    status_table = Table(title="Status Changes", show_header=True, header_style="bold")
    status_table.add_column("Metric", style="cyan")
    status_table.add_column("Count", justify="right")

    if report.n_status_improved > 0:
        improved = f"[green]{report.n_status_improved}[/green]"
        status_table.add_row("Improved (fail -> pass)", improved)
    if report.n_status_regressed > 0:
        status_table.add_row("Regressed (pass -> fail)", f"[red]{report.n_status_regressed}[/red]")

    if report.n_status_improved > 0 or report.n_status_regressed > 0:
        console.print(status_table)
        console.print()

    # Timing summary
    time_table = Table(title="Timing Changes", show_header=True, header_style="bold")
    time_table.add_column("Metric", style="cyan")
    time_table.add_column("Value", justify="right")

    if report.time_ratio_geom < 1.0:
        ratio_color = "green"
    elif report.time_ratio_geom > 1.1:
        ratio_color = "red"
    else:
        ratio_color = "yellow"
    ratio_str = f"[{ratio_color}]{report.time_ratio_geom:.2f}x[/{ratio_color}]"
    time_table.add_row("Geom Mean Ratio", ratio_str)
    time_table.add_row("Faster (>10%)", f"[green]{report.n_faster}[/green]")
    slower_str = f"[red]{report.n_slower}[/red]" if report.n_slower > 0 else "0"
    time_table.add_row("Slower (>10%)", slower_str)
    time_table.add_row("Same (±10%)", str(report.n_same))

    console.print(time_table)

    # Details for notable changes
    notable = [
        c for c in report.details
        if c.status_change != "same" or abs(c.time_ratio - 1.0) > 0.2
    ]

    if notable:
        console.print()
        detail_table = Table(title="Notable Changes", show_header=True, header_style="bold yellow")
        detail_table.add_column("Problem", style="cyan")
        detail_table.add_column("Status", justify="center")
        detail_table.add_column("Time Ratio", justify="right")
        detail_table.add_column("Baseline", justify="right")
        detail_table.add_column("Current", justify="right")

        for c in sorted(notable, key=lambda x: -abs(x.time_ratio - 1.0))[:10]:
            status_str = {
                "improved": "[green]FIXED[/green]",
                "regressed": "[red]BROKEN[/red]",
                "same": "",
            }[c.status_change]

            ratio_str = f"{c.time_ratio:.2f}x"
            if c.time_ratio < 0.8:
                ratio_str = f"[green]{ratio_str}[/green]"
            elif c.time_ratio > 1.2:
                ratio_str = f"[red]{ratio_str}[/red]"

            detail_table.add_row(
                c.problem,
                status_str,
                ratio_str,
                f"{c.baseline_time * 1000:.1f}ms",
                f"{c.current_time * 1000:.1f}ms",
            )

        console.print(detail_table)
