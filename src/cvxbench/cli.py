"""CLI for cvxbench using tyro."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

import tyro

from cvxbench.results import SolveResult, display_summary
from cvxbench.runner import run_benchmarks


class Solver(str, Enum):
    """Available solvers."""

    scs = "scs"
    ecos = "ecos"
    minix = "minix"
    clarabel = "clarabel"
    # glpk = "glpk"
    # highs = "highs"


class Suite(str, Enum):
    """Available benchmark suites."""

    maros = "maros"
    smp = "smp"
    sdplib = "sdplib"
    maxcut = "maxcut"
    # qplib = "qplib"
    # miplib = "miplib"


@dataclass
class RunConfig:
    """Run benchmarks with specified configuration.

    Example usage:
        cvxbench --solvers scs ecos --suites maros --sample 0.1 --runs 3
    """

    solvers: Annotated[
        list[Solver],
        tyro.conf.arg(aliases=["-s"]),
    ] = field(default_factory=lambda: [Solver.scs])
    """Solvers to benchmark."""

    suites: Annotated[
        list[Suite],
        tyro.conf.arg(aliases=["-t"]),
    ] = field(default_factory=lambda: [Suite.maros])
    """Test suites to run."""

    sample: float = 1.0
    """Sample rate (0.0 to 1.0). Use 0.01 for 1% of problems."""

    runs: Annotated[int, tyro.conf.arg(aliases=["-r"])] = 1
    """Number of runs per problem."""

    timeout: int = 300
    """Timeout per problem in seconds."""

    output: Annotated[Path | None, tyro.conf.arg(aliases=["-o"])] = None
    """Output file for results (JSON/CSV)."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    """Verbose output."""

    seed: int | None = None
    """Random seed for reproducible sampling."""

    validate: bool = False
    """Validate solutions by computing residuals and constraint violations."""

    save_baseline: str | None = None
    """Save results as baseline with this name."""

    baseline: str | None = None
    """Compare results against this baseline (name or path)."""


@dataclass
class ListSuitesConfig:
    """List available benchmark suites."""

    pass


@dataclass
class ListSolversConfig:
    """List available solvers."""

    pass


@dataclass
class DownloadConfig:
    """Download and cache benchmark data."""

    suite: Suite
    """Suite to download."""


@dataclass
class BaselineListConfig:
    """List saved baselines."""

    pass


@dataclass
class QuickConfig:
    """Run quick benchmark tiers for rapid feedback."""

    tier: Annotated[int, tyro.conf.arg(aliases=["-t"])] = 1
    """Tier level: 1(~10s), 2(~30s), 3(~2min), 4(~5min), 5(~15min)."""

    solvers: Annotated[list[Solver] | None, tyro.conf.arg(aliases=["-s"])] = None
    """Solvers to test (default: scs, clarabel, ecos)."""

    suite: Suite = Suite.maros
    """Benchmark suite to use."""

    validate: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    """Validate solutions (compute residuals)."""

    baseline: Annotated[str | None, tyro.conf.arg(aliases=["-b"])] = None
    """Compare against this baseline."""


def main() -> None:
    """Main entry point for the CLI."""
    config = tyro.cli(  # type: ignore[call-overload]
        Annotated[RunConfig, tyro.conf.subcommand("run", default=True)]
        | Annotated[QuickConfig, tyro.conf.subcommand("quick")]
        | Annotated[ListSuitesConfig, tyro.conf.subcommand("list-suites")]
        | Annotated[ListSolversConfig, tyro.conf.subcommand("list-solvers")]
        | Annotated[DownloadConfig, tyro.conf.subcommand("download")]
        | Annotated[BaselineListConfig, tyro.conf.subcommand("baselines")],
        prog="cvxbench",
        description="CVXBench: Convex Optimization Benchmark Tool",
    )

    if isinstance(config, RunConfig):
        _run_benchmarks(config)
    elif isinstance(config, QuickConfig):
        _run_quick(config)
    elif isinstance(config, ListSuitesConfig):
        _list_suites()
    elif isinstance(config, ListSolversConfig):
        _list_solvers()
    elif isinstance(config, DownloadConfig):
        _download_suite(config)
    elif isinstance(config, BaselineListConfig):
        _list_baselines()


def _run_benchmarks(config: RunConfig) -> None:
    """Run benchmarks with given configuration."""
    from rich.console import Console

    console = Console()

    # Convert enums to strings
    solver_names = [s.value for s in config.solvers]
    suite_names = [s.value for s in config.suites]

    console.print("[bold]CVXBench[/bold] - Convex Optimization Benchmark")
    console.print(f"Solvers: {', '.join(solver_names)}")
    console.print(f"Suites: {', '.join(suite_names)}")
    console.print(f"Sample rate: {config.sample * 100:.1f}%")
    console.print(f"Runs per problem: {config.runs}")
    console.print()

    # Run benchmarks
    results = run_benchmarks(
        solver_names=solver_names,
        suite_names=suite_names,
        sample_rate=config.sample,
        runs_per_problem=config.runs,
        timeout=config.timeout,
        verbose=config.verbose,
        seed=config.seed,
        validate=config.validate,
    )

    # Display results
    display_summary(results, console)

    # Compare to baseline if requested
    if config.baseline:
        from cvxbench.baseline import compare_to_baseline, display_comparison, load_baseline

        try:
            baseline = load_baseline(config.baseline)
            report = compare_to_baseline(results, baseline)
            display_comparison(report, console)
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")

    # Save as baseline if requested
    if config.save_baseline:
        from cvxbench.baseline import save_baseline

        path = save_baseline(results, config.save_baseline)
        console.print(f"\n[green]Baseline saved to {path}[/green]")

    # Save to file if requested
    if config.output:
        _save_results(results, config.output)
        console.print(f"\nResults saved to {config.output}")


def _run_quick(config: QuickConfig) -> None:
    """Run quick benchmark tier."""
    from cvxbench.quick import run_tier

    # Convert enum list to strings if provided
    solver_names = None
    if config.solvers is not None:
        solver_names = [s.value for s in config.solvers]

    run_tier(
        tier=config.tier,
        solvers=solver_names,
        suite=config.suite.value,
        validate=config.validate,
        baseline=config.baseline,
    )


def _list_suites() -> None:
    """List available benchmark suites."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Benchmark Suites")
    table.add_column("Name", style="cyan")
    table.add_column("Problems", justify="right")
    table.add_column("Description")

    suites = [
        ("maros", "138", "Maros-Mészáros QP test set (classic QPs)"),
        ("smp", "1515", "SMP/NASOQ QP repository (graphics/simulation)"),
        ("qplib", "~134", "QPLIB continuous subset"),
        ("miplib", "varies", "MIPLIB mixed-integer benchmarks"),
    ]

    for name, count, desc in suites:
        table.add_row(name, count, desc)

    console.print(table)


def _list_solvers() -> None:
    """List available solvers."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Solvers")
    table.add_column("Name", style="cyan")
    table.add_column("LP", justify="center")
    table.add_column("QP", justify="center")
    table.add_column("SOCP", justify="center")
    table.add_column("SDP", justify="center")
    table.add_column("MIP", justify="center")

    solvers = [
        ("scs", "✓", "✓", "✓", "✓", "✗"),
        ("ecos", "✓", "✗", "✓", "✗", "✗"),
        ("clarabel", "✓", "✓", "✓", "✓", "✗"),
        ("minix", "✓", "✓", "✓", "—", "—"),
        ("glpk", "✓", "✗", "✗", "✗", "✓"),
        ("highs", "✓", "✓", "✗", "✗", "✓"),
    ]

    for row in solvers:
        table.add_row(*row)

    console.print(table)


def _download_suite(config: DownloadConfig) -> None:
    """Download benchmark suite data."""
    from rich.console import Console

    console = Console()
    suite_name = config.suite.value

    console.print(f"Downloading {suite_name} benchmark suite...")

    if suite_name == "maros":
        from cvxbench.loaders.maros_meszaros import MarosMeszarosLoader

        maros_loader = MarosMeszarosLoader()
        problems = maros_loader.list_problems()

        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task("Downloading...", total=len(problems))
            for name in problems:
                try:
                    maros_loader.load_problem(name)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to download {name}: {e}[/yellow]")
                progress.update(task, advance=1)

        console.print(f"[green]Downloaded {len(problems)} to {maros_loader.cache_dir}[/green]")
    elif suite_name == "smp":
        from cvxbench.loaders.smp import SMPLoader

        smp_loader = SMPLoader()
        console.print("Downloading SMP problems from Google Drive...")
        smp_loader.ensure_downloaded()
        problems = smp_loader.list_problems()
        console.print(f"[green]Downloaded {len(problems)} to {smp_loader.cache_dir}[/green]")
    else:
        console.print(f"[red]Suite {suite_name} not yet implemented[/red]")


def _save_results(results: list[SolveResult], output: Path) -> None:
    """Save results to file."""
    import json


    if output.suffix == ".json":
        data = [
            {
                "problem": r.problem,
                "source": r.source,
                "solver": r.solver,
                "run": r.run_index,
                "status": r.status,
                "time_sec": r.time_sec,
                "iterations": r.iterations,
                "primal_obj": r.primal_obj,
            }
            for r in results
        ]
        output.write_text(json.dumps(data, indent=2))
    elif output.suffix == ".csv":
        import csv

        with output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["problem", "source", "solver", "run", "status", "time_sec", "iterations", "obj"]
            )
            for r in results:
                writer.writerow(
                    [
                        r.problem,
                        r.source,
                        r.solver,
                        r.run_index,
                        r.status,
                        r.time_sec,
                        r.iterations,
                        r.primal_obj,
                    ]
                )
    else:
        # Default to JSON
        _save_results(results, output.with_suffix(".json"))


def _list_baselines() -> None:
    """List saved baselines."""
    from rich.console import Console
    from rich.table import Table

    from cvxbench.baseline import list_baselines

    console = Console()
    baselines = list_baselines()

    if not baselines:
        console.print("[yellow]No baselines found.[/yellow]")
        console.print("Save a baseline with: cvxbench run --save-baseline NAME")
        return

    table = Table(title="Saved Baselines")
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Problems", justify="right")

    for name, created, count in baselines:
        # Format date nicely
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(created)
            created_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            created_str = created[:16]

        table.add_row(name, created_str, str(count))

    console.print(table)
    console.print("\nCompare with: cvxbench run --baseline NAME")


if __name__ == "__main__":
    main()
