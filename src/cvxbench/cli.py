"""CLI for cvxbench using tyro."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import tyro

if TYPE_CHECKING:
    from rich.console import Console

from cvxbench.results import SolveResult, display_summary
from cvxbench.runner import run_benchmarks


class Solver(str, Enum):
    """Available solvers."""

    scs = "scs"
    ecos = "ecos"
    minix = "minix"
    clarabel = "clarabel"


# Default solvers for benchmarking (3 solvers for good comparison)
DEFAULT_SOLVERS = ["scs", "clarabel", "ecos"]


def format_time(seconds: float) -> str:
    """Format time with appropriate units.

    - Sub-second: show in milliseconds (e.g., 123.4ms)
    - Multi-second: show in seconds (e.g., 12.34s)
    - Multi-minute: show in minutes (e.g., 2m15s)
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.0f}s"


@dataclass
class BenchConfig:
    """Run benchmarks on convex optimization problems.

    Examples:
        cvxbench bench                      Quick sanity check (~10s)
        cvxbench bench -t 3                 Medium benchmark (~2min)
        cvxbench bench --full               Full benchmark suite
        cvxbench bench -s scs minix         Compare specific solvers
        cvxbench bench --baseline v1.0      Compare against baseline
    """

    tier: Annotated[int, tyro.conf.arg(aliases=["-t"])] = 1
    """Tier level 1-5: 1(~10s), 2(~30s), 3(~2min), 4(~5min), 5(~15min)."""

    full: bool = False
    """Run all problems in selected suites (overrides --tier)."""

    solvers: Annotated[
        list[str],
        tyro.conf.arg(aliases=["-s"]),
    ] = field(default_factory=lambda: ["scs", "clarabel", "ecos"])
    """Solvers to benchmark (scs, clarabel, ecos, minix)."""

    suites: list[str] = field(default_factory=lambda: ["maros"])
    """Benchmark suites to use (maros, smp, sdplib, maxcut)."""

    runs: Annotated[int, tyro.conf.arg(aliases=["-r"])] = 1
    """Number of runs per problem."""

    timeout: int = 300
    """Timeout per problem in seconds."""

    sample: float = 1.0
    """Sample rate 0.0-1.0 (only applies with --full)."""

    seed: int | None = None
    """Random seed for reproducible sampling."""

    validate: bool = False
    """Validate solutions by computing residuals."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    """Show detailed solver output."""

    quiet: bool = False
    """Show only final summary."""

    baseline: Annotated[str | None, tyro.conf.arg(aliases=["-b"])] = None
    """Compare results against this baseline."""

    save: str | None = None
    """Save results as new baseline with this name."""

    output: Annotated[Path | None, tyro.conf.arg(aliases=["-o"])] = None
    """Export results to file (JSON or CSV)."""


@dataclass
class CompareConfig:
    """Compare two solvers head-to-head.

    Examples:
        cvxbench compare scs clarabel       Compare SCS vs Clarabel
        cvxbench compare scs minix -t 3     Compare with more problems
    """

    solver1: Annotated[str, tyro.conf.Positional]
    """First solver to compare (scs, clarabel, ecos, minix)."""

    solver2: Annotated[str, tyro.conf.Positional]
    """Second solver to compare (scs, clarabel, ecos, minix)."""

    tier: Annotated[int, tyro.conf.arg(aliases=["-t"])] = 2
    """Tier level for comparison (default: 2 for thorough comparison)."""

    suites: list[str] = field(default_factory=lambda: ["maros"])
    """Benchmark suites to use (maros, smp, sdplib, maxcut)."""

    validate: bool = False
    """Validate solutions by computing residuals."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    """Show detailed solver output."""


class ListTarget(str, Enum):
    """What to list."""

    solvers = "solvers"
    suites = "suites"
    baselines = "baselines"
    problems = "problems"


@dataclass
class ListConfig:
    """List available solvers, suites, baselines, or problems.

    Examples:
        cvxbench list solvers       Show available solvers
        cvxbench list suites        Show benchmark suites
        cvxbench list baselines     Show saved baselines
        cvxbench list problems      Show problems in a suite
    """

    target: Annotated[ListTarget, tyro.conf.Positional]
    """What to list: solvers, suites, baselines, or problems."""

    suites: list[str] = field(default_factory=lambda: ["maros"])
    """Filter problems by suite (only for 'problems' target)."""


class BaselineAction(str, Enum):
    """Baseline management action."""

    save = "save"
    show = "show"
    delete = "delete"


@dataclass
class BaselineConfig:
    """Manage saved baselines.

    Examples:
        cvxbench baseline save v1.0     Save current results as baseline
        cvxbench baseline show v1.0     Show baseline details
        cvxbench baseline delete v1.0   Delete a baseline
    """

    action: Annotated[BaselineAction, tyro.conf.Positional]
    """Action to perform: save, show, or delete."""

    name: Annotated[str, tyro.conf.Positional]
    """Baseline name."""


@dataclass
class DownloadConfig:
    """Download benchmark data.

    Examples:
        cvxbench download maros     Download Maros-Meszaros suite
        cvxbench download smp       Download SMP suite
    """

    suite: Annotated[str, tyro.conf.Positional]
    """Suite to download (maros, smp, sdplib, maxcut)."""


def _show_welcome() -> None:
    """Show welcome screen with quick start guide."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Check available solvers
    available_solvers = []
    for solver in Solver:
        try:
            import cvxpy as cp

            if solver.value == "minix":
                try:
                    from minix.cvxpy_backend import MINIX  # noqa: F401

                    available_solvers.append(solver.value)
                except ImportError:
                    pass
            elif hasattr(cp, solver.value.upper()):
                available_solvers.append(solver.value)
        except Exception:
            pass

    welcome_text = f"""[bold cyan]CVXBench[/bold cyan] - Convex Optimization Benchmark Tool

[dim]Solvers available:[/dim]  {', '.join(available_solvers) or 'none detected'}

[bold]Quick start:[/bold]
  cvxbench bench                Run default benchmark (~10s)
  cvxbench bench -t 3           Run medium benchmark (~2min)
  cvxbench bench --full         Run full benchmark suite

[bold]Common tasks:[/bold]
  cvxbench list solvers         Show available solvers
  cvxbench list suites          Show benchmark suites
  cvxbench compare scs clarabel Compare solvers head-to-head
  cvxbench baseline save v1.0   Save results as baseline

Run [cyan]cvxbench --help[/cyan] for full documentation."""

    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))


def main() -> None:
    """Main entry point for the CLI."""
    # If no arguments, show welcome screen
    if len(sys.argv) == 1:
        _show_welcome()
        return

    config = tyro.cli(  # type: ignore[call-overload]
        Annotated[BenchConfig, tyro.conf.subcommand("bench", default=True)]
        | Annotated[CompareConfig, tyro.conf.subcommand("compare")]
        | Annotated[ListConfig, tyro.conf.subcommand("list")]
        | Annotated[BaselineConfig, tyro.conf.subcommand("baseline")]
        | Annotated[DownloadConfig, tyro.conf.subcommand("download")],
        prog="cvxbench",
        description="CVXBench: Convex Optimization Benchmark Tool",
    )

    if isinstance(config, BenchConfig):
        _run_bench(config)
    elif isinstance(config, CompareConfig):
        _run_compare(config)
    elif isinstance(config, ListConfig):
        _run_list(config)
    elif isinstance(config, BaselineConfig):
        _run_baseline(config)
    elif isinstance(config, DownloadConfig):
        _download_suite(config)


def _run_bench(config: BenchConfig) -> None:
    """Run benchmarks with given configuration."""
    from rich.console import Console

    console = Console()

    solver_names = config.solvers
    suite_names = config.suites

    if config.full:
        # Full benchmark mode (like old 'run' command)
        if not config.quiet:
            console.print("[bold]CVXBench[/bold] - Full Benchmark")
            console.print(f"Solvers: {', '.join(solver_names)}")
            console.print(f"Suites: {', '.join(suite_names)}")
            console.print(f"Sample rate: {config.sample * 100:.1f}%")
            console.print(f"Runs per problem: {config.runs}")
            console.print()

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

        display_summary(results, console)
    else:
        # Tier-based quick benchmark mode
        from cvxbench.quick import run_tier

        run_tier(
            tier=config.tier,
            solvers=solver_names,
            suite=suite_names[0],  # Quick mode uses first suite
            validate=config.validate,
            baseline=config.baseline,
        )

        # For tier mode, we need to return early as run_tier handles display
        # But we still want baseline/save functionality
        if config.save:
            console.print(
                "\n[yellow]Note: --save not yet supported in tier mode. "
                "Use --full to save baselines.[/yellow]"
            )
        return

    # Compare to baseline if requested
    if config.baseline:
        from cvxbench.baseline import compare_to_baseline, display_comparison, load_baseline

        try:
            baseline = load_baseline(config.baseline)
            report = compare_to_baseline(results, baseline)
            display_comparison(report, console)
        except FileNotFoundError:
            _show_baseline_error(config.baseline, console)

    # Save as baseline if requested
    if config.save:
        from cvxbench.baseline import save_baseline

        path = save_baseline(results, config.save)
        console.print(f"\n[green]Baseline saved to {path}[/green]")

    # Save to file if requested
    if config.output:
        _save_results(results, config.output)
        console.print(f"\nResults saved to {config.output}")

    # Show suggested next steps
    if not config.quiet:
        _show_suggestions(config, console)


def _run_compare(config: CompareConfig) -> None:
    """Run head-to-head solver comparison."""
    from cvxbench.quick import run_tier

    solver_names = [config.solver1, config.solver2]

    run_tier(
        tier=config.tier,
        solvers=solver_names,
        suite=config.suites[0],
        validate=config.validate,
        baseline=None,
    )


def _run_list(config: ListConfig) -> None:
    """List requested information."""
    if config.target == ListTarget.solvers:
        _list_solvers()
    elif config.target == ListTarget.suites:
        _list_suites()
    elif config.target == ListTarget.baselines:
        _list_baselines()
    elif config.target == ListTarget.problems:
        _list_problems(config.suites)


def _run_baseline(config: BaselineConfig) -> None:
    """Manage baselines."""
    from rich.console import Console

    console = Console()

    if config.action == BaselineAction.save:
        console.print(
            "[yellow]To save a baseline, run a benchmark with --save NAME:[/yellow]"
        )
        console.print("  cvxbench bench --full --save " + config.name)
    elif config.action == BaselineAction.show:
        _show_baseline(config.name, console)
    elif config.action == BaselineAction.delete:
        _delete_baseline(config.name, console)


def _show_baseline(name: str, console: Console) -> None:
    """Show baseline details."""
    from cvxbench.baseline import load_baseline

    try:
        baseline = load_baseline(name)
        console.print(f"[bold]Baseline: {name}[/bold]")
        console.print(f"Created: {baseline.get('created', 'unknown')}")
        console.print(f"Problems: {len(baseline.get('results', []))}")

        # Show solver breakdown
        solvers: dict[str, int] = {}
        for r in baseline.get("results", []):
            solver = r.get("solver", "unknown")
            solvers[solver] = solvers.get(solver, 0) + 1

        if solvers:
            console.print(f"Solvers: {', '.join(f'{k}({v})' for k, v in solvers.items())}")
    except FileNotFoundError:
        _show_baseline_error(name, console)


def _delete_baseline(name: str, console: Console) -> None:
    """Delete a baseline."""
    from pathlib import Path

    baseline_dir = Path.home() / ".cache" / "cvxbench" / "baselines"
    path = baseline_dir / f"{name}.json"

    if path.exists():
        path.unlink()
        console.print(f"[green]Deleted baseline: {name}[/green]")
    else:
        _show_baseline_error(name, console)


def _show_baseline_error(name: str, console: Console) -> None:
    """Show helpful error when baseline not found."""
    from cvxbench.baseline import list_baselines

    console.print(f"[red]Error: Baseline '{name}' not found.[/red]")
    console.print()

    baselines = list_baselines()
    if baselines:
        console.print("[dim]Available baselines:[/dim]")
        for bl_name, created, count in baselines:
            console.print(f"  - {bl_name} ({created[:10]}, {count} problems)")
    else:
        console.print("[dim]No baselines saved yet.[/dim]")

    console.print()
    console.print(f"To save a new baseline: cvxbench bench --full --save {name}")


def _show_suggestions(config: BenchConfig, console: Console) -> None:
    """Show suggested next steps after benchmark run."""
    suggestions = []

    if config.tier < 3 and not config.full:
        suggestions.append(("cvxbench bench -t 3", "More comprehensive benchmark"))

    if not config.save and config.full:
        suggestions.append(("cvxbench bench --full --save v1.0", "Save these results as baseline"))

    if len(config.solvers) == 1:
        other = "clarabel" if config.solvers[0] != "clarabel" else "scs"
        cmd = f"cvxbench compare {config.solvers[0]} {other}"
        suggestions.append((cmd, "Compare against another solver"))

    if suggestions:
        console.print()
        console.print("[dim]Next steps:[/dim]")
        for cmd, desc in suggestions[:2]:  # Show max 2 suggestions
            console.print(f"  [cyan]{cmd}[/cyan]  {desc}")


def _list_suites() -> None:
    """List available benchmark suites."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Benchmark Suites")
    table.add_column("Name", style="cyan")
    table.add_column("Problems", justify="right")
    table.add_column("Type", justify="center")
    table.add_column("Description")

    suites = [
        ("maros", "138", "QP", "Maros-Meszaros classic QP test set"),
        ("smp", "1515", "QP", "SMP/NASOQ graphics/simulation QPs"),
        ("sdplib", "92", "SDP", "SDPLIB semidefinite programs"),
        ("maxcut", "14", "SDP", "Max-Cut SDP relaxations (generated)"),
    ]

    for name, count, ptype, desc in suites:
        table.add_row(name, count, ptype, desc)

    console.print(table)


def _list_solvers() -> None:
    """List available solvers."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Solvers")
    table.add_column("Name", style="cyan")
    table.add_column("Type", justify="center")
    table.add_column("LP", justify="center")
    table.add_column("QP", justify="center")
    table.add_column("SOCP", justify="center")
    table.add_column("SDP", justify="center")

    Y = "[green]Y[/green]"  # noqa: N806
    N = "[dim]-[/dim]"  # noqa: N806
    solvers = [
        ("scs", "First-order", Y, Y, Y, Y),
        ("clarabel", "Interior-point", Y, Y, Y, Y),
        ("ecos", "Interior-point", Y, N, Y, N),
        ("minix", "First-order", Y, Y, Y, N),
    ]

    for row in solvers:
        table.add_row(*row)

    console.print(table)


def _list_problems(suites: list[str]) -> None:
    """List problems in given suites."""
    from rich.console import Console
    from rich.table import Table

    from cvxbench.runner import get_loader

    console = Console()

    for suite in suites:
        loader = get_loader(suite)
        problems = loader.list_problems()

        table = Table(title=f"Problems in {suite} ({len(problems)} total)")
        table.add_column("Name", style="cyan")

        # Show first 20 problems
        for name in problems[:20]:
            table.add_row(name)

        if len(problems) > 20:
            table.add_row(f"... and {len(problems) - 20} more")

        console.print(table)


def _list_baselines() -> None:
    """List saved baselines."""
    from rich.console import Console
    from rich.table import Table

    from cvxbench.baseline import list_baselines

    console = Console()
    baselines = list_baselines()

    if not baselines:
        console.print("[yellow]No baselines found.[/yellow]")
        console.print("Save a baseline with: cvxbench bench --full --save NAME")
        return

    table = Table(title="Saved Baselines")
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Problems", justify="right")

    for name, created, count in baselines:
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(created)
            created_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            created_str = created[:16]

        table.add_row(name, created_str, str(count))

    console.print(table)
    console.print("\nCompare with: cvxbench bench --baseline NAME")


def _download_suite(config: DownloadConfig) -> None:
    """Download benchmark suite data."""
    from rich.console import Console

    console = Console()
    suite_name = config.suite

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

        console.print(
            f"[green]Downloaded {len(problems)} problems to {maros_loader.cache_dir}[/green]"
        )
    elif suite_name == "smp":
        from cvxbench.loaders.smp import SMPLoader

        smp_loader = SMPLoader()
        console.print("Downloading SMP problems from Google Drive...")
        smp_loader.ensure_downloaded()
        problems = smp_loader.list_problems()
        console.print(
            f"[green]Downloaded {len(problems)} problems to {smp_loader.cache_dir}[/green]"
        )
    elif suite_name == "sdplib":
        from cvxbench.loaders.sdplib import SDPLIBLoader

        sdplib_loader = SDPLIBLoader()
        problems = sdplib_loader.list_problems()

        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task("Downloading...", total=len(problems))
            for name in problems:
                try:
                    sdplib_loader.load_problem(name)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to download {name}: {e}[/yellow]")
                progress.update(task, advance=1)

        console.print(
            f"[green]Downloaded {len(problems)} problems to {sdplib_loader.cache_dir}[/green]"
        )
    elif suite_name == "maxcut":
        console.print(
            "[green]Max-Cut problems are generated on-the-fly, no download needed.[/green]"
        )
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


if __name__ == "__main__":
    main()
