"""Quick benchmark tiers for development feedback.

Usage:
    uv run python -m cvxbench.quick          # tier 1 (~1 min)
    uv run python -m cvxbench.quick --tier 2 # tier 2 (~2 min)
    uv run python -m cvxbench.quick --tier 5 # tier 5 (~20 min)
"""

from __future__ import annotations

import math
import sys
import argparse
import warnings

from scipy.sparse import SparseEfficiencyWarning

# Suppress noisy warnings during benchmarks
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

# Pre-selected problems by tier (curated for coverage and target runtime)
# Times are targets for running with default solvers (scs + clarabel)
# Note: CVXQP*_L and CVXQP*_M problems often fail - avoid them
TIERS = {
    1: [  # ~10s, 8 problems - quick sanity check
        "HS35", "HS52", "QAFIRO", "LOTSCHD",
        "CVXQP1_S", "DUAL1",
        "QGROW7",       # ~3s SCS
        "CONT-050",     # ~2s SCS
    ],
    2: [  # ~30s, 12 problems
        "HS21", "HS35", "HS52", "HS76", "QAFIRO", "LOTSCHD",
        "CVXQP1_S", "CVXQP2_S", "DUAL1",
        "QGROW7", "QGROW15",   # ~9s SCS total
        "CONT-050",            # ~2s SCS
    ],
    3: [  # ~2 min, 20 problems
        # Small (quick)
        "HS21", "HS35", "HS52", "HS76", "QAFIRO", "LOTSCHD",
        # Medium
        "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
        "DUAL1", "DUAL2", "PRIMAL1",
        # Large
        "QGROW7", "QGROW15", "QGROW22",     # ~18s SCS total
        "CONT-050", "CONT-100",             # ~45s SCS total
        "AUG3DQP", "AUG3DCQP",
        "LISWET1",                          # ~5s SCS
    ],
    4: [  # ~5 min, 30 problems
        # Small
        "HS21", "HS35", "HS52", "HS76", "HS118", "QAFIRO", "LOTSCHD", "GENHS28",
        # Medium
        "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
        "DUAL1", "DUAL2", "DUAL3", "DUAL4",
        "PRIMAL1", "PRIMALC1", "PRIMALC2",
        # Large
        "QGROW7", "QGROW15", "QGROW22",
        "CONT-050", "CONT-100", "CONT-101",  # CONT-101 ~100s SCS
        "AUG3DQP", "AUG3DCQP",
        "LISWET1", "LISWET2",
        "QSEBA",
    ],
    5: [  # ~15 min, 45 problems - comprehensive
        # Small
        "HS21", "HS35", "HS52", "HS76", "HS118", "HS268",
        "QAFIRO", "LOTSCHD", "DPKLO1", "GENHS28",
        # Medium
        "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
        "DUAL1", "DUAL2", "DUAL3", "DUAL4",
        "DUALC1", "DUALC2", "DUALC5", "DUALC8",
        "PRIMAL1", "PRIMAL2", "PRIMALC1", "PRIMALC2", "PRIMALC5",
        # Large
        "QGROW7", "QGROW15", "QGROW22",
        "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201",
        "AUG3D", "AUG3DC", "AUG3DQP", "AUG3DCQP",
        "LISWET1", "LISWET2", "LISWET3", "LISWET4",
        "QSEBA", "QSCSD1",
        "STADAT1", "STADAT2", "STADAT3",
        "STCQP1", "STCQP2",
    ],
}


def get_loader(suite: str):
    """Get the appropriate loader for a suite."""
    if suite == "maros":
        from cvxbench.loaders.maros_meszaros import MarosMeszarosLoader
        return MarosMeszarosLoader()
    elif suite == "smp":
        from cvxbench.loaders.smp import SMPLoader
        return SMPLoader()
    else:
        raise ValueError(f"Unknown suite: {suite}")


def run_tier(
    tier: int,
    solvers: list[str] | None = None,
    suite: str = "maros",
    validate: bool = False,
    baseline: str | None = None,
) -> None:
    """Run benchmark tier.

    Args:
        tier: Tier level (1-5).
        solvers: List of solver names.
        suite: Benchmark suite to use ("maros" or "smp").
        validate: Whether to validate solutions.
        baseline: Baseline name to compare against.
    """
    from cvxbench.runner import run_single

    if solvers is None:
        solvers = ["scs", "clarabel"]

    loader = get_loader(suite)

    # Get problem list for tier
    # TIERS only applies to maros suite; other suites use size-based selection
    if suite != "maros" or tier == 5 or TIERS[tier][0] is None:
        # Load problems by size threshold based on tier
        size_limits = {1: 50, 2: 100, 3: 200, 4: 500, 5: 1000}
        max_size = size_limits.get(tier, 500)
        problems = []
        for name in loader.list_problems():
            try:
                p = loader.load_problem(name)
                if p.n_vars <= max_size:
                    problems.append(name)
            except Exception:
                continue
        problems = sorted(problems)
        # Limit count for lower tiers
        count_limits = {1: 5, 2: 10, 3: 20, 4: 35, 5: None}
        limit = count_limits.get(tier)
        if limit and len(problems) > limit:
            problems = problems[:limit]
    else:
        problems = TIERS[tier]

    print(f"CVXBench Quick Tier {tier} ({len(problems)} problems, {suite})")
    print(f"Solvers: {', '.join(solvers)}")
    if validate:
        print("Validation: enabled")
    print()

    # Header
    header = f"{'Problem':<18} {'n':>5} {'m':>5} |"
    for s in solvers:
        header += f" {s:>8}"
    header += " | Winner"
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    results = {s: [] for s in solvers}
    all_results = []
    wins = {s: 0 for s in solvers}
    failed = {s: 0 for s in solvers}
    inaccurate = {s: 0 for s in solvers}

    for name in problems:
        try:
            problem = loader.load_problem(name)
        except Exception as e:
            print(f"{name:16} ERROR: {e}")
            continue

        times = {}
        statuses = {}
        solve_results = {}
        for solver in solvers:
            r = run_single(problem, solver, 0, timeout=30, verbose=False, validate=validate)
            t = r.time_sec * 1000
            times[solver] = t
            statuses[solver] = r.status
            solve_results[solver] = r
            all_results.append(r)

            if r.status == "optimal":
                results[solver].append(t)
            elif r.status == "optimal_inaccurate":
                results[solver].append(t)
                inaccurate[solver] += 1
            else:
                failed[solver] += 1

        # Find winner among optimal solutions
        optimal = {s: t for s, t in times.items() if statuses[s] in ("optimal", "optimal_inaccurate")}
        winner = min(optimal, key=optimal.get) if optimal else "-"
        if winner != "-":
            wins[winner] += 1

        # Format output line
        line = f"{name:18} {problem.n_vars:5} {problem.n_constraints:5} |"
        for s in solvers:
            r = solve_results[s]
            # Determine marker
            if statuses[s] not in ("optimal", "optimal_inaccurate"):
                marker = "!"  # Failed
            elif validate and r.primal_residual is not None and r.primal_residual > 1e-6:
                marker = "~"  # Poor accuracy
            elif validate and r.constraint_violation is not None and r.constraint_violation > 1e-6:
                marker = "~"  # Poor accuracy
            elif statuses[s] == "optimal_inaccurate":
                marker = "~"  # Solver reported inaccurate
            else:
                marker = " "
            line += f" {times[s]:7.1f}{marker}"
        line += f" | {winner}"
        print(line)
        sys.stdout.flush()

    # Summary
    print("-" * len(header))
    print("! = failed, ~ = inaccurate")
    print()

    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title=f"Summary (Tier {tier}, {len(problems)} problems)")
    table.add_column("Solver", style="cyan")
    table.add_column("Optimal", justify="right")
    table.add_column("Inaccurate", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Geom Mean", justify="right")
    table.add_column("Wins", justify="right")

    for solver in solvers:
        n_opt = len(results[solver])
        if n_opt > 0:
            geom = math.exp(sum(math.log(t) for t in results[solver]) / n_opt)
            geom_str = f"{geom:.2f}ms"
        else:
            geom_str = "-"

        opt_str = f"{n_opt}/{len(problems)}"
        inacc_str = str(inaccurate[solver]) if inaccurate[solver] > 0 else "-"
        fail_str = f"[red]{failed[solver]}[/red]" if failed[solver] > 0 else "-"

        table.add_row(solver, opt_str, inacc_str, fail_str, geom_str, str(wins[solver]))

    console.print(table)

    # Baseline comparison if requested
    if baseline:
        from cvxbench.baseline import compare_to_baseline, display_comparison, load_baseline
        from rich.console import Console

        try:
            bl = load_baseline(baseline)
            report = compare_to_baseline(all_results, bl)
            console = Console()
            display_comparison(report, console)
        except FileNotFoundError as e:
            print(f"\nBaseline not found: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick benchmark tiers")
    parser.add_argument("--tier", "-t", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Tier level: 1(~1min), 2(~2min), 3(~5min), 4(~10min), 5(~20min)")
    parser.add_argument("--solvers", "-s", nargs="+", default=None,
                        help="Solvers to test (default: scs clarabel)")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Validate solutions (compute residuals)")
    parser.add_argument("--baseline", "-b", type=str, default=None,
                        help="Compare against this baseline")
    args = parser.parse_args()

    run_tier(args.tier, args.solvers, args.validate, args.baseline)


if __name__ == "__main__":
    main()
