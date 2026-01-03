# CVXBench

Convex optimization benchmark suite for comparing solvers against standard test problems.

## Installation

```bash
uv sync
```

## Quick Start

```bash
# Run benchmarks with SCS on 10% sample of Maros-Mészáros
uv run cvxbench run --sample 0.1

# Compare multiple solvers
uv run cvxbench run --solvers scs clarabel --sample 0.2

# List available suites and solvers
uv run cvxbench list-suites
uv run cvxbench list-solvers
```

## Supported Benchmark Suites

| Suite | Problems | Type | Status |
|-------|----------|------|--------|
| Maros-Mészáros | 138 | QP | ✅ |
| SMP/NASOQ | 1515 | QP | Planned |
| QPLIB | ~134 | QP/QCQP | Planned |
| CBLIB | ~41 | Conic | Planned |
| SDPLIB | 92 | SDP | Planned |

## Supported Solvers

- **SCS** - Splitting Conic Solver (first-order, ADMM)
- **Clarabel** - Interior point (Rust)
- **ECOS** - Interior point (C)
- **OSQP** - First-order QP solver

## Output

CVXBench reports:
- Per-problem solve times and status
- Shifted geometric mean (standard benchmark metric)
- Win/loss counts across solvers

## Adding a New Loader

1. Create `src/cvxbench/loaders/your_suite.py`
2. Subclass `BenchmarkLoader` from `base.py`
3. Implement `name()`, `list_problems()`, `load_problem()`
4. Return `BenchmarkProblem` in canonical conic form
5. Register in `cli.py` Suite enum and `runner.py`

## License

MIT
