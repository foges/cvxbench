# CVXBench

Convex optimization benchmark tool for comparing solvers against standard test suites.

Related project: `../minix` - Rust conic optimization solver with Python bindings.

## Quick Start

```bash
# Install dependencies (with minix)
uv sync --extra minix

# Run benchmarks (1% sample of Maros-Mészáros with SCS)
uv run cvxbench run --sample 0.01

# Compare solvers (including minix)
uv run cvxbench run --solvers minix scs clarabel --sample 0.1

# List available suites/solvers
uv run cvxbench list-suites
uv run cvxbench list-solvers
```

## Architecture

```
src/cvxbench/
├── cli.py           # tyro-based CLI (run, list-suites, list-solvers, download)
├── runner.py        # Benchmark runner, builds CVXPY problems from BenchmarkProblem
├── results.py       # Result aggregation (shifted geometric mean), rich display
└── loaders/
    ├── base.py      # BenchmarkProblem (conic form), BenchmarkLoader ABC
    └── maros_meszaros.py  # QPS/SIF parser for Maros-Mészáros (138 QPs)
```

## Problem Representation

All problems use canonical conic form in `BenchmarkProblem`:

```
minimize    (1/2) x'Px + q'x
subject to  Ax + s = b
            s ∈ K
```

Where K is a Cartesian product of cones: `zero` (equality), `nonneg` (inequality ≤), `soc`, `psd`, `exp`.

Matrices P, A are scipy sparse CSC. Cones stored as `list[tuple[str, int]]`.

## Adding a New Loader

1. Create `src/cvxbench/loaders/your_suite.py`
2. Subclass `BenchmarkLoader`, implement `name()`, `list_problems()`, `load_problem()`
3. Return `BenchmarkProblem` with problem converted to conic form
4. Register in `runner.py:get_loader()` and `cli.py:Suite` enum

## Planned Suites

See `convex_benchmarks_continuous.md` and `convex_benchmarks_mip_integer.md` for full list:
- SMP/NASOQ (1515 QPs) - YAML with MatrixMarket
- QPLIB continuous (~134 QPs)
- SDPLIB (92 SDPs)
- CBLIB continuous (~41 conic)
- MIPLIB (MIP)

## Available Solvers

| Solver   | LP | QP | SOCP | SDP | Notes |
|----------|----|----|------|-----|-------|
| scs      | ✓  | ✓  | ✓    | ✓   | Default, good for large problems |
| clarabel | ✓  | ✓  | ✓    | ✓   | Fast, well-maintained |
| ecos     | ✓  | ✗  | ✓    | ✗   | Small SOCP problems |
| minix    | ✓  | ✓  | ✓    | —   | Local solver (../minix), requires `--extra minix` |

## Minix Integration

Minix is installed as an editable local dependency from `../minix/solver-py`. The CVXPY backend is at `minix.cvxpy_backend.MINIX`.

```python
# Direct usage
import minix
result = minix.solve(A, b, q, cones, P=P)

# Via CVXPY
from minix.cvxpy_backend import MINIX
problem.solve(solver=MINIX())
```

## Key Dependencies

- cvxpy: Problem modeling and solver interface
- scipy: Sparse matrices, MatrixMarket I/O
- tyro: CLI argument parsing
- rich: Terminal output formatting
- minix (optional): Local conic solver from ../minix
