# CVXBench

A benchmarking tool for convex optimization solvers. Compare solver performance on standard test problems with validation and regression tracking.

## Installation

```bash
pip install cvxbench
```

Or with uv:
```bash
uv add cvxbench
```

## Quick Start

```bash
# Quick benchmark (tier 1 = ~10s, tier 5 = ~15min)
cvxbench quick -t 1

# Compare specific solvers
cvxbench quick -t 2 -s scs clarabel

# Full benchmark run
cvxbench run --sample 0.1 --solvers scs clarabel ecos

# With solution validation
cvxbench quick -t 2 --validate
```

## Example Output

```
CVXBench Quick Tier 2 (12 problems, maros)
Solvers: scs, clarabel, ecos

Problem                n     m |      scs clarabel     ecos | Winner
--------------------------------------------------------------------
HS35                   3     4 |     4.1      2.3      3.4  | clarabel
QGROW7               301   721 |  2739.7     19.5     20.1  | clarabel
CONT-050            2597  7595 |  1764.8     99.6    226.2  | clarabel
...

                 Summary (Tier 2, 12 problems)
┏━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┓
┃ Solver   ┃ Optimal ┃ Inaccurate ┃ Failed ┃ Geom Mean ┃ Wins ┃
┡━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━┩
│ scs      │   12/12 │          1 │      - │   42.3ms  │    3 │
│ clarabel │   12/12 │          - │      - │   12.1ms  │    9 │
│ ecos     │   12/12 │          - │      - │   23.5ms  │    0 │
└──────────┴─────────┴────────────┴────────┴───────────┴──────┘
```

## Features

- **Quick Benchmarks**: Tiered problem sets for fast feedback during development
- **Solution Validation**: Verify primal/dual residuals and constraint violations
- **Baseline Tracking**: Save results and detect regressions
- **Multiple Suites**: Maros-Mészáros QP, SMP/NASOQ, and more

## Benchmark Suites

| Suite | Problems | Type | Description |
|-------|----------|------|-------------|
| Maros-Mészáros | 138 | QP | Classic QP test set |
| SMP/NASOQ | 1515 | QP | Graphics/simulation QPs |

## Supported Solvers

| Solver | Type | LP | QP | SOCP | SDP |
|--------|------|:--:|:--:|:----:|:---:|
| SCS | First-order | ✓ | ✓ | ✓ | ✓ |
| Clarabel | Interior-point | ✓ | ✓ | ✓ | ✓ |
| ECOS | Interior-point | ✓ | - | ✓ | - |

## Commands

```bash
cvxbench quick          # Quick tiered benchmarks
cvxbench run            # Full benchmark run
cvxbench list-suites    # Show available suites
cvxbench list-solvers   # Show available solvers
cvxbench download       # Download benchmark data
cvxbench baselines      # List saved baselines
```

## Baseline Tracking

```bash
# Save a baseline
cvxbench run -s clarabel --save-baseline v1.0

# Compare against baseline
cvxbench run -s clarabel --baseline v1.0
```

## Development

```bash
git clone https://github.com/cvxbench/cvxbench
cd cvxbench
uv sync --dev
uv run pytest
```

## License

MIT
