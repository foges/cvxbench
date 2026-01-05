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
# Run default benchmark (~10s)
cvxbench bench

# Run medium benchmark (~2min)
cvxbench bench -t 3

# Compare specific solvers
cvxbench bench -s scs clarabel

# Head-to-head comparison
cvxbench compare scs clarabel

# Full benchmark run
cvxbench bench --full --sample 0.1

# With solution validation
cvxbench bench -t 2 --validate
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
- **Multiple Suites**: Maros-Mészáros QP, SMP/NASOQ, SDPLIB, Max-Cut SDP

## Benchmark Suites

| Suite | Problems | Type | Description |
|-------|----------|------|-------------|
| maros | 138 | QP | Maros-Mészáros classic QP test set |
| smp | 1515 | QP | SMP/NASOQ graphics/simulation QPs |
| sdplib | 92 | SDP | SDPLIB semidefinite programs |
| maxcut | 14 | SDP | Max-Cut SDP relaxations (generated) |

## Supported Solvers

| Solver | Type | LP | QP | SOCP | SDP |
|--------|------|:--:|:--:|:----:|:---:|
| SCS | First-order | ✓ | ✓ | ✓ | ✓ |
| Clarabel | Interior-point | ✓ | ✓ | ✓ | ✓ |
| ECOS | Interior-point | ✓ | - | ✓ | - |
| MINIX | First-order | ✓ | ✓ | ✓ | - |

## Commands

```bash
cvxbench                    # Show welcome screen with quick start
cvxbench bench              # Run tiered benchmark (default: tier 1)
cvxbench bench -t 3         # Run tier 3 benchmark (~2min)
cvxbench bench --full       # Run full benchmark suite
cvxbench compare scs minix  # Head-to-head solver comparison
cvxbench list solvers       # Show available solvers
cvxbench list suites        # Show benchmark suites
cvxbench list baselines     # Show saved baselines
cvxbench download maros     # Download benchmark data
```

## Baseline Tracking

```bash
# Save a baseline after full benchmark
cvxbench bench --full --save v1.0

# Compare against baseline
cvxbench bench -t 3 --baseline v1.0

# Manage baselines
cvxbench baseline show v1.0
cvxbench baseline delete v1.0
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
