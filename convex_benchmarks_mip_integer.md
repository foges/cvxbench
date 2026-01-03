# Mixed-integer benchmarks (MILP/MIQP/MISOCP) for CVXPY

This document focuses on **integer / mixed-integer** problems and benchmark sources.
For **continuous convex** benchmarks (no integer variables), see `convex_benchmarks_continuous.md`.

> Note: CVXPY can model mixed-integer problems, but solution quality/scale depends heavily on the MILP/MIQP/MISOCP backend you use (CBC, GLPK_MI, SCIP, GUROBI, CPLEX, MOSEK, …).

---

## 0) CVXPY mixed-integer problem families (what you can benchmark)

### MILP
\[
\min_x \; c^\top x \quad \text{s.t.}\; Ax \le b,\; x_i \in \mathbb{Z}\ \text{for some }i
\]
CVXPY: `x = cp.Variable(n, integer=True)` (or mixed integer via boolean masks) + linear constraints.

### MIQP
\[
\min_x \; \tfrac12 x^\top Hx + q^\top x \quad \text{s.t. linear constraints, some } x_i \in \mathbb{Z}
\]
CVXPY supports this if the quadratic is convex (`H ⪰ 0`) and your solver supports MIQP.

### MISOCP / mixed-integer conic
Norm constraints + integrality (requires solvers that support MISOCP).

---

## 1) Existing benchmark suites / instance libraries (already optimization problems)

### 1.1 MIPLIB (Mixed Integer Programming Library)
- **Landing:** https://miplib.zib.de/
- **Why it’s useful:** widely used reference set; lots of structure and diversity; multiple curated subsets (e.g., “benchmark set”) plus full collections.
- **Typical format:** `.mps(.gz)` and related.

**How to formulate:**  
Parse MPS → build MILP (or MIQP if quadratic sections exist) in CVXPY:
- variables with integer/binary tags
- linear constraints
- linear (or convex quadratic) objective

---

### 1.2 QPLIB (discrete subset)
- **Landing:** https://qplib.zib.de/
- **Why it’s useful:** includes **319 discrete** quadratic(-constraint) instances (often MIQP / MIQCQP).
- **How to use with CVXPY:** filter to instances that are **convex** in the continuous relaxation if you want “mixed-integer convex”, otherwise you’ll have nonconvexities CVXPY won’t accept under DCP rules.

---

### 1.3 CBLIB (mixed-integer conic)
- **Landing:** https://cblib.zib.de/
- **Instance directory:** http://cblib.zib.de/download/all/
- **Why it’s useful:** realistic **MISOCP**-style models, especially if you want to benchmark conic + branching.
- **How to formulate:** parse CBF → map cone constraints to CVXPY primitives + declare integer variables.

---

### 1.4 MINLPLib (Mixed-Integer Nonlinear Programming Library)
- **Landing:** https://www.minlplib.org/
- **Why it’s useful:** large catalog of MINLP instances, including many convex MINLPs (and many nonconvex ones).
- **How to use with CVXPY:** only the **convex** MINLP subset is compatible with CVXPY’s DCP rules (and you still need a capable solver).

---

### 1.5 OR-Library (classic combinatorial datasets)
- **Landing:** https://people.brunel.ac.uk/~mastjjb/jeb/orlib/orlib.html
- **Includes:** assignment, set covering, knapsack, facility location variants, etc.
- **How to formulate:** most are MILPs with standard linear constraints.

---

### 1.6 TSPLIB / VRP / scheduling libraries (MILP formulations)
- **TSPLIB:** http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- **VRP instances (one common portal):** http://vrp.atd-lab.inf.puc-rio.br/
- **PSPLIB (project scheduling):** https://www.om-db.wi.tum.de/psplib/

These are not always distributed as “MPS ready”, but data is open and standard MILP formulations exist (MTZ / flow-based / time-indexed / set-partitioning, etc.). Great for variety.

---

## 2) Real-world data you can turn into MIPs (open data → optimization)

These are *not* standard MIP benchmarks but are open datasets that naturally lead to MIPs:

### 2.1 Power systems unit commitment (UC)
- Data sources vary; many UC benchmarks exist in research repos.
- Typical formulation: binary on/off variables, ramping constraints, minimum up/down, etc.
- If you already use MATPOWER network cases, you can combine with demand time series to make UC instances.

### 2.2 Facility location / covering from OpenStreetMap (OSM)
- Use OSM points of interest + demand points to create:
  - p-median / p-center / set cover / maximal coverage MIPs
- Data is open; you control scaling (cities → countries).

### 2.3 Logistics / routing with open VRP datasets
- VRP datasets (above) → vehicle routing MILPs (often large).
- Also supports time windows, capacities, etc.

---

## 3) Practical ingestion (MIP formats)

### MPS / LP
- Most MIP benchmark suites ship MPS.
- Use an MPS reader (Python) to extract:
  - variable bounds, integrality tags
  - linear constraints
  - objective

### CBF (conic)
- For CBLIB: parse CBF → reconstruct conic constraints.

---

## 4) How to get to 1k+ MIP instances quickly

1. Start with **MIPLIB** full collections (already ~1k-scale depending on edition).
2. Add **OR-Library** families (hundreds more).
3. Add **QPLIB discrete** (hundreds of MIQP/MIQCQP).
4. Add **CBLIB mixed-integer** for MISOCP diversity.

---

## 5) Suggested approach if you benchmark solvers

- Keep **continuous convex** and **mixed-integer** suites separate in reporting:
  - solver support differs
  - performance scaling differs
- Track per-instance metadata:
  - number of vars/constraints
  - number of integer/binary vars
  - convexity flags (where applicable)
  - cone types used (SOC/PSD/EXP/…)
