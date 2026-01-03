# Continuous convex optimization benchmarks (CVXPY-friendly, **no integer variables**)

This document focuses on **continuous** convex problems (LP/QP/SOCP/SDP/EXP/GP, etc.) that can be modeled and solved with **CVXPY**.  
A separate document covers **mixed-integer / MIP** sources.

---

## 0) What “clear formulation” means in practice

A dataset/problem source below is “usable” if you can consistently map it into one of CVXPY’s standard convex templates:

### Quadratic Program (QP)
\[
\min_x \; \tfrac12 x^\top H x + q^\top x \quad \text{s.t.}\; A x = b,\;\; l \le Cx \le u
\]
CVXPY skeleton:
```python
import cvxpy as cp

x = cp.Variable(n)
obj = 0.5 * cp.quad_form(x, H) + q @ x
constraints = []
if A is not None: constraints += [A @ x == b]
if C is not None: constraints += [l <= C @ x, C @ x <= u]
prob = cp.Problem(cp.Minimize(obj), constraints)
```

### Second-Order Cone Program (SOCP)
Common pattern: norms in constraints/objective, e.g.
\[
\min_x\; c^\top x \;\;\text{s.t.}\;\; \|F_i x + g_i\|_2 \le a_i^\top x + b_i
\]
CVXPY uses `cp.SOC(t, Fx_plus_g)` or `cp.norm2(...) <= ...`.

### Semidefinite Program (SDP)
\[
\min_X \; \langle C, X\rangle \;\;\text{s.t.}\;\; \langle A_i, X\rangle = b_i,\;\; X \succeq 0
\]
CVXPY skeleton:
```python
X = cp.Variable((n, n), PSD=True)
obj = cp.Minimize(cp.sum(cp.multiply(C, X)))  # <C, X>
constraints = [cp.sum(cp.multiply(Ai, X)) == bi for Ai, bi in eqs]
prob = cp.Problem(obj, constraints)
```

### Exponential cone / entropy / log-sum-exp models
Logistic regression, maximum entropy, geometric programming (via `cp.geo_mean`, `cp.log_sum_exp`, etc.). These provide *non-QP* convex benchmarks.

---

## 1) Existing **continuous convex** benchmark suites (already optimization problems)

The list is ordered so the first items get you to **≥ 1k high-quality continuous instances** quickly.

### 1.1 SMP Repository (NASOQ) — **1515 real, strictly-convex QPs** (your “1k+” anchor)
- **What:** A large, curated repository of sparse **strictly convex quadratic programs** from graphics/simulation/control and other applications.
- **Scale/variety:** includes tiny to huge sparse QPs; stresses sparse linear algebra and KKT systems.
- **Count:** the public repository page enumerates **1515** problems across many application groups.
- **Format:** **SMP YAML** where each matrix/vector is embedded as a **MatrixMarket** block.
- **Landing page:** https://nasoq.github.io/smp.html
- **Format spec + example:** https://nasoq.github.io/docs/repository/
- **Download (bulk):**
  - Repo + scripts: https://github.com/sympiler/nasoq-benchmarks
  - SMP format tools: https://github.com/sympiler/smp-format
  - Many mirrors point to a bulk tarball such as: `SMP_Repository_SIGGRAPH20.tgz` (see `nasoq-benchmarks` README)

**How to formulate (generic for *all* SMP instances):**  
SMP stores QPs in the compressed “bounded inequality” form:
\[
\min_x \; \tfrac12 x^\top Hx + q^\top x
\quad\text{s.t.}\quad
A x = b,\;\;
l \le Cx \le u.
\]
Map each YAML block to `H, q, A, b, C, l, u`, then use the QP CVXPY skeleton in §0.

**Parsing notes:**
- Each YAML field is a string that begins with `%%MatrixMarket ...`.
- You can parse those with `scipy.io.mmread(io.StringIO(mm_text))`.
- Bounds may include `inf` / `-inf`; map them to `None` or large sentinels.

---

### 1.2 QPLIB — **134 continuous QP/QCQP instances** (filter to convex + continuous)
- **What:** A curated library of quadratic programming instances with rich metadata (convex/nonconvex, discrete/continuous, linear/quadratic constraints, etc.).
- **Counts:** final version includes **134 continuous** and **319 discrete** instances.
- **Landing + download:** https://qplib.zib.de/  
  (follow “download QPLIB archive” and “documentation” on the site)

**How to get the *convex continuous* subset:**
- Use QPLIB metadata to filter for:
  - **continuous variables only**
  - **convex quadratic objective** (or linear)
  - **constraints that are linear or convex quadratic** (to remain DCP-friendly)
- Prefer instances already provided in formats you can parse (LP/MPS/QPS/AMPL/GAMS depending on your tooling).

**How to formulate:**
- Many QPLIB continuous convex instances are QPs:
  \[
  \min \tfrac12 x^\top Hx + q^\top x \;\text{s.t.}\; Gx \le h,\; Ax=b,\; l\le x\le u
  \]
  which maps directly to CVXPY.

---

### 1.3 Mittelmann “continuous convex QPLIB benchmark” (+ extra convex QPs) — includes **very large** QPs
- **What:** Hans Mittelmann maintains benchmark pages and downloadable problem sets (including a well-known **32-problem** continuous convex QPLIB subset used in solver comparisons, plus additional convex QPs).
- **Entry point:** https://plato.asu.edu/bench.html
- **Extra convex QPs (directory):** https://plato.asu.edu/ftp/cconv/  
  (contains additional convex QPs in MPS-like formats; some are extremely large)

**How to formulate:** QP as in §0.  
**Why it’s useful:** adds a “hard, large, real-world-ish” tail beyond SMP.

---

### 1.4 Maros–Mészáros convex QP test set — **138** classic QPs
- **What:** A standard, widely used set of **138 convex QPs** designed to be challenging (large, sparse, ill-conditioned, etc.).
- **Easy download (GitHub repack for benchmarking):** https://github.com/qpsolvers/maros_meszaros_qpbenchmark  
- **Mirror of the original SVN/QPS files:** https://github.com/optimizers/maros-meszaros-mirror  
- **CUTEr page (background):** https://www.cuter.rl.ac.uk/Problems/marmes.shtml

**How to formulate:** QP as in §0.  
**Notes:** some instances have only *semidefinite* Hessians; pick solvers accordingly (OSQP handles PSD; some solvers need strict convexity).

---

### 1.5 CBLIB — continuous conic problems (SOCP-heavy) + conversion tooling
- **What:** Conic Benchmark Library (CBLIB): realistic conic instances, mostly SOCP; includes both continuous and mixed-integer conic problems.
- **Landing page:** https://cblib.zib.de/
- **Raw instance directory listing:** http://cblib.zib.de/download/all/
- **Scripts/tools:** https://github.com/HFriberg/cblib-base/

**How to use for *continuous only*:**
- CBLIB historically has 121 instances with **80 mixed-integer**, so you can filter to ~**41 continuous** for your non-integer suite.
- Many instances are in **CBF** format (conic benchmark format).

**How to formulate in CVXPY (two practical paths):**
1. **Parse CBF → high-level CVXPY**:
   - Use a CBF parser (CBLIB lists prototype parsers for Python/Julia/etc.).
   - Translate each cone block into `cp.SOC`, PSD constraints, nonnegativity, etc.
2. **Parse CBF → standard conic matrices → build CVXPY “by cone type”**:
   - Convert to a cone program representation and then map to CVXPY primitives.

CBLIB is one of the best sources to ensure you aren’t “only QPs”.

---

### 1.6 SDPLIB — **92 SDP** test problems
- **What:** A canonical library of semidefinite programming test instances from many applications.
- **GitHub mirror:** https://github.com/coin-or/SDPLIB

**How to formulate:** parse SDPA data files into \(\langle C, X\rangle\) + linear equalities + PSD constraints as in §0.

---

### 1.7 DIMACS SDP/SQLP instances (SeDuMi `.mat` format)
- **What:** Semidefinite-quadratic-linear benchmark instances (often used in SDP solver testing).
- **GitHub repo:** https://github.com/vsdp/DIMACS

**How to formulate:**
- Load MATLAB `.mat` data (`scipy.io.loadmat`).
- Translate SeDuMi-style cone data into CVXPY primitives (PSD blocks, SOC blocks, nonnegativity).

---

### 1.8 PENNON / Kocvara sparse SDP problems (structural optimization) — zipped groups
- **What:** A small-but-valuable set of **sparse** SDPs arising in structural optimization.
- **Download page (SDPA format zips):** https://web.mat.bham.ac.uk/kocvara/pennon/problems.html

**How to formulate:** SDPs as in §0 (after parsing SDPA).

---

## 2) Real-world datasets you can reformulate into **continuous convex** benchmarks

These are not necessarily “standard optimization benchmarks” but are open datasets that can be turned into convex programs in a uniform way. They’re useful for variety beyond QP/SOCP/SDP libraries.

### 2.1 SuiteSparse Matrix Collection → least squares, LASSO, Huber, basis pursuit, …
- **What:** Thousands of real sparse matrices from many application domains.
- **Landing:** https://sparse.tamu.edu/  
- **License overview:** https://sparse.tamu.edu/about  
- **Python tooling (search + download):** `ssgetpy` https://github.com/drdarshan/ssgetpy

**Canonical convex problems you can generate from each matrix \(A\):**
1. **Least squares / ridge (QP):**
   \[
   \min_x \; \tfrac12\|Ax-b\|_2^2 + \tfrac\lambda2\|x\|_2^2
   \]
2. **LASSO / basis pursuit denoising (convex, often SOCP/QP after reduction):**
   \[
   \min_x \; \tfrac12\|Ax-b\|_2^2 + \lambda\|x\|_1
   \]
3. **Huber regression (SOCP representable):**
   \[
   \min_x \;\sum_i \mathrm{Huber}_\delta((Ax-b)_i)
   \]
4. **Chebyshev / \(L_\infty\) regression (LP):**
   \[
   \min_x \; \|Ax-b\|_\infty
   \]

**How to pick \(b\):** common benchmark choices are random \(b\), or \(b = A x^\star + \epsilon\) with a planted sparse \(x^\star\).  
**Why it’s “high quality”:** the *data matrices* are real and diverse; you can generate **thousands** of instances with controlled conditioning/sparsity.

---

### 2.2 OpenML / LIBSVM datasets → logistic regression, SVM, elastic net, …
- **OpenML:** https://www.openml.org/ (very large catalog; mostly tabular ML datasets)
- **LIBSVM datasets:** https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

**Convex formulations:**
- **SVM (QP):**
  \[
  \min_{w,b,\xi}\; \tfrac12\|w\|_2^2 + C\sum_i \xi_i
  \;\text{s.t.}\;
  y_i(w^\top x_i + b) \ge 1-\xi_i,\; \xi_i\ge 0
  \]
- **Logistic regression (EXP cone):**
  \[
  \min_{w,b}\; \sum_i \log(1+\exp(-y_i(w^\top x_i+b))) + \tfrac\lambda2\|w\|_2^2
  \]
- **Group lasso / elastic net** for structured sparsity (convex, sometimes SOCP).

**Tip to avoid “10k slight variants”:**
- Assign *one canonical model* per dataset type:
  - binary classification → logistic or SVM
  - regression → ridge/LASSO/Huber
  - count data → Poisson regression (exp cone)

---

### 2.3 MovieLens (recommendation) → nuclear-norm matrix completion (SDP-ish / convex)
- **Dataset:** https://grouplens.org/datasets/movielens/
- **Convex formulation (matrix completion):**
  \[
  \min_X \; \|X\|_* \quad \text{s.t.}\;\; X_{ij}=M_{ij}\;\forall (i,j)\in\Omega
  \]
CVXPY supports `cp.normNuc(X)` directly.

**Why it’s useful:** introduces **matrix variables** and low-rank structure; can be very large.

---

### 2.4 SNAP graph datasets → max-cut SDP relaxations, Laplacian smoothing, graph lasso, …
- **Dataset portal:** https://snap.stanford.edu/data/

**Convex problems you can derive:**
- **Max-cut SDP relaxation** on a graph \(G\) (SDP):
  \[
  \max_X \; \tfrac14 \sum_{(i,j)\in E} w_{ij}(1 - X_{ij})
  \quad \text{s.t.}\;\; \mathrm{diag}(X)=\mathbf{1},\; X\succeq 0
  \]
- **Graph Laplacian smoothing / semi-supervised learning (QP):**
  \[
  \min_x \; \|x-y\|_2^2 + \lambda x^\top L x
  \]
- **Graphical lasso** if you build covariances from graph signals (convex but heavier).

---

### 2.5 MATPOWER / power-system cases → convex relaxations & dispatch
- **MATPOWER (cases repository):** https://github.com/MATPOWER/matpower

**Convex formulations:**
- **DC optimal power flow / economic dispatch** often reduces to LP/QP.
- **SOCP relaxations** of AC OPF on certain network classes.

**Why it’s useful:** power systems yield structured large sparse constraints and domain-realistic scaling.

---

### 2.6 Transportation Networks (traffic assignment) → Beckmann formulation (convex)
- **TNTP / transportation network test data:**  
  One common portal/mirror: https://github.com/bstabler/TransportationNetworks

**Convex formulation (Beckmann):** traffic assignment with separable convex link costs:
\[
\min_f \; \sum_{e\in E}\int_0^{f_e} c_e(t)\,dt \;\;\text{s.t. flow conservation / demand constraints}
\]
Often large and sparse; convex if \(c_e(\cdot)\) is nondecreasing.

---

## 3) Practical ingestion notes (continuous convex)

### SMP (NASOQ) YAML
- YAML keys map to matrices/vectors embedded in MatrixMarket text.
- Strategy: `yaml.safe_load` → for each field → `scipy.io.mmread(StringIO(text))`.

### QP in QPS/MPS-like formats
- Maros–Mészáros and many classical QPs use **QPS** (a subset of SIF).
- Practical approach:
  - use an existing parser/converter in Python (many libraries exist), or
  - convert QPS → sparse `H,q,A,b,G,h,l,u` then build CVXPY QP.

### SDP formats
- **SDPA**: common for SDPLIB / PENNON problems.
- **SeDuMi `.mat`**: common for DIMACS; load via `scipy.io.loadmat`.

### SuiteSparse matrices
- Download as `.mtx` and load with `scipy.io.mmread`, or `.mat` and load via `scipy.io.loadmat`.

---

## 4) How to get to **1k+ high-quality** continuous convex instances (and beyond)

A practical “high quality first” recipe:

1. **SMP repository** → **1515** real strictly-convex QPs (already exceeds your 1k target).
2. Add **non-QP variety**:
   - **SDPLIB** (92 SDPs)
   - **DIMACS** SDPs
   - **CBLIB continuous** (~41 conic)
   - **PENNON** sparse SDPs
3. For **10k+** instances without devolving into trivial random generation:
   - Use **SuiteSparse matrices** and create *several distinct convex problem families* (LS, LASSO, Huber, \(L_\infty\)), but cap per-matrix variants (e.g., 1–3) and diversify by matrix group/domain.
   - Use **OpenML/LIBSVM** datasets, but assign a single canonical model per dataset class to avoid near-duplicates.

---

## 5) “High value” continuous convex sources (quick checklist)

If you only implement a few pipelines first, implement these in this order:

1. **SMP (NASOQ)** — 1515 QPs in one consistent format (biggest immediate win).
2. **Maros–Mészáros** — classic hard convex QPs.
3. **QPLIB (continuous convex filter)** — curated QP/QCQP variety.
4. **CBLIB (continuous)** — SOCP-heavy, good for conic diversity.
5. **SDPLIB + DIMACS + PENNON** — SDP diversity.
6. **SuiteSparse-derived convex problems** — scalable path to 10k+ with real data.

---
