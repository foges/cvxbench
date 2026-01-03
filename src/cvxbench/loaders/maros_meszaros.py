"""Loader for the Maros-Mészáros QP test set.

The Maros-Mészáros test set contains 138 classic quadratic programming problems.
These are available in QPS format (an extension of MPS for QPs).

Reference: https://github.com/qpsolvers/maros_meszaros_qpbenchmark
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TextIO

import numpy as np
import requests
from scipy import sparse

from cvxbench.loaders.base import BenchmarkLoader, BenchmarkProblem

# All 138 problems in the Maros-Mészáros test set
PROBLEM_LIST = [
    "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP", "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
    "BOYD1", "BOYD2", "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201",
    "CONT-300", "CVXQP1_L", "CVXQP1_M", "CVXQP1_S", "CVXQP2_L", "CVXQP2_M", "CVXQP2_S",
    "CVXQP3_L", "CVXQP3_M", "CVXQP3_S", "DPKLO1", "DTOC3", "DUAL1", "DUAL2", "DUAL3",
    "DUAL4", "DUALC1", "DUALC2", "DUALC5", "DUALC8", "EXDATA", "GENHS28", "GOULDQP2",
    "GOULDQP3", "HS118", "HS21", "HS268", "HS35", "HS35MOD", "HS51", "HS52", "HS53",
    "HS76", "HUES-MOD", "HUESTIS", "KSIP", "LASER", "LISWET1", "LISWET10", "LISWET11",
    "LISWET12", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6", "LISWET7",
    "LISWET8", "LISWET9", "LOTSCHD", "MOSARQP1", "MOSARQP2", "POWELL20", "PRIMAL1",
    "PRIMAL2", "PRIMAL3", "PRIMAL4", "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
    "Q25FV47", "QADLITTL", "QAFIRO", "QBANDM", "QBEACONF", "QBORE3D", "QBRANDY",
    "QCAPRI", "QE226", "QETAMACR", "QFFFFF80", "QFORPLAN", "QGFRDXPN", "QGROW15",
    "QGROW22", "QGROW7", "QISRAEL", "QPCBLEND", "QPCBOEI1", "QPCBOEI2", "QPCSTAIR",
    "QPILOTNO", "QPTEST", "QRECIPE", "QSC205", "QSCAGR25", "QSCAGR7", "QSCFXM1",
    "QSCFXM2", "QSCFXM3", "QSCORPIO", "QSCRS8", "QSCSD1", "QSCSD6", "QSCSD8",
    "QSCTAP1", "QSCTAP2", "QSCTAP3", "QSEBA", "QSHARE1B", "QSHARE2B", "QSHELL",
    "QSHIP04L", "QSHIP04S", "QSHIP08L", "QSHIP08S", "QSHIP12L", "QSHIP12S", "QSIERRA",
    "QSTAIR", "QSTANDAT", "S268", "STADAT1", "STADAT2", "STADAT3", "STCQP1", "STCQP2",
    "TAME", "UBH1", "VALUES", "YAO", "ZECEVIC2",
]

# Base URL for downloading SIF/QPS files (from the mirror repository)
BASE_URL = "https://raw.githubusercontent.com/optimizers/maros-meszaros-mirror/master"


class MarosMeszarosLoader(BenchmarkLoader):
    """Loader for the Maros-Mészáros QP test set (138 problems)."""

    def name(self) -> str:
        return "maros_meszaros"

    def list_problems(self) -> list[str]:
        return PROBLEM_LIST.copy()

    def estimated_size_mb(self) -> float:
        return 50.0  # ~50 MB for all QPS files

    def load_problem(self, name: str) -> BenchmarkProblem:
        """Load a problem by name."""
        if name not in PROBLEM_LIST:
            msg = f"Unknown problem: {name}. Use list_problems() to see available problems."
            raise ValueError(msg)

        # Check cache first
        sif_path = self.cache_dir / f"{name}.SIF"
        if not sif_path.exists():
            self._download_problem(name)

        # Parse the SIF/QPS file
        return self._parse_sif(name, sif_path)

    def _download_problem(self, name: str) -> None:
        """Download a problem from the repository."""
        url = f"{BASE_URL}/{name}.SIF"
        sif_path = self.cache_dir / f"{name}.SIF"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        sif_path.write_bytes(response.content)

    def _parse_sif(self, name: str, path: Path) -> BenchmarkProblem:
        """Parse a SIF/QPS file into a BenchmarkProblem."""
        with open(path, "r") as f:
            return parse_qps(name, f, source=self.name())


def parse_qps(name: str, f: TextIO, source: str = "qps") -> BenchmarkProblem:
    """Parse a QPS format file.

    QPS is an extension of MPS format for quadratic programs.
    Sections: NAME, ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUADOBJ/QMATRIX

    The standard form is:
        min  0.5 x'Qx + c'x
        s.t. l_A <= Ax <= u_A
             l <= x <= u
    """
    # Data structures for parsing
    row_names: list[str] = []
    row_types: dict[str, str] = {}  # 'N', 'E', 'L', 'G'
    obj_row: str | None = None

    col_names: list[str] = []
    col_indices: dict[str, int] = {}

    # Triplets for A matrix
    a_triplets: list[tuple[int, int, float]] = []
    # Triplets for Q matrix (upper triangle)
    q_triplets: list[tuple[int, int, float]] = []

    c: dict[int, float] = {}  # Linear objective coefficients
    rhs: dict[str, float] = {}  # Right-hand side values
    ranges: dict[str, float] = {}  # Range values for ranged rows
    bounds_lo: dict[int, float] = {}
    bounds_up: dict[int, float] = {}

    current_section = None
    content = f.read()

    # Parse line by line
    for line in content.split("\n"):
        line = line.rstrip()
        if not line or line.startswith("*"):
            continue

        # Check for section headers
        if line.startswith("NAME"):
            current_section = "NAME"
            continue
        elif line.startswith("ROWS"):
            current_section = "ROWS"
            continue
        elif line.startswith("COLUMNS"):
            current_section = "COLUMNS"
            continue
        elif line.startswith("RHS"):
            current_section = "RHS"
            continue
        elif line.startswith("RANGES"):
            current_section = "RANGES"
            continue
        elif line.startswith("BOUNDS"):
            current_section = "BOUNDS"
            continue
        elif line.startswith("QUADOBJ") or line.startswith("QMATRIX"):
            current_section = "QUADOBJ"
            continue
        elif line.startswith("ENDATA"):
            break

        # Parse section content
        if current_section == "ROWS":
            parts = line.split()
            if len(parts) >= 2:
                row_type, row_name = parts[0], parts[1]
                row_types[row_name] = row_type
                if row_type == "N":
                    if obj_row is None:
                        obj_row = row_name
                else:
                    row_names.append(row_name)

        elif current_section == "COLUMNS":
            parts = line.split()
            if len(parts) >= 3:
                col_name = parts[0]
                if col_name not in col_indices:
                    col_indices[col_name] = len(col_names)
                    col_names.append(col_name)
                col_idx = col_indices[col_name]

                # Process pairs of (row_name, value)
                i = 1
                while i + 1 < len(parts):
                    row_name = parts[i]
                    try:
                        value = float(parts[i + 1])
                    except ValueError:
                        break

                    if row_name == obj_row:
                        c[col_idx] = c.get(col_idx, 0.0) + value
                    elif row_name in row_types:
                        row_idx = row_names.index(row_name)
                        a_triplets.append((row_idx, col_idx, value))

                    i += 2

        elif current_section == "RHS":
            parts = line.split()
            if len(parts) >= 2:
                i = 1
                while i + 1 < len(parts):
                    row_name = parts[i]
                    try:
                        value = float(parts[i + 1])
                    except ValueError:
                        break
                    rhs[row_name] = value
                    i += 2

        elif current_section == "RANGES":
            parts = line.split()
            if len(parts) >= 2:
                i = 1
                while i + 1 < len(parts):
                    row_name = parts[i]
                    try:
                        value = float(parts[i + 1])
                    except ValueError:
                        break
                    ranges[row_name] = value
                    i += 2

        elif current_section == "BOUNDS":
            parts = line.split()
            if len(parts) >= 3:
                bound_type = parts[0]
                col_name = parts[2]
                if col_name in col_indices:
                    col_idx = col_indices[col_name]
                    if len(parts) >= 4:
                        try:
                            value = float(parts[3])
                        except ValueError:
                            value = 0.0
                    else:
                        value = 0.0

                    if bound_type == "LO":
                        bounds_lo[col_idx] = value
                    elif bound_type == "UP":
                        bounds_up[col_idx] = value
                    elif bound_type == "FX":
                        bounds_lo[col_idx] = value
                        bounds_up[col_idx] = value
                    elif bound_type == "FR":
                        bounds_lo[col_idx] = -np.inf
                        bounds_up[col_idx] = np.inf
                    elif bound_type == "MI":
                        bounds_lo[col_idx] = -np.inf
                    elif bound_type == "PL":
                        bounds_up[col_idx] = np.inf
                    elif bound_type == "BV":
                        bounds_lo[col_idx] = 0.0
                        bounds_up[col_idx] = 1.0

        elif current_section == "QUADOBJ":
            parts = line.split()
            if len(parts) >= 3:
                col1 = parts[0]
                col2 = parts[1]
                try:
                    value = float(parts[2])
                except ValueError:
                    continue

                if col1 in col_indices and col2 in col_indices:
                    i = col_indices[col1]
                    j = col_indices[col2]
                    # Store upper triangle only
                    if i <= j:
                        q_triplets.append((i, j, value))
                    else:
                        q_triplets.append((j, i, value))

    # Build matrices
    n_vars = len(col_names)
    n_rows = len(row_names)

    # Build A matrix
    if a_triplets:
        rows, cols, data = zip(*a_triplets)
        A = sparse.csc_matrix((data, (rows, cols)), shape=(n_rows, n_vars))
    else:
        A = sparse.csc_matrix((n_rows, n_vars))

    # Build Q matrix (symmetric from upper triangle)
    P: sparse.csc_matrix | None = None
    if q_triplets:
        rows, cols, data = zip(*q_triplets)
        Q = sparse.csc_matrix((data, (rows, cols)), shape=(n_vars, n_vars))
        # Make symmetric
        P = (Q + Q.T).tocsc()
        # Diagonal was doubled, fix it
        diag_idx = np.arange(n_vars)
        P[diag_idx, diag_idx] = Q[diag_idx, diag_idx]
        P = P.tocsc()

    # Build c vector
    q_vec = np.zeros(n_vars)
    for i, v in c.items():
        q_vec[i] = v

    # Build b vector and convert to conic form
    # We need to convert from standard form to conic form:
    # Ax + s = b, s in K
    b_vec = np.zeros(n_rows)
    cones: list[tuple[str, int]] = []

    # Track which rows need slack variables
    equality_rows: list[int] = []
    inequality_rows: list[int] = []

    for i, row_name in enumerate(row_names):
        row_type = row_types.get(row_name, "E")
        b_val = rhs.get(row_name, 0.0)

        if row_type == "E":  # Equality: Ax = b
            b_vec[i] = b_val
            equality_rows.append(i)
        elif row_type == "L":  # Less than: Ax <= b => Ax + s = b, s >= 0
            b_vec[i] = b_val
            inequality_rows.append(i)
        elif row_type == "G":  # Greater than: Ax >= b => -Ax + s = -b, s >= 0
            b_vec[i] = -b_val
            # Negate the row of A
            A[i, :] = -A[i, :]
            inequality_rows.append(i)

    # Add variable bounds as additional constraints
    # For now, we'll include them as part of the cone structure
    bound_rows: list[int] = []
    bound_A_triplets: list[tuple[int, int, float]] = []
    bound_b: list[float] = []

    for j in range(n_vars):
        lo = bounds_lo.get(j, 0.0)  # Default lower bound is 0
        up = bounds_up.get(j, np.inf)  # Default upper bound is inf

        if lo > -np.inf:
            # x >= lo => -x + s = -lo, s >= 0
            row_idx = n_rows + len(bound_rows)
            bound_A_triplets.append((len(bound_rows), j, -1.0))
            bound_b.append(-lo)
            bound_rows.append(row_idx)
        if up < np.inf:
            # x <= up => x + s = up, s >= 0
            bound_A_triplets.append((len(bound_rows), j, 1.0))
            bound_b.append(up)
            bound_rows.append(len(bound_rows))

    # Combine A with bound constraints
    if bound_A_triplets:
        n_bound_rows = len(bound_b)
        rows, cols, data = zip(*bound_A_triplets)
        A_bounds = sparse.csc_matrix((data, (rows, cols)), shape=(n_bound_rows, n_vars))
        A = sparse.vstack([A, A_bounds]).tocsc()
        b_vec = np.concatenate([b_vec, np.array(bound_b)])
        inequality_rows.extend(range(n_rows, n_rows + n_bound_rows))

    n_constraints = len(b_vec)

    # Build cone specification
    n_eq = len(equality_rows)
    n_ineq = len(inequality_rows)

    if n_eq > 0:
        cones.append(("zero", n_eq))
    if n_ineq > 0:
        cones.append(("nonneg", n_ineq))

    # Reorder rows so equality constraints come first
    if n_eq > 0 and n_ineq > 0:
        new_order = equality_rows + inequality_rows
        A = A[new_order, :]
        b_vec = b_vec[new_order]

    return BenchmarkProblem(
        name=name,
        source=source,
        n_vars=n_vars,
        n_constraints=n_constraints,
        P=P,
        q=q_vec,
        A=A,
        b=b_vec,
        cones=cones,
        problem_class="QP" if P is not None else "LP",
    )
