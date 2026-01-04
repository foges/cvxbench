"""Loader for SDPLIB semidefinite programming test problems.

SDPLIB is a collection of 92 SDP test problems in SDPA sparse format.
Source: https://github.com/vsdp/SDPLIB
"""

from __future__ import annotations

from typing import Any

import numpy as np
import requests
from scipy import sparse

from cvxbench.loaders.base import BenchmarkLoader, BenchmarkProblem

# SDPLIB problems from https://github.com/vsdp/SDPLIB
SDPLIB_PROBLEMS = [
    # Architecture/Topology
    "arch0", "arch2", "arch4", "arch8",
    # Control systems
    "control1", "control2", "control3", "control4", "control5",
    "control6", "control7", "control8", "control9", "control10", "control11",
    # Graph partitioning
    "equalG11", "equalG51",
    "gpp100", "gpp124-1", "gpp124-2", "gpp124-3", "gpp124-4",
    "gpp250-1", "gpp250-2", "gpp250-3", "gpp250-4",
    "gpp500-1", "gpp500-2", "gpp500-3", "gpp500-4",
    # Hinf control
    "hinf1", "hinf2", "hinf3", "hinf4", "hinf5", "hinf6", "hinf7", "hinf8",
    "hinf9", "hinf10", "hinf11", "hinf12", "hinf13", "hinf14", "hinf15",
    # Infeasibility examples
    "infd1", "infd2", "infp1", "infp2",
    # Max-cut
    "maxG11", "maxG32", "maxG51", "maxG55", "maxG60",
    "mcp100", "mcp124-1", "mcp124-2", "mcp124-3", "mcp124-4",
    "mcp250-1", "mcp250-2", "mcp250-3", "mcp250-4",
    "mcp500-1", "mcp500-2", "mcp500-3", "mcp500-4",
    # Quadratic assignment
    "qap5", "qap6", "qap7", "qap8", "qap9", "qap10",
    "qpG11", "qpG51",
    # Lovász theta
    "theta1", "theta2", "theta3", "theta4", "theta5", "theta6",
    "thetaG11", "thetaG51",
    # Truss topology
    "truss1", "truss2", "truss3", "truss4", "truss5", "truss6", "truss7", "truss8",
    # Structural
    "ss30",
]

BASE_URL = "https://raw.githubusercontent.com/vsdp/SDPLIB/master/data/"


def parse_sdpa_sparse(content: str) -> dict[str, Any]:
    """Parse SDPA sparse format (.dat-s) file.

    SDPA format represents:
        min  c^T x
        s.t. sum(x_i * F_i) - F_0 >= 0  (PSD)

    Which is equivalent to:
        min  <C, X>
        s.t. <A_i, X> = b_i  for i = 1..m
             X >= 0  (PSD)

    Args:
        content: File content as string.

    Returns:
        Dictionary with C, A_list, b, block_sizes.
    """
    lines = []
    for line in content.strip().split("\n"):
        line = line.strip()
        # Skip comments (lines starting with " or * or empty)
        if not line or line.startswith('"') or line.startswith("*"):
            continue
        lines.append(line)

    if len(lines) < 4:
        msg = f"SDPA file too short: {len(lines)} lines"
        raise ValueError(msg)

    # Line 1: m (number of constraint matrices, i.e., number of primal variables)
    m = int(lines[0])

    # Line 2: nBlocks (number of blocks)
    n_blocks = int(lines[1])

    # Line 3: block sizes (negative means diagonal block)
    # Can be comma or space separated, with optional braces
    block_line = lines[2].replace("{", "").replace("}", "").replace(",", " ")
    block_sizes = [int(x) for x in block_line.split()]

    if len(block_sizes) != n_blocks:
        msg = f"Expected {n_blocks} block sizes, got {len(block_sizes)}"
        raise ValueError(msg)

    # Line 4: cost vector c (m elements)
    # Can be comma or space separated, with optional braces
    c_line = lines[3].replace("{", "").replace("}", "").replace(",", " ")
    c = np.array([float(x) for x in c_line.split()])

    if len(c) != m:
        msg = f"Expected {m} cost coefficients, got {len(c)}"
        raise ValueError(msg)

    # Remaining lines: matrix entries
    # Format: matno blkno i j value
    # matno=0 is F_0 (constant term), matno=1..m are F_1..F_m

    # Initialize matrices for each block
    # F_matrices[matno][blkno] = dict of (i,j) -> value
    f_matrices: dict[int, dict[int, dict[tuple[int, int], float]]] = {}

    for line in lines[4:]:
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            continue

        matno = int(parts[0])
        blkno = int(parts[1])
        i = int(parts[2])
        j = int(parts[3])
        value = float(parts[4])

        if matno not in f_matrices:
            f_matrices[matno] = {}
        if blkno not in f_matrices[matno]:
            f_matrices[matno][blkno] = {}

        # Store both (i,j) and (j,i) for symmetric matrices
        f_matrices[matno][blkno][(i, j)] = value
        if i != j:
            f_matrices[matno][blkno][(j, i)] = value

    return {
        "m": m,
        "n_blocks": n_blocks,
        "block_sizes": block_sizes,
        "c": c,
        "f_matrices": f_matrices,
    }


def sdpa_to_conic(parsed: dict[str, Any], name: str) -> BenchmarkProblem:
    """Convert parsed SDPA data to conic BenchmarkProblem.

    The SDPA primal form is:
        min  c^T x               (x is m-dimensional)
        s.t. X = F_0 + sum(x_i * F_i) >= 0  (PSD)

    We convert to conic form:
        min  c^T x
        s.t. A x + s = b
             s ∈ K  (where K is PSD/nonneg cones)

    Where:
        - x has dimension m (the scalar decision variables)
        - s = vec(X) has dimension = total vectorized matrix size
        - A[row, i] = -vec(F_i)[row]  (negative because s = b - Ax)
        - b = vec(F_0)
        - K is the product of PSD/nonneg cones from block structure

    Args:
        parsed: Parsed SDPA data.
        name: Problem name.

    Returns:
        BenchmarkProblem in conic form.
    """
    m = parsed["m"]  # Number of scalar decision variables
    block_sizes = parsed["block_sizes"]
    c = parsed["c"]  # Objective coefficients
    f_matrices = parsed["f_matrices"]

    # Compute total constraint/slack dimension (vectorized matrix size)
    total_cone_dim = 0
    block_offsets = []
    for size in block_sizes:
        block_offsets.append(total_cone_dim)
        if size < 0:
            # Diagonal block: -size indicates size of diagonal
            total_cone_dim += abs(size)
        else:
            # Dense symmetric block: size x size vectorized
            total_cone_dim += size * size

    n_vars = m  # Decision variables are the m scalars

    # Build objective: q = c
    q = c.copy()

    # Build RHS b = vec(F_0)
    b = np.zeros(total_cone_dim)
    if 0 in f_matrices:
        for blkno, entries in f_matrices[0].items():
            blk_idx = blkno - 1  # 1-indexed to 0-indexed
            blk_size = block_sizes[blk_idx]
            offset = block_offsets[blk_idx]

            for (i, j), val in entries.items():
                if blk_size < 0:
                    # Diagonal block
                    if i == j:
                        b[offset + i - 1] = val
                else:
                    # Dense block: column-major vectorization
                    idx = offset + (j - 1) * blk_size + (i - 1)
                    b[idx] = val

    # Build constraint matrix A
    # For constraint: A x + s = b with s = vec(X) in cone
    # We have: vec(X) = vec(F_0) + sum(x_i * vec(F_i))
    # So: s = b + sum(x_i * vec(F_i))
    # Rearranging: -sum(x_i * vec(F_i)) + s = b
    # Thus: A[row, i] = -vec(F_i)[row]
    rows = []
    cols = []
    data = []

    for mat_idx in range(1, m + 1):
        if mat_idx not in f_matrices:
            continue

        for blkno, entries in f_matrices[mat_idx].items():
            blk_idx = blkno - 1
            blk_size = block_sizes[blk_idx]
            offset = block_offsets[blk_idx]

            for (i, j), val in entries.items():
                if blk_size < 0:
                    # Diagonal block
                    if i == j:
                        row = offset + i - 1
                        rows.append(row)
                        cols.append(mat_idx - 1)  # Column = variable index
                        data.append(-val)  # Negative for A x + s = b form
                else:
                    # Dense block: column-major vectorization
                    row = offset + (j - 1) * blk_size + (i - 1)
                    rows.append(row)
                    cols.append(mat_idx - 1)
                    data.append(-val)

    A = sparse.csc_matrix((data, (rows, cols)), shape=(total_cone_dim, n_vars))

    # Build cones list
    # Each block is either a PSD cone or a nonneg cone (for diagonal blocks)
    cones: list[tuple[str, int]] = []
    for size in block_sizes:
        if size < 0:
            # Diagonal block = nonneg cone
            cones.append(("nonneg", abs(size)))
        else:
            # Dense block = PSD cone (n^2 for vectorized form)
            cones.append(("psd", size * size))

    return BenchmarkProblem(
        name=name,
        source="sdplib",
        n_vars=n_vars,
        n_constraints=total_cone_dim,
        P=None,  # Linear objective
        q=q,
        A=A,
        b=b,
        cones=cones,
        problem_class="SDP",
    )


class SDPLIBLoader(BenchmarkLoader):
    """Loader for SDPLIB semidefinite programming test problems."""

    def name(self) -> str:
        """Return loader name."""
        return "sdplib"

    def list_problems(self) -> list[str]:
        """Return list of available problems."""
        return SDPLIB_PROBLEMS.copy()

    def load_problem(self, name: str) -> BenchmarkProblem:
        """Load a problem by name.

        Args:
            name: Problem name (e.g., "control1").

        Returns:
            BenchmarkProblem instance.
        """
        if name not in SDPLIB_PROBLEMS:
            msg = f"Unknown problem: {name}"
            raise ValueError(msg)

        # Check cache
        cache_path = self.cache_dir / f"{name}.dat-s"
        if cache_path.exists():
            content = cache_path.read_text()
        else:
            # Download from GitHub
            url = f"{BASE_URL}{name}.dat-s"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.text

            # Cache it
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(content)

        # Parse and convert
        parsed = parse_sdpa_sparse(content)
        return sdpa_to_conic(parsed, name)

    def estimated_size_mb(self) -> float:
        """Estimate total download size in MB."""
        return 5.0  # SDPLIB is small
