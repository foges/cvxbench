"""Loader for the SMP (Sparse Mathematical Programming) repository.

The SMP repository contains QP problems from graphics, animation, and simulation
applications. Problems are stored in YAML format with embedded MatrixMarket matrices.

Reference: https://nasoq.github.io/smp.html
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import TextIO

import numpy as np
import requests
from scipy import sparse
from scipy.io import mmread

from cvxbench.loaders.base import BenchmarkLoader, BenchmarkProblem

# Download URLs for SMP problem groups (from Google Drive)
# These are the main problem groups from the SMP repository
SMP_DOWNLOADS = {
    "cube": "1DPsUDJmtKNGtHnQJ9G8WiIDcnljVovu7",
    "beam": "1Q83GZF9SFGPoyMZZU9Fzj45xNAzD4bYr",
    "simulation3d": "1OUs8d0cmVqpdsVOGuOP-xAWHg9ihBK7Y",
    "contact": "1_5HoBH7eD4x4p0B9mz0X5LXxL0H5Y3D4",
}

# Fallback: Direct GitHub URLs for a few sample problems
GITHUB_SMP_BASE = "https://raw.githubusercontent.com/sympiler/nasoq-benchmarks/master/SMP_Repository"
GITHUB_SAMPLE_PROBLEMS = ["matt_qp1", "test05_0"]


class SMPLoader(BenchmarkLoader):
    """Loader for the SMP QP repository (~1515 problems)."""

    def __init__(self, cache_dir: Path | None = None, groups: list[str] | None = None) -> None:
        """Initialize the SMP loader.

        Args:
            cache_dir: Directory to cache downloaded data.
            groups: Optional list of problem groups to load. If None, loads all.
        """
        super().__init__(cache_dir)
        self.groups = groups or list(SMP_DOWNLOADS.keys())
        self._problem_cache: dict[str, Path] | None = None

    def name(self) -> str:
        return "smp"

    def estimated_size_mb(self) -> float:
        return 500.0  # Approximate size of full SMP dataset

    def list_problems(self) -> list[str]:
        """Return list of available problem names."""
        self._ensure_index()
        assert self._problem_cache is not None
        return sorted(self._problem_cache.keys())

    def _ensure_index(self) -> None:
        """Build index of available problems."""
        if self._problem_cache is not None:
            return

        self._problem_cache = {}

        # Scan cache directory for .yml files
        if self.cache_dir.exists():
            for yml_file in self.cache_dir.rglob("*.yml"):
                name = yml_file.stem
                self._problem_cache[name] = yml_file

    def ensure_downloaded(self) -> None:
        """Download SMP problems if not already cached."""
        # First try to download from Google Drive groups
        for group in self.groups:
            if group not in SMP_DOWNLOADS:
                continue

            group_dir = self.cache_dir / group
            if group_dir.exists() and any(group_dir.glob("*.yml")):
                continue  # Already downloaded

            self._download_group(group)

        # Also download sample problems from GitHub as fallback
        self._download_github_samples()

        # Rebuild index after download
        self._problem_cache = None
        self._ensure_index()

    def _download_github_samples(self) -> None:
        """Download sample problems from GitHub."""
        samples_dir = self.cache_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        for name in GITHUB_SAMPLE_PROBLEMS:
            out_path = samples_dir / f"{name}.yml"
            if out_path.exists():
                continue

            url = f"{GITHUB_SMP_BASE}/{name}.yml"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                out_path.write_bytes(response.content)
                print(f"  Downloaded {name} from GitHub")
            except Exception as e:
                print(f"  Warning: Failed to download {name}: {e}")

    def _download_group(self, group: str) -> None:
        """Download a problem group from Google Drive."""
        drive_id = SMP_DOWNLOADS[group]
        url = f"https://drive.google.com/uc?export=download&id={drive_id}"

        print(f"Downloading SMP group: {group}...")

        try:
            # Google Drive may require confirmation for large files
            session = requests.Session()
            response = session.get(url, stream=True, timeout=60)

            # Check for download confirmation page
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    url = f"https://drive.google.com/uc?export=download&confirm={value}&id={drive_id}"
                    response = session.get(url, stream=True, timeout=60)
                    break

            response.raise_for_status()

            # Save and extract
            group_dir = self.cache_dir / group
            group_dir.mkdir(parents=True, exist_ok=True)

            content = response.content

            # Try to extract as zip
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for member in zf.namelist():
                        if member.endswith(".yml"):
                            # Extract to group directory
                            data = zf.read(member)
                            out_path = group_dir / Path(member).name
                            out_path.write_bytes(data)
            except zipfile.BadZipFile:
                # Not a zip file, save as single yml
                out_path = group_dir / f"{group}.yml"
                out_path.write_bytes(content)

            print(f"  Downloaded {group} to {group_dir}")

        except Exception as e:
            print(f"  Warning: Failed to download {group}: {e}")

    def load_problem(self, name: str) -> BenchmarkProblem:
        """Load a problem by name."""
        self._ensure_index()
        assert self._problem_cache is not None

        if name not in self._problem_cache:
            # Try to download if not found
            self.ensure_downloaded()
            if name not in self._problem_cache:
                msg = f"Unknown problem: {name}. Use list_problems() to see available problems."
                raise ValueError(msg)

        yml_path = self._problem_cache[name]
        return self._parse_smp_yaml(name, yml_path)

    def _parse_smp_yaml(self, name: str, path: Path) -> BenchmarkProblem:
        """Parse an SMP YAML file into a BenchmarkProblem."""
        with open(path, "r") as f:
            return parse_smp(name, f.read(), source=self.name())


def parse_smp(name: str, content: str, source: str = "smp") -> BenchmarkProblem:
    """Parse SMP YAML format content.

    The SMP format represents QP problems as:
        minimize    1/2 x^T H x + q^T x
        subject to  A x = b       (equality constraints from "Fixed")
                    l <= C x <= u (inequality constraints from "Inequality")

    Args:
        name: Problem name.
        content: YAML content string.
        source: Source identifier.

    Returns:
        BenchmarkProblem in conic form.
    """
    # Parse YAML-like format with embedded MatrixMarket
    # The format uses quoted keys like "Description": |
    sections = _parse_smp_sections(content)

    # Parse each section
    description = sections.get("Description", "")

    # Parse quadratic objective H (may be None for LP)
    H = None
    if "Quadratic" in sections:
        H = _parse_matrix_market(sections["Quadratic"])
        if H is not None:
            H = H.tocsc()
            # Ensure symmetric
            if (H - H.T).nnz > 0:
                H = (H + H.T) / 2

    # Parse linear objective q
    q = None
    n_vars = 0
    if "Linear" in sections:
        q = _parse_matrix_market_dense(sections["Linear"])
        if q is not None:
            n_vars = len(q)

    # Infer n_vars from H if q not available
    if n_vars == 0 and H is not None:
        n_vars = H.shape[0]

    if n_vars == 0:
        msg = f"Cannot determine problem dimension for {name}"
        raise ValueError(msg)

    # Default q to zeros
    if q is None:
        q = np.zeros(n_vars)

    # Parse equality constraints (Fixed)
    A_eq = None
    b_eq = None
    n_eq = 0
    if "Fixed" in sections:
        fixed_content = sections["Fixed"].strip()
        if fixed_content and fixed_content != "0":
            A_eq = _parse_matrix_market(fixed_content)
            if A_eq is not None:
                A_eq = A_eq.tocsc()
                n_eq = A_eq.shape[0]
                # For equality constraints, we need b vector
                # In SMP format, b is often stored separately or implicitly zero
                b_eq = np.zeros(n_eq)

    # Parse inequality constraints (Inequality, bounds)
    C = None
    l_bounds = None
    u_bounds = None
    n_ineq = 0

    if "Inequality" in sections:
        C = _parse_matrix_market(sections["Inequality"])
        if C is not None:
            C = C.tocsc()
            n_ineq = C.shape[0]

    if "Inequality l-bounds" in sections:
        l_bounds = _parse_matrix_market_dense(sections["Inequality l-bounds"])

    if "Inequality u-bounds" in sections:
        u_bounds = _parse_matrix_market_dense(sections["Inequality u-bounds"])

    # Build conic form
    # Convert to: A_conic @ x + s = b_conic, s in K
    constraint_blocks = []
    b_blocks = []
    cones: list[tuple[str, int]] = []

    # Equality constraints: A x = b => A x + s = b, s in zero cone
    if A_eq is not None and n_eq > 0:
        constraint_blocks.append(A_eq)
        b_blocks.append(b_eq)
        cones.append(("zero", n_eq))

    # Inequality constraints: l <= Cx <= u
    # Convert to:
    #   Cx <= u  =>  Cx + s = u, s >= 0  (nonneg cone)
    #   Cx >= l  =>  -Cx + s = -l, s >= 0  (nonneg cone)
    if C is not None and n_ineq > 0:
        if u_bounds is not None:
            # Upper bound: Cx <= u
            constraint_blocks.append(C)
            b_blocks.append(u_bounds)
            cones.append(("nonneg", n_ineq))

        if l_bounds is not None:
            # Check if l_bounds are finite (not all -inf)
            finite_l = np.isfinite(l_bounds)
            if np.any(finite_l):
                # Lower bound: Cx >= l => -Cx <= -l => -Cx + s = -l, s >= 0
                constraint_blocks.append(-C)
                b_blocks.append(-l_bounds)
                cones.append(("nonneg", n_ineq))

    # Build final matrices
    if constraint_blocks:
        A_conic = sparse.vstack(constraint_blocks).tocsc()
        b_conic = np.concatenate(b_blocks)
    else:
        # No constraints - create dummy
        A_conic = sparse.csc_matrix((0, n_vars))
        b_conic = np.array([])
        cones = []

    n_constraints = A_conic.shape[0]

    # Ensure cone dimensions sum correctly
    total_cone_dim = sum(dim for _, dim in cones)
    if total_cone_dim != n_constraints:
        msg = f"Cone dimension mismatch: {total_cone_dim} != {n_constraints}"
        raise ValueError(msg)

    return BenchmarkProblem(
        name=name,
        source=source,
        n_vars=n_vars,
        n_constraints=n_constraints,
        P=H,
        q=q,
        A=A_conic,
        b=b_conic,
        cones=cones,
        problem_class="QP" if H is not None else "LP",
        metadata={"description": description},
    )


def _parse_smp_sections(content: str) -> dict[str, str]:
    """Parse SMP YAML-like format into sections.

    The format uses quoted keys followed by | for multiline values:
        "Key": |
          value content
          more content
    """
    sections: dict[str, str] = {}

    # Pattern to match section headers: "Key": |
    header_pattern = re.compile(r'^"([^"]+)":\s*\|?\s*$', re.MULTILINE)

    matches = list(header_pattern.finditer(content))

    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        # Extract content, removing leading indentation
        section_content = content[start:end].strip()
        sections[key] = section_content

    return sections


def _parse_matrix_market(content: str) -> sparse.csc_matrix | None:
    """Parse MatrixMarket format string into sparse matrix.

    Note: SMP format uses 'symmetric' keyword even for non-square matrices,
    which violates the MatrixMarket spec. We handle this by parsing manually.
    """
    if not content or content.strip() == "0":
        return None

    try:
        lines = content.strip().split("\n")
        if not lines:
            return None

        # Parse header
        header = lines[0].strip()
        if not header.startswith("%%MatrixMarket"):
            # Try scipy's mmread for standard format
            matrix = mmread(io.StringIO(content))
            if sparse.issparse(matrix):
                return matrix.tocsc()
            return sparse.csc_matrix(matrix)

        # Check if coordinate or array format
        header_parts = header.lower().split()
        is_coordinate = "coordinate" in header_parts
        is_symmetric = "symmetric" in header_parts

        # Find the dimension line (first non-comment line after header)
        data_start = 1
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            # This is the dimension line
            parts = line.split()
            if is_coordinate:
                m, n, nnz = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                m, n = int(parts[0]), int(parts[1])
                nnz = m * n
            data_start = i + 1
            break
        else:
            return None

        if is_coordinate:
            # Parse coordinate format
            rows, cols, vals = [], [], []
            for line in lines[data_start:]:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r, c, v = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                        rows.append(r)
                        cols.append(c)
                        vals.append(v)
                        # For symmetric, add the transpose entry if not on diagonal
                        # But only if the matrix is square
                        if is_symmetric and m == n and r != c:
                            rows.append(c)
                            cols.append(r)
                            vals.append(v)
                    except (ValueError, IndexError):
                        continue

            if not rows:
                return None

            return sparse.csc_matrix((vals, (rows, cols)), shape=(m, n))
        else:
            # Array format - let scipy handle it
            matrix = mmread(io.StringIO(content))
            if sparse.issparse(matrix):
                return matrix.tocsc()
            return sparse.csc_matrix(matrix)

    except Exception:
        return None


def _parse_matrix_market_dense(content: str) -> np.ndarray | None:
    """Parse MatrixMarket format string into dense array."""
    if not content or content.strip() == "0":
        return None

    try:
        matrix = mmread(io.StringIO(content))
        if sparse.issparse(matrix):
            return matrix.toarray().flatten()
        else:
            return np.asarray(matrix).flatten()
    except Exception:
        return None
