"""Tests for baseline save/load/compare."""

import json
import tempfile
from pathlib import Path

import pytest

from cvxbench.baseline import (
    Baseline,
    BaselineEntry,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)
from cvxbench.results import SolveResult


@pytest.fixture
def sample_results() -> list[SolveResult]:
    """Sample solve results for testing."""
    return [
        SolveResult(
            problem="prob1",
            source="test",
            solver="scs",
            run_index=0,
            status="optimal",
            time_sec=0.1,
            iterations=10,
            primal_obj=1.0,
            dual_obj=1.0,
            gap=0.0,
        ),
        SolveResult(
            problem="prob2",
            source="test",
            solver="scs",
            run_index=0,
            status="optimal",
            time_sec=0.2,
            iterations=20,
            primal_obj=2.0,
            dual_obj=2.0,
            gap=0.0,
        ),
        SolveResult(
            problem="prob3",
            source="test",
            solver="scs",
            run_index=0,
            status="failed",
            time_sec=0.3,
            iterations=None,
            primal_obj=None,
            dual_obj=None,
            gap=None,
        ),
    ]


class TestSaveBaseline:
    def test_save_creates_file(self, sample_results):
        """Saving baseline creates JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            result_path = save_baseline(sample_results, "test", path=path)

            assert result_path.exists()
            data = json.loads(result_path.read_text())
            assert data["name"] == "test"
            assert "created" in data
            assert len(data["entries"]) == 2  # Only successful results

    def test_only_saves_successful(self, sample_results):
        """Only successful results are saved to baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_baseline(sample_results, "test", path=path)

            data = json.loads(path.read_text())
            assert "test:prob1" in data["entries"]
            assert "test:prob2" in data["entries"]
            assert "test:prob3" not in data["entries"]


class TestLoadBaseline:
    def test_load_from_path(self, sample_results):
        """Load baseline from file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_baseline(sample_results, "test", path=path)

            baseline = load_baseline(path)
            assert baseline.name == "test"
            assert len(baseline.entries) == 2

    def test_load_not_found(self):
        """Loading nonexistent baseline raises error."""
        with pytest.raises(FileNotFoundError):
            load_baseline("/nonexistent/path.json")


class TestCompareToBaseline:
    def test_no_change(self, sample_results):
        """Compare identical results shows no change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_baseline(sample_results, "test", path=path)
            baseline = load_baseline(path)

            report = compare_to_baseline(sample_results, baseline)
            assert report.n_problems == 2
            assert report.n_faster == 0
            assert report.n_slower == 0
            assert report.n_same == 2
            assert abs(report.time_ratio_geom - 1.0) < 0.01

    def test_detect_speedup(self, sample_results):
        """Detect speedup vs baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_baseline(sample_results, "test", path=path)
            baseline = load_baseline(path)

            # Create faster results
            faster_results = [
                SolveResult(
                    problem="prob1",
                    source="test",
                    solver="scs",
                    run_index=0,
                    status="optimal",
                    time_sec=0.05,  # 2x faster
                    iterations=10,
                    primal_obj=1.0,
                    dual_obj=1.0,
                    gap=0.0,
                ),
                SolveResult(
                    problem="prob2",
                    source="test",
                    solver="scs",
                    run_index=0,
                    status="optimal",
                    time_sec=0.1,  # 2x faster
                    iterations=20,
                    primal_obj=2.0,
                    dual_obj=2.0,
                    gap=0.0,
                ),
            ]

            report = compare_to_baseline(faster_results, baseline)
            assert report.n_faster == 2
            assert report.n_slower == 0
            assert report.time_ratio_geom < 0.6  # ~0.5

    def test_detect_regression(self, sample_results):
        """Detect status regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_baseline(sample_results, "test", path=path)
            baseline = load_baseline(path)

            # Create regressed results
            regressed_results = [
                SolveResult(
                    problem="prob1",
                    source="test",
                    solver="scs",
                    run_index=0,
                    status="failed",  # Was optimal
                    time_sec=0.1,
                    iterations=None,
                    primal_obj=None,
                    dual_obj=None,
                    gap=None,
                ),
            ]

            report = compare_to_baseline(regressed_results, baseline)
            assert report.n_status_regressed == 1
            assert report.has_regressions
