"""
Integration test: dynamic sampler smoke test via run_tests.py --quick.

This wraps the dev/task_004_dynamic/dev_dynamic/run_tests.py script and
asserts exit code 0. It exercises the full dynamic pipeline (base run +
batch loop + combine + stop) on a 2D Gaussian and runs the F1-F9
faithfulness unit tests.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_dynamic_quick_matrix_passes():
    run_tests = (Path(__file__).resolve().parents[2]
                 / "dev" / "task_004_dynamic" / "dev_dynamic" / "run_tests.py")
    if not run_tests.exists():
        pytest.skip(f"run_tests.py not found at {run_tests}")
    result = subprocess.run(
        [sys.executable, str(run_tests), "--quick"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    assert result.returncode == 0, "run_tests.py --quick failed"
    assert "Final verdict: PASS" in result.stdout
