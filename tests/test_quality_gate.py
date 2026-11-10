"""Tests for the quality gate itself.

A gate nobody tests becomes a gate everybody disables.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_quality as gate  # noqa: E402

BUDGETS = {
    "default_file_max": 50,
    "default_file_warn": 40,
    "default_function_max": 10,
    "default_function_warn": 8,
    "files": {},
    "functions": {},
    "vendored": {},
}


def test_gate_passes_on_the_current_repository() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_quality.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_oversized_file_fails(tmp_path) -> None:
    path = tmp_path / "big.py"
    path.write_text("x = 1\n" * 60)
    report = gate.Report()

    gate.check_file_sizes([path], BUDGETS, report, root=tmp_path)

    assert report.failures
    assert "extract a module" in report.failures[0]


def test_file_at_the_cap_passes(tmp_path) -> None:
    path = tmp_path / "ok.py"
    path.write_text("x = 1\n" * 50)
    report = gate.Report()

    gate.check_file_sizes([path], BUDGETS, report, root=tmp_path)

    assert not report.failures


def test_per_file_cap_overrides_the_default(tmp_path) -> None:
    path = tmp_path / "legacy.py"
    path.write_text("x = 1\n" * 60)
    budgets = {**BUDGETS, "files": {"legacy.py": 80}}
    report = gate.Report()

    gate.check_file_sizes([path], budgets, report, root=tmp_path)

    assert not report.failures


def test_oversized_function_fails(tmp_path) -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(20))
    path = tmp_path / "fn.py"
    path.write_text(f"def big():\n{body}\n")
    report = gate.Report()

    gate.check_function_sizes([path], BUDGETS, report, root=tmp_path)

    assert report.failures
    assert "big()" in report.failures[0]


def test_modified_vendored_file_fails(tmp_path, monkeypatch) -> None:
    vendored = tmp_path / "server" / "jarvis" / "client.py"
    vendored.parent.mkdir(parents=True)
    vendored.write_text("print('edited by an agent')\n")
    budgets = {**BUDGETS, "vendored": {"server/jarvis/client.py": "0" * 64}}
    report = gate.Report()
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    gate.check_vendored_checksums(budgets, report)

    assert report.failures
    assert "re-vendor" in report.failures[0]


def test_forbidden_import_fails(tmp_path) -> None:
    path = tmp_path / "server" / "main.py"
    path.parent.mkdir(parents=True)
    path.write_text("from server.jarvis import JarvisLabsClient\n")
    report = gate.Report()

    gate.check_import_layers([path], report, root=tmp_path)

    assert report.failures
    assert "server.jarvis" in report.failures[0]


def test_allowed_import_passes(tmp_path) -> None:
    path = tmp_path / "server" / "gpu.py"
    path.parent.mkdir(parents=True)
    path.write_text("from server.jarvis import JarvisLabsClient\n")
    report = gate.Report()

    gate.check_import_layers([path], report, root=tmp_path)

    assert not report.failures


@pytest.mark.parametrize(
    ("module", "forbidden", "expected"),
    [
        ("server.jarvis", "server.jarvis", True),
        ("server.jarvis.client", "server.jarvis", True),
        ("server.jarvisx", "server.jarvis", False),
        ("server.domain.jobs", "server.", True),
        ("fastapi", "server.", False),
    ],
)
def test_matches(module: str, forbidden: str, expected: bool) -> None:
    assert gate.matches(module, forbidden) is expected
