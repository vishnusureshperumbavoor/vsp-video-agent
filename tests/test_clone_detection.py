"""Tests for the near-duplicate detector.

The failure mode it exists to catch: an agent copies a function, renames the
variables and changes a constant, and the two copies drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from clone_detection import normalize, scan  # noqa: E402

SETTINGS = {"window": 10, "min_tokens": 20, "min_lines": 3}

COPY_A = '''
def process(items, limit):
    total = 0
    for item in items:
        if item > limit:
            total += item
    results = []
    for item in items:
        results.append(item * 2)
    return total, results
'''

# Same structure, different names and literals.
COPY_B = '''
def transform(records, cutoff):
    acc = 0
    for record in records:
        if record > cutoff:
            acc += record
    output = []
    for record in records:
        output.append(record * 4)
    return acc, output
'''

UNRELATED = '''
def write_header(path, title):
    with open(path, "w") as handle:
        handle.write(title.upper())
        handle.write("\\n")
    return path
'''


def test_renamed_copy_is_detected(tmp_path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text(COPY_A)
    second.write_text(COPY_B)

    report = scan([first, second], **SETTINGS)

    assert report.regions, "a renamed copy must be reported"
    assert report.duplicate_lines > 0
    assert {report.regions[0].file_a, report.regions[0].file_b} == {
        str(first),
        str(second),
    }


def test_unrelated_functions_are_not_reported(tmp_path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text(COPY_A)
    second.write_text(UNRELATED)

    report = scan([first, second], **SETTINGS)

    assert report.regions == []
    assert report.duplicate_lines == 0


def test_single_file_is_not_compared_with_itself(tmp_path) -> None:
    only = tmp_path / "a.py"
    only.write_text(COPY_A)

    report = scan([only], **SETTINGS)

    assert report.regions == []


def test_excluded_directories_are_skipped(tmp_path, monkeypatch) -> None:
    import clone_detection

    monkeypatch.setattr(clone_detection, "ROOT", tmp_path)
    first = tmp_path / "tests" / "a.py"
    second = tmp_path / "tests" / "b.py"
    first.parent.mkdir()
    first.write_text(COPY_A)
    second.write_text(COPY_B)

    report = scan([first, second], **SETTINGS)

    assert report.regions == []


def test_normalize_collapses_names_numbers_and_strings() -> None:
    import tokenize

    assert normalize("my_variable", tokenize.NAME) == "N"
    assert normalize("for", tokenize.NAME) == "for"
    assert normalize("self", tokenize.NAME) == "self"
    assert normalize("1234", tokenize.NUMBER) == "0"
    assert normalize('"hello"', tokenize.STRING) == "S"
    assert normalize("+", tokenize.OP) == "+"
