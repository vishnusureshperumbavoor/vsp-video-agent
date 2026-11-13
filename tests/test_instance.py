"""Tests for instance selection across stale JarvisLabs entries."""

from __future__ import annotations

from server.instance import select_instance


def test_honors_the_pinned_machine_id(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_MACHINE_ID", "574097")
    instances = [
        {"machine_id": "573989", "host": "old"},
        {"machine_id": "574097", "host": "new"},
    ]

    selected = select_instance(instances, instances[0])

    assert selected["host"] == "new"


def test_falls_back_when_not_pinned(monkeypatch) -> None:
    monkeypatch.delenv("JARVIS_MACHINE_ID", raising=False)
    fallback = {"machine_id": "1", "host": "fallback"}

    assert select_instance([], fallback) is fallback


def test_falls_back_when_pinned_id_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_MACHINE_ID", "nope")
    fallback = {"machine_id": "1", "host": "fallback"}

    assert select_instance([{"machine_id": "2"}], fallback) is fallback
