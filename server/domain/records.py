"""Shared SQLite row mapping for the domain dataclasses.

The classes have different fields but the same shape, so one helper keeps the
near-duplicate detector from flagging two copies of the same loop.
"""

from __future__ import annotations

from dataclasses import fields


def build_from_row(cls, row):
    """Build a dataclass from a ``sqlite3.Row`` by field name."""
    return cls(**{field.name: row[field.name] for field in fields(cls)})
