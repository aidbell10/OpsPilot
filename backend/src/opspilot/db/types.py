"""Reusable column types."""

from __future__ import annotations

import enum

from sqlalchemy import Enum as SAEnum


def str_enum[E: enum.Enum](enum_cls: type[E], *, length: int = 32) -> SAEnum:
    """A non-native (VARCHAR + CHECK) enum that stores ``member.value``.

    Portable, migration-friendly, and avoids Postgres native ENUM types whose
    ``ALTER TYPE ... ADD VALUE`` cannot run inside a transaction.
    """
    return SAEnum(
        enum_cls,
        native_enum=False,
        length=length,
        name=f"{enum_cls.__name__.lower()}_enum",
        values_callable=lambda e: [str(m.value) for m in e],
        validate_strings=True,
    )
