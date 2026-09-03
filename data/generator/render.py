"""Small Markdown helpers.

Documents are written with ``## `` section headings so the Phase 3 token-based
chunker splits them on semantic boundaries rather than mid-sentence.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping


def iso(ts: dt.datetime) -> str:
    """UTC ISO-8601 with an explicit offset, seconds precision."""
    return ts.astimezone(dt.UTC).replace(microsecond=0).isoformat()


def human_date(ts: dt.datetime) -> str:
    """e.g. "March 4, 2025" — platform-independent (no %-d)."""
    d = ts.astimezone(dt.UTC)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def heading(text: str, level: int = 2) -> str:
    return f"{'#' * level} {text}"


def bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def numbered(items: Iterable[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def kv_table(rows: Mapping[str, str]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    lines.extend(f"| {k} | {v} |" for k, v in rows.items())
    return "\n".join(lines)


def code_block(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


def join_sections(*sections: str) -> str:
    return "\n\n".join(s.strip() for s in sections if s and s.strip()) + "\n"
