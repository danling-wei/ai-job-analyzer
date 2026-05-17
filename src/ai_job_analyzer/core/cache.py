"""Tiny on-disk cache for raw scraper payloads.

Layout::

    ${CACHE_DIR}/
      serpapi/
        2026-05-15/
          backend-engineer.json
          backend-engineer.meta.json
        ...

Cached files are *advisory* — scrapers can ignore the cache, write through it,
or read it back when offline / debugging. The cache deliberately stores raw
upstream payloads (not normalised :class:`JobPosting` objects) so we can
re-parse historical data with newer parsers.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import get_settings

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 64) -> str:
    """Produce a filesystem-safe slug from arbitrary text."""
    cleaned = _SLUG_RE.sub("-", text.lower()).strip("-")
    if not cleaned:
        cleaned = "_"
    if len(cleaned) > max_len:
        digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[: max_len - 9]}-{digest}"
    return cleaned


def cache_dir(source: str, *, day: str | None = None) -> Path:
    """Return (and create) the cache directory for ``source`` on ``day``."""
    settings = get_settings()
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    path = Path(settings.cache_dir) / slugify(source) / day
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(source: str, key: str, *, suffix: str = ".json", day: str | None = None) -> Path:
    """Compute the canonical cache file path for ``(source, key)``."""
    return cache_dir(source, day=day) / f"{slugify(key)}{suffix}"


def write_json(path: Path, payload: Any, *, meta: dict[str, Any] | None = None) -> None:
    """Write ``payload`` as JSON, plus a sidecar ``.meta.json``."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(),
                **(meta or {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any | None:
    """Return the parsed JSON at ``path`` or ``None`` if missing/invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
