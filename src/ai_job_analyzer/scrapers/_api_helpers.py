"""Small helpers shared by API-backed job scrapers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup


def strip_html(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


def parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def first_apply_url(raw: dict[str, Any]) -> str | None:
    options = raw.get("apply_options")
    if isinstance(options, list):
        for option in options:
            if isinstance(option, dict) and option.get("link"):
                return str(option["link"])
    return None
