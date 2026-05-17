"""Scraper registry — discover implementations by string ID."""

from __future__ import annotations

from .base import JobScraper

SCRAPER_REGISTRY: dict[str, type[JobScraper]] = {}


def register_scraper(cls: type[JobScraper]) -> type[JobScraper]:
    """Class decorator (or plain function) to register a scraper."""
    if not cls.source_id:
        raise ValueError(f"{cls.__name__} has no source_id.")
    if cls.source_id in SCRAPER_REGISTRY:
        raise ValueError(f"Scraper '{cls.source_id}' already registered.")
    SCRAPER_REGISTRY[cls.source_id] = cls
    return cls


def get_scraper(source_id: str) -> JobScraper:
    """Instantiate a scraper by ID."""
    try:
        cls = SCRAPER_REGISTRY[source_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown scraper '{source_id}'. Available: {sorted(SCRAPER_REGISTRY)}"
        ) from exc
    return cls()
