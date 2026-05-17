"""Job board scrapers.

Each scraper implements :class:`~ai_job_analyzer.scrapers.base.JobScraper` and is
registered into :data:`SCRAPER_REGISTRY` so the agent can discover it by ID.

Importing this package eagerly imports every concrete scraper module so their
``@register_scraper`` decorators run.
"""

# Side-effect imports: each module self-registers via @register_scraper.
from . import mock as _mock  # noqa: F401
from . import serpapi as _serpapi  # noqa: F401
from .base import JobScraper, ScrapeQuery
from .registry import SCRAPER_REGISTRY, get_scraper, register_scraper

__all__ = [
    "SCRAPER_REGISTRY",
    "JobScraper",
    "ScrapeQuery",
    "get_scraper",
    "register_scraper",
]
