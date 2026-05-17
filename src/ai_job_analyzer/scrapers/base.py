"""Abstract base class and shared types for job board scrapers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models import JobPosting


@dataclass(slots=True, frozen=True)
class ScrapeQuery:
    """Search parameters passed to every scraper."""

    job_title: str
    location: str | None = None
    max_results: int = 20


class JobScraper(ABC):
    """Interface every scraper implementation must follow.

    Subclasses must declare a unique :attr:`source_id` (used in the registry
    and in :class:`JobPosting.source`) and implement :meth:`search`.
    """

    source_id: str = ""
    display_name: str = ""
    requires_browser: bool = False

    def is_configured(self) -> tuple[bool, str | None]:
        """Return whether this scraper can run with the current settings."""
        return True, None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Only enforce on concrete (non-abstract) subclasses.
        is_abstract = bool(getattr(cls, "__abstractmethods__", None))
        if not is_abstract and not cls.source_id:
            raise TypeError(f"{cls.__name__} must define a non-empty `source_id`.")

    @abstractmethod
    async def search(self, query: ScrapeQuery) -> list[JobPosting]:
        """Return a list of (best-effort normalised) job postings.

        Implementations should:

        * Respect ``query.max_results``.
        * Be polite (rate-limit, jitter, robots.txt where applicable).
        * Never raise on a single broken posting — log and skip.
        """
        raise NotImplementedError
