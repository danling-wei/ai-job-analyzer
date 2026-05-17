"""A deterministic mock scraper used for tests and the initial walking skeleton.

It returns a tiny, plausible-looking set of postings so that the agent
workflow and the API can be exercised end-to-end without any network calls.
"""

from __future__ import annotations

from ..models import JobPosting
from .base import JobScraper, ScrapeQuery
from .registry import register_scraper


@register_scraper
class MockScraper(JobScraper):
    source_id = "mock"
    display_name = "Mock (offline fixture)"
    requires_browser = False

    async def search(self, query: ScrapeQuery) -> list[JobPosting]:
        title = query.job_title.strip()
        location = query.location or "Remote"
        return [
            JobPosting(
                source=self.source_id,
                source_id=f"mock-{i}",
                url=f"https://example.invalid/jobs/{i}",  # type: ignore[arg-type]
                title=f"{title} #{i}",
                company=f"Mock Co. {i}",
                location=location,
                posted_at=None,
                description=(
                    f"We are hiring a {title}. Responsibilities include designing "
                    "scalable systems, collaborating with cross-functional teams, and "
                    "shipping production-quality code. Required: Python, SQL, cloud "
                    "platforms (AWS or GCP), and strong communication skills. "
                    "Nice to have: experience with LLMs, Kubernetes, and Terraform."
                ),
                raw_html_path=None,
            )
            for i in range(1, min(query.max_results, 3) + 1)
        ]
