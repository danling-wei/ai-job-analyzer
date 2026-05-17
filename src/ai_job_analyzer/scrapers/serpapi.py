"""SerpApi Google Jobs scraper."""

from __future__ import annotations

from typing import Any, cast

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.cache import cache_path, read_json, write_json
from ..core.config import get_settings
from ..models import JobPosting
from ._api_helpers import first_apply_url
from .base import JobScraper, ScrapeQuery
from .registry import register_scraper

SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
SERPAPI_PAGE_SIZE = 10


def _to_posting(raw: dict[str, Any]) -> JobPosting | None:
    try:
        url = first_apply_url(raw) or raw.get("share_link")
        if not url:
            return None
        description = raw.get("description") or ""
        return JobPosting(
            source="serpapi",
            source_id=str(raw.get("job_id") or raw.get("share_link") or url),
            url=str(url),  # type: ignore[arg-type]
            title=str(raw.get("title") or "").strip() or "Untitled",
            company=(raw.get("company_name") or None),
            location=(raw.get("location") or None),
            posted_at=None,
            description=str(description),
            raw_html_path=None,
        )
    except Exception as exc:
        logger.warning("serpapi: skipping malformed row: {}", exc)
        return None


@register_scraper
class SerpApiScraper(JobScraper):
    source_id = "serpapi"
    display_name = "SerpApi Google Jobs (API)"
    requires_browser = False

    def is_configured(self) -> tuple[bool, str | None]:
        if not get_settings().serpapi_api_key:
            return False, "missing SERPAPI_API_KEY"
        return True, None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    async def _fetch(self, params: dict[str, str]) -> dict[str, Any]:
        settings = get_settings()
        headers = {"User-Agent": settings.user_agent, "Accept": "application/json"}
        async with httpx.AsyncClient(
            timeout=settings.http_timeout_seconds, headers=headers
        ) as client:
            response = await client.get(SERPAPI_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected SerpApi payload type: {type(data).__name__}")
        return cast(dict[str, Any], data)

    async def _load_or_fetch(self, query: ScrapeQuery) -> dict[str, Any] | None:
        configured, reason = self.is_configured()
        if not configured:
            logger.warning("serpapi: {}; skipping", reason)
            return None
        api_key = get_settings().serpapi_api_key
        if api_key is None:
            logger.warning("serpapi: missing SERPAPI_API_KEY; skipping")
            return None
        cache_key = f"{query.job_title}-{query.location or 'any'}-{query.max_results}"
        path = cache_path(self.source_id, cache_key, suffix=".json")
        cached = read_json(path)
        if isinstance(cached, dict):
            logger.info("serpapi: using cached search at {}", path)
            return cached
        base_params = {
            "engine": "google_jobs",
            "q": query.job_title,
            "api_key": api_key,
        }
        if query.location:
            base_params["location"] = query.location

        pages: list[dict[str, Any]] = []
        collected_rows: list[Any] = []
        next_page_token: str | None = None
        max_pages = max(1, (query.max_results + SERPAPI_PAGE_SIZE - 1) // SERPAPI_PAGE_SIZE)

        for page in range(max_pages):
            params = dict(base_params)
            if next_page_token:
                params["next_page_token"] = next_page_token
            data = await self._fetch(params)
            pages.append(data)

            rows = data.get("jobs_results")
            if isinstance(rows, list):
                collected_rows.extend(rows)
            if len(collected_rows) >= query.max_results:
                break

            pagination = data.get("serpapi_pagination")
            next_page_token = (
                str(pagination.get("next_page_token"))
                if isinstance(pagination, dict) and pagination.get("next_page_token")
                else None
            )
            if not next_page_token:
                break
            logger.info("serpapi: fetching page {} via next_page_token", page + 2)

        payload = {
            "jobs_results": collected_rows[: query.max_results],
            "pages": pages,
        }
        write_json(
            path,
            payload,
            meta={
                "url": SERPAPI_SEARCH_URL,
                "rows": len(payload["jobs_results"]),
                "pages": len(pages),
                "job_title": query.job_title,
                "location": query.location,
            },
        )
        return payload

    async def search(self, query: ScrapeQuery) -> list[JobPosting]:
        try:
            payload = await self._load_or_fetch(query)
        except Exception as exc:
            logger.warning("serpapi: search failed for {!r}: {}", query.job_title, exc)
            return []
        if payload is None:
            return []
        rows = payload.get("jobs_results")
        if not isinstance(rows, list):
            logger.warning("serpapi: payload has no jobs_results list")
            return []

        results: list[JobPosting] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            posting = _to_posting(row)
            if posting is None:
                continue
            results.append(posting)
            if len(results) >= query.max_results:
                break
        return results
