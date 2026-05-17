"""Unit tests for the SerpApi Google Jobs scraper."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from ai_job_analyzer.scrapers.base import ScrapeQuery
from ai_job_analyzer.scrapers.serpapi import SERPAPI_SEARCH_URL, SerpApiScraper

_FAKE_PAYLOAD = {
    "jobs_results": [
        {
            "job_id": "serp-1",
            "title": "ML Engineer",
            "company_name": "Acme",
            "location": "Remote",
            "description": "Python, PyTorch, Kubernetes.",
            "share_link": "https://google.example/jobs/serp-1",
            "apply_options": [{"title": "Acme", "link": "https://acme.example/jobs/1"}],
        }
    ]
}


def _job(idx: int) -> dict[str, object]:
    return {
        "job_id": f"serp-{idx}",
        "title": f"ML Engineer #{idx}",
        "company_name": "Acme",
        "location": "Remote",
        "description": "Python, PyTorch, Kubernetes.",
        "share_link": f"https://google.example/jobs/serp-{idx}",
    }


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@respx.mock
async def test_serpapi_fetches_and_normalises(isolated_cache: Path) -> None:
    route = respx.get(SERPAPI_SEARCH_URL).mock(return_value=httpx.Response(200, json=_FAKE_PAYLOAD))

    postings = await SerpApiScraper().search(
        ScrapeQuery(job_title="ML Engineer", location="Remote", max_results=10)
    )

    request_url = route.calls[0].request.url
    assert request_url.params["engine"] == "google_jobs"
    assert request_url.params["q"] == "ML Engineer"
    assert request_url.params["api_key"] == "serp-key"
    assert request_url.params["location"] == "Remote"
    assert [p.title for p in postings] == ["ML Engineer"]
    assert str(postings[0].url) == "https://acme.example/jobs/1"


@respx.mock
async def test_serpapi_paginates_with_next_page_token(isolated_cache: Path) -> None:
    route = respx.get(SERPAPI_SEARCH_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "jobs_results": [_job(i) for i in range(1, 11)],
                    "serpapi_pagination": {"next_page_token": "page-2"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "jobs_results": [_job(i) for i in range(11, 21)],
                    "serpapi_pagination": {"next_page_token": "page-3"},
                },
            ),
            httpx.Response(200, json={"jobs_results": [_job(i) for i in range(21, 31)]}),
        ]
    )

    postings = await SerpApiScraper().search(ScrapeQuery(job_title="ML Engineer", max_results=25))

    assert len(postings) == 25
    assert route.call_count == 3
    assert "next_page_token" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["next_page_token"] == "page-2"
    assert route.calls[2].request.url.params["next_page_token"] == "page-3"
    assert postings[-1].title == "ML Engineer #25"


@respx.mock
async def test_serpapi_stops_when_no_next_page_token(isolated_cache: Path) -> None:
    route = respx.get(SERPAPI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"jobs_results": [_job(i) for i in range(1, 11)]})
    )

    postings = await SerpApiScraper().search(ScrapeQuery(job_title="ML Engineer", max_results=25))

    assert len(postings) == 10
    assert route.call_count == 1


async def test_serpapi_skips_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert await SerpApiScraper().search(ScrapeQuery(job_title="Python")) == []
    finally:
        get_settings.cache_clear()
