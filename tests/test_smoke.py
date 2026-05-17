"""Smoke tests that exercise the walking-skeleton end-to-end (no network)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_job_analyzer.agents import run_analysis
from ai_job_analyzer.api import app
from ai_job_analyzer.models import AnalysisRequest


def test_healthz() -> None:
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sources_list_includes_mock() -> None:
    client = TestClient(app)
    response = client.get("/sources")
    assert response.status_code == 200
    ids = {s["id"] for s in response.json()}
    assert "mock" in ids
    assert "serpapi" in ids
    assert "theirstack" not in ids
    assert "adzuna" not in ids
    assert "usajobs" not in ids
    assert "remoteok" not in ids
    serpapi = next(s for s in response.json() if s["id"] == "serpapi")
    assert "configured" in serpapi
    assert "disabled_reason" in serpapi


@pytest.mark.asyncio
async def test_run_analysis_with_mock_scraper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    request = AnalysisRequest(job_title="Senior ML Engineer", sources=["mock"])
    try:
        result = await run_analysis(request)
        assert result.postings_analysed > 0
        assert result.summary
        skill_names = {s.name for s in result.top_skills}
        assert "python" in skill_names
    finally:
        get_settings.cache_clear()
