"""Tests for the SSE streaming endpoint and the underlying event generator."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import ai_job_analyzer.agents.workflow as workflow
from ai_job_analyzer.agents import WorkflowEvent, run_analysis_events
from ai_job_analyzer.api import app
from ai_job_analyzer.extractors.llm import MockExtractor
from ai_job_analyzer.models import AnalysisRequest, AnalysisResult, JobPosting


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Tiny SSE parser: returns a list of (event, data-dict) tuples."""
    events: list[tuple[str, dict]] = []
    for frame in body.split("\n\n"):
        frame = frame.strip("\n")
        if not frame:
            continue
        ev_type = "message"
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                ev_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((ev_type, json.loads("\n".join(data_lines)) if data_lines else {}))
    return events


async def test_run_analysis_events_emits_full_pipeline(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    request = AnalysisRequest(job_title="Senior ML Engineer", sources=["mock"])
    try:
        events: list[WorkflowEvent] = [e async for e in run_analysis_events(request)]
    finally:
        get_settings.cache_clear()

    types = [e.type for e in events]
    # Required ordered milestones (other events may interleave between them).
    for required in (
        "plan:start",
        "plan:done",
        "scrape:start",
        "scrape:done",
        "dedupe:start",
        "dedupe:done",
        "extract:start",
        "extract:done",
        "result",
    ):
        assert required in types, f"missing event {required!r}; got {types}"
    assert types[-1] == "result"

    # Per-source events must be emitted for every selected source.
    started = next(e for e in events if e.type == "scrape:source:start")
    assert started.data["sources"] == ["mock"]
    done = [e for e in events if e.type == "scrape:source:done"]
    assert len(done) == 1
    assert done[0].data["source"] == "mock"
    assert done[0].data["count"] >= 1


def test_analyze_stream_endpoint_returns_sse(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)
    try:
        with client.stream(
            "POST",
            "/analyze/stream",
            json={"job_title": "Backend Engineer", "sources": ["mock"]},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(chunk for chunk in response.iter_text())
    finally:
        get_settings.cache_clear()

    events = _parse_sse(body)
    types = [t for t, _ in events]
    assert "plan:start" in types
    assert types[-1] == "result"

    # The terminal event should carry the full AnalysisResult dict.
    _, result_data = events[-1]
    assert result_data["postings_analysed"] >= 1
    assert {"top_skills", "core_responsibilities", "summary"} <= set(result_data)


def test_analyze_stream_emits_fatal_on_unknown_source() -> None:
    client = TestClient(app)
    with client.stream(
        "POST",
        "/analyze/stream",
        json={"job_title": "Backend Engineer", "sources": ["does-not-exist"]},
    ) as response:
        assert response.status_code == 200
        body = "".join(chunk for chunk in response.iter_text())

    events = _parse_sse(body)
    fatal = [(t, d) for t, d in events if t == "fatal"]
    assert len(fatal) == 1
    assert fatal[0][1]["stage"] == "plan"


async def test_run_analysis_events_marks_unconfigured_source_skipped(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    try:
        request = AnalysisRequest(job_title="Backend Engineer", sources=["serpapi"])
        events: list[WorkflowEvent] = [e async for e in run_analysis_events(request)]
    finally:
        get_settings.cache_clear()

    skipped = [e for e in events if e.type == "scrape:source:skipped"]
    assert len(skipped) == 1
    assert skipped[0].data["source"] == "serpapi"
    assert "SERPAPI_API_KEY" in skipped[0].data["error"]


async def test_run_analysis_events_emits_qwen_extract_progress(
    monkeypatch,
) -> None:
    class _FakeQwenServiceExtractor:
        def __init__(self, _settings: object, progress_callback: object) -> None:
            self.progress_callback = progress_callback

        async def extract(
            self,
            postings: list[JobPosting],
            job_title: str,
        ) -> AnalysisResult:
            callback = self.progress_callback
            for index, posting in enumerate(postings, 1):
                await callback(
                    {
                        "completed": index,
                        "total": len(postings),
                        "batch_size": 1,
                        "skills": 1,
                        "title": posting.title,
                        "source": posting.source,
                    }
                )
            return await MockExtractor().extract(postings, job_title)

    monkeypatch.setenv("LLM_PROVIDER", "qwen_service")
    monkeypatch.setattr(workflow, "QwenServiceExtractor", _FakeQwenServiceExtractor)
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    try:
        request = AnalysisRequest(job_title="Backend Engineer", sources=["mock"], max_per_source=2)
        events: list[WorkflowEvent] = [event async for event in run_analysis_events(request)]
    finally:
        get_settings.cache_clear()

    progress = [event for event in events if event.type == "extract:progress"]
    assert progress
    assert progress[-1].data["completed"] == progress[-1].data["total"]
