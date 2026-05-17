"""Agentic workflow.

The workflow is implemented as an *async generator* that yields
:class:`WorkflowEvent` objects for every stage transition. Two public
entry-points wrap that generator:

* :func:`run_analysis_events` — async iterator of events. Used by the SSE
  endpoint and by the frontend's live timeline.
* :func:`run_analysis` — convenience wrapper that drains the generator and
  returns only the final :class:`AnalysisResult`. Used by the existing
  request/response ``POST /analyze`` endpoint and by the CLI.

Pipeline stages
---------------
1. **plan** — pick which scrapers to invoke based on the request.
2. **scrape** — run scrapers concurrently, emit one event per source as it
   finishes (success or error). Postings are accumulated into the state.
3. **dedupe** — drop duplicates by ``(source, source_id)`` / URL.
4. **extract** — call the LLM extractor to produce structured insights.
5. **result** — emit the final :class:`AnalysisResult`.

Each stage will eventually become a LangGraph node so we can add per-stage
retries, fan-out, and human-in-the-loop checkpoints.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ..core.config import get_settings
from ..extractors import QwenServiceExtractor, build_extractor
from ..models import AnalysisRequest, AnalysisResult, JobPosting
from ..scrapers import SCRAPER_REGISTRY, ScrapeQuery, get_scraper

# ---------------------------------------------------------------------------
# Event type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkflowEvent:
    """A single event emitted by :func:`run_analysis_events`.

    ``type`` follows a ``"<stage>:<phase>"`` convention so consumers can group
    them easily, e.g. ``"scrape:source:done"`` -> all per-source completions.
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}


# ---------------------------------------------------------------------------
# Internal state (kept for debugging / future LangGraph migration)
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    request: AnalysisRequest
    selected_sources: list[str] = field(default_factory=list)
    postings: list[JobPosting] = field(default_factory=list)
    result: AnalysisResult | None = None


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _resolve_sources(request: AnalysisRequest) -> list[str]:
    if request.sources:
        unknown = [s for s in request.sources if s not in SCRAPER_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown scraper(s): {unknown}")
        return list(request.sources)
    return sorted(SCRAPER_REGISTRY.keys())


async def _scrape_one(source_id: str, query: ScrapeQuery) -> tuple[str, list[JobPosting]]:
    scraper = get_scraper(source_id)
    configured, reason = scraper.is_configured()
    if not configured:
        raise RuntimeError(reason or "source is not configured")
    postings = await scraper.search(query)
    return source_id, postings


def _dedupe(postings: list[JobPosting]) -> list[JobPosting]:
    seen: set[tuple[str, str]] = set()
    unique: list[JobPosting] = []
    for posting in postings:
        key = (posting.source, posting.source_id or str(posting.url))
        if key in seen:
            continue
        seen.add(key)
        unique.append(posting)
    return unique


# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------


async def run_analysis_events(request: AnalysisRequest) -> AsyncIterator[WorkflowEvent]:
    """Run the workflow and yield one :class:`WorkflowEvent` per transition.

    The final event is always ``"result"`` with the full ``AnalysisResult``
    serialised as JSON-friendly dict. ``"error"`` events may be emitted for
    individual scrapers; one fatal ``"fatal"`` event is emitted if the whole
    pipeline aborts.
    """
    settings = get_settings()
    state = AgentState(request=request)

    yield WorkflowEvent(
        "plan:start",
        {
            "job_title": request.job_title,
            "location": request.location,
            "sources_requested": list(request.sources or []),
        },
    )

    try:
        state.selected_sources = _resolve_sources(request)
    except ValueError as exc:
        yield WorkflowEvent("fatal", {"stage": "plan", "error": str(exc)})
        return

    cap = request.max_per_source or settings.scrape_max_per_source
    yield WorkflowEvent(
        "plan:done",
        {
            "sources": state.selected_sources,
            "max_per_source": cap,
            "llm_provider": settings.llm_provider,
        },
    )

    # ---- Scrape (concurrent, emit per source as they finish) ----------
    query = ScrapeQuery(
        job_title=request.job_title,
        location=request.location,
        max_results=cap,
    )
    yield WorkflowEvent("scrape:start", {"sources": state.selected_sources})
    yield WorkflowEvent("scrape:source:start", {"sources": list(state.selected_sources)})

    # _scrape_one returns (source_id, postings) so we don't need to keep a
    # task->source map (which doesn't survive asyncio.as_completed wrappers).
    async def _safe_one(sid: str) -> tuple[str, list[JobPosting] | Exception]:
        try:
            return await _scrape_one(sid, query)
        except Exception as exc:
            logger.exception("scrape[{}] failed: {}", sid, exc)
            return sid, exc

    tasks = [asyncio.create_task(_safe_one(sid)) for sid in state.selected_sources]
    for awaitable in asyncio.as_completed(tasks):
        source_id, outcome = await awaitable
        if isinstance(outcome, Exception):
            event_type = (
                "scrape:source:skipped"
                if "missing " in str(outcome).lower()
                else "scrape:source:error"
            )
            yield WorkflowEvent(
                event_type,
                {"source": source_id, "error": str(outcome)},
            )
            continue
        state.postings.extend(outcome)
        logger.info("scrape[{}] -> {} postings", source_id, len(outcome))
        yield WorkflowEvent(
            "scrape:source:done",
            {
                "source": source_id,
                "count": len(outcome),
                "postings_so_far": len(state.postings),
            },
        )

    yield WorkflowEvent("scrape:done", {"total": len(state.postings)})

    # ---- Dedupe -------------------------------------------------------
    yield WorkflowEvent("dedupe:start", {"input": len(state.postings)})
    before = len(state.postings)
    state.postings = _dedupe(state.postings)
    yield WorkflowEvent(
        "dedupe:done",
        {"before": before, "after": len(state.postings), "removed": before - len(state.postings)},
    )

    # ---- Extract ------------------------------------------------------
    extract_progress_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def _on_extract_progress(data: dict[str, Any]) -> None:
        await extract_progress_queue.put(data)

    extractor = (
        QwenServiceExtractor(settings, progress_callback=_on_extract_progress)
        if settings.llm_provider == "qwen_service"
        else build_extractor()
    )
    yield WorkflowEvent(
        "extract:start",
        {
            "extractor": type(extractor).__name__,
            "provider": settings.llm_provider,
            "model": (
                settings.qwen_base_model
                if settings.llm_provider in {"qwen_lora", "qwen_service"}
                else settings.llm_model
                if settings.llm_provider != "mock"
                else None
            ),
            "postings": len(state.postings),
        },
    )
    try:
        extract_task = asyncio.create_task(extractor.extract(state.postings, request.job_title))
        while True:
            progress_task = asyncio.create_task(extract_progress_queue.get())
            done, _pending = await asyncio.wait(
                {extract_task, progress_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if progress_task in done:
                progress = progress_task.result()
                if progress is not None:
                    yield WorkflowEvent("extract:progress", progress)
                continue
            progress_task.cancel()
            result = extract_task.result()
            break
    except Exception as exc:
        logger.exception("extract failed: {}", exc)
        yield WorkflowEvent("fatal", {"stage": "extract", "error": str(exc)})
        return
    result.request = request
    state.result = result
    yield WorkflowEvent(
        "extract:done",
        {
            "top_skills": len(result.top_skills),
            "responsibilities": len(result.core_responsibilities),
            "nice_to_have": len(result.nice_to_have),
        },
    )

    # ---- Final result -------------------------------------------------
    yield WorkflowEvent("result", result.model_dump(mode="json"))


async def run_analysis(request: AnalysisRequest) -> AnalysisResult:
    """Drain :func:`run_analysis_events` and return the final result."""
    result: AnalysisResult | None = None
    async for event in run_analysis_events(request):
        if event.type == "result":
            # Re-hydrate from the dict so we always return a real model.
            result = AnalysisResult.model_validate(event.data)
        elif event.type == "fatal":
            raise RuntimeError(
                f"Workflow aborted at stage {event.data.get('stage')}: {event.data.get('error')}"
            )
    if result is None:
        raise RuntimeError("Workflow finished without producing a result.")
    return result
