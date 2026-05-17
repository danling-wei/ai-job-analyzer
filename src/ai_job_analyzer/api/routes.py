"""HTTP routes."""

from __future__ import annotations

import json
import traceback
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .. import __version__
from ..agents import run_analysis, run_analysis_events
from ..models import AnalysisRequest, AnalysisResult
from ..scrapers import SCRAPER_REGISTRY

router = APIRouter()


@router.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/sources", tags=["meta"])
async def list_sources() -> list[dict[str, object]]:
    return [
        {
            "id": cls.source_id,
            "name": cls.display_name or cls.source_id,
            "requires_browser": cls.requires_browser,
            "configured": cls().is_configured()[0],
            "disabled_reason": cls().is_configured()[1],
        }
        for cls in SCRAPER_REGISTRY.values()
    ]


@router.post("/analyze", response_model=AnalysisResult, tags=["analysis"])
async def analyze(request: AnalysisRequest) -> AnalysisResult:
    """Run the full agentic workflow and return the final result."""
    return await run_analysis(request)


def _format_sse(event_type: str, data: dict[str, object]) -> str:
    """Encode a single Server-Sent Event frame.

    See https://html.spec.whatwg.org/multipage/server-sent-events.html.
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


@router.post("/analyze/stream", tags=["analysis"])
async def analyze_stream(request: AnalysisRequest) -> StreamingResponse:
    """Stream the workflow as Server-Sent Events.

    Each event has a named ``event:`` field (see ``WorkflowEvent.type``) and
    a JSON ``data:`` payload. The terminal event is always ``result`` (or
    ``fatal`` if the pipeline aborted).
    """

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in run_analysis_events(request):
                yield _format_sse(event.type, event.data)
        except Exception as exc:
            yield _format_sse(
                "fatal",
                {
                    "stage": "stream",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                },
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for nginx etc.
        },
    )
