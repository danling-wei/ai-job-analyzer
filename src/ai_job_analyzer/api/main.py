"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..core.config import get_settings
from ..core.logging import setup_logging
from .routes import router

# Resolve frontend/ relative to the project root (src/ai_job_analyzer/api -> root).
_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()

    app = FastAPI(
        title="AI Job Analyzer",
        version=__version__,
        description=(
            "Agentic workflow that scrapes job postings and uses LLMs to extract "
            "the key tech stack and core competencies for a given job title."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # Serve the bundled vanilla-JS frontend at /ui (and redirect / -> /ui/).
    if _FRONTEND_DIR.exists() and (_FRONTEND_DIR / "index.html").exists():
        app.mount(
            "/ui",
            StaticFiles(directory=str(_FRONTEND_DIR), html=True),
            name="ui",
        )

        @app.get("/", include_in_schema=False)
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

    return app


app = create_app()
