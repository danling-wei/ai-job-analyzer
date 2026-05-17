"""Typer-based command line interface.

Usage examples
--------------
    # Run the API server (auto-reload in dev)
    uv run ai-job-analyzer serve --reload

    # Quick one-off analysis from the terminal (uses the mock scraper by default)
    uv run ai-job-analyzer analyze "Senior ML Engineer" --location Remote
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
import uvicorn
from rich import print as rprint
from rich.table import Table

from .agents import run_analysis
from .core.config import get_settings
from .core.logging import setup_logging
from .models import AnalysisRequest

app = typer.Typer(add_completion=False, help="AI Job Analyzer CLI")


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host (defaults to settings)."),
    port: int | None = typer.Option(None, help="Bind port (defaults to settings)."),
    reload: bool = typer.Option(False, help="Enable uvicorn auto-reload (dev)."),
) -> None:
    """Start the FastAPI server."""
    setup_logging()
    settings = get_settings()
    uvicorn.run(
        "ai_job_analyzer.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@app.command("serve-qwen")
def serve_qwen(
    host: str | None = typer.Option(None, help="Bind host (defaults to settings)."),
    port: int | None = typer.Option(None, help="Bind port (defaults to settings)."),
) -> None:
    """Start the standalone Qwen LoRA model service."""
    setup_logging()
    settings = get_settings()
    sys.path.insert(0, str(Path.cwd()))
    uvicorn.run(
        "model_lab.server.qwen_service:app",
        host=host or settings.qwen_service_host,
        port=port or settings.qwen_service_port,
    )


@app.command()
def analyze(
    job_title: str = typer.Argument(..., help="Target job title."),
    location: str | None = typer.Option(None, help="Optional location filter."),
    sources: list[str] | None = typer.Option(
        None, "--source", help="Limit to specific scraper IDs (repeatable)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON instead of a table."),
) -> None:
    """Run the agentic workflow once and print the result."""
    setup_logging()
    request = AnalysisRequest(
        job_title=job_title,
        location=location,
        sources=sources,
        max_per_source=None,
        language="auto",
    )
    result = asyncio.run(run_analysis(request))

    if json_output:
        rprint(json.loads(result.model_dump_json()))
        return

    rprint(f"\n[bold]Job title:[/bold] {result.request.job_title}")
    rprint(f"[bold]Postings analysed:[/bold] {result.postings_analysed}")
    rprint(f"[bold]Summary:[/bold] {result.summary}\n")

    table = Table(title="Top skills")
    table.add_column("Skill")
    table.add_column("Category")
    table.add_column("Frequency", justify="right")
    table.add_column("Importance", justify="right")
    for s in result.top_skills:
        table.add_row(s.name, s.category, str(s.frequency), f"{s.importance:.2f}")
    rprint(table)


def main() -> None:  # pragma: no cover - thin entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
