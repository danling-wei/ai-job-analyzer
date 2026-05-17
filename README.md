# AI Job Analyzer

Agentic workflow that, given a job title, scrapes matching postings from
multiple job boards and uses an LLM to extract the **key tech stack** and
**core competencies** required for the role.

> Companion doc: see [`CLAUDE.md`](./CLAUDE.md) for the living architecture
> notes, conventions, and roadmap.

## Status

**M1.7 in progress.** Single-page frontend at `/ui/` with a **live workflow
timeline** that streams every stage of the agent (plan → scrape per source
→ dedupe → extract → result) over Server-Sent Events. The noisy public/RSS
demo sources have been replaced with **SerpApi Google Jobs** as the API-backed
source. The LLM falls back to a safe mock when no API key is configured, so the
project still runs offline for mock-source demos.

## Requirements

- Python `>=3.11,<3.14`
- [`uv`](https://docs.astral.sh/uv/) (`python -m pip install --user uv` if you
  don't have it).

## Quick start

```bash
# 1. Install deps (creates .venv automatically)
uv sync

# 2. (Optional) configure your environment
cp .env.example .env

# 3. Run the server
uv run ai-job-analyzer serve --reload
# -> open http://127.0.0.1:8000/        (frontend with live workflow timeline)
# -> open http://127.0.0.1:8000/docs    (OpenAPI / Swagger)

# 4. Or skip the UI and run a one-off analysis from the CLI
uv run ai-job-analyzer analyze "Senior ML Engineer" --source mock     # offline demo
uv run ai-job-analyzer analyze "Python Backend"     --source serpapi
```

To use a real LLM for the summary, set `LLM_PROVIDER=openai` and
`OPENAI_API_KEY=...` in `.env`. You can also run a local fine-tuned Qwen LoRA
model as a separate service with `LLM_PROVIDER=qwen_service`; see
[`model_lab/README.md`](model_lab/README.md). In hybrid mode, Qwen handles the
high-volume per-JD skill extraction while `FINALIZER_PROVIDER=openai` lets a
stronger OpenAI model canonicalize skill synonyms and write the final summary.
Without a configured provider the pipeline silently falls back to a
keyword-frequency heuristic so analysis always returns something.

API sources require their own keys in `.env`: `SERPAPI_API_KEY`. Sources with
missing credentials are skipped safely.

## API endpoints

| Method | Path                | Purpose                                            |
| ------ | ------------------- | -------------------------------------------------- |
| GET    | `/`                 | Redirects to `/ui/`.                               |
| GET    | `/ui/`              | Single-page frontend (live workflow timeline).     |
| GET    | `/healthz`          | Liveness probe.                                    |
| GET    | `/sources`          | List of registered scrapers.                       |
| POST   | `/analyze`          | Run the workflow, return final `AnalysisResult`.   |
| POST   | `/analyze/stream`   | Same workflow, **streamed as Server-Sent Events**. |

## Project layout

```
src/ai_job_analyzer/
├── api/          # FastAPI app + routes (incl. SSE) + /ui mount
├── agents/       # Agentic workflow as an async-generator of WorkflowEvents
├── scrapers/     # JobScraper interface + mock + real sources (registry-based)
├── extractors/   # LLM (langchain) + heuristic mock extractors
├── models/       # Pydantic schemas shared across layers
├── core/         # Settings + logging + on-disk cache
└── cli.py        # `ai-job-analyzer` entry point (typer)
frontend/
└── index.html    # Single-file UI (Tailwind CDN, vanilla JS, SSE consumer)
tests/            # pytest: smoke + scraper + extractor + stream coverage
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src

# optional local Qwen LoRA training/inference stack
uv sync --extra qwen-lora
uv run ai-job-analyzer serve-qwen

# optional hybrid app mode (requires OPENAI_API_KEY)
LLM_PROVIDER=qwen_service FINALIZER_PROVIDER=openai uv run ai-job-analyzer serve --reload
```

## License

[Apache 2.0](./LICENSE)
