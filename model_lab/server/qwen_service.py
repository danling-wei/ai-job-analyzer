"""Standalone FastAPI service for Qwen LoRA extraction.

Run from the repo root:

    uv run ai-job-analyzer serve-qwen

The main application can then use it with:

    LLM_PROVIDER=qwen_service
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ai_job_analyzer.core.config import get_settings
from ai_job_analyzer.extractors.llm import QwenLoRAExtractor
from ai_job_analyzer.models import AnalysisResult, JobPosting, SkillInsight


class QwenExtractionRequest(BaseModel):
    """Payload accepted by the standalone Qwen service."""

    job_title: str
    postings: list[JobPosting]


class QwenSynthesisRequest(BaseModel):
    """Payload for final cross-posting synthesis."""

    job_title: str
    postings: list[JobPosting]
    top_skills: list[SkillInsight] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)


class QwenSynthesisResponse(BaseModel):
    """Final narrative fields generated after per-posting extraction."""

    summary: str
    core_responsibilities: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)


class QwenSkillCanonicalizationRequest(BaseModel):
    """Payload for final skill cleanup/canonicalization."""

    job_title: str
    skills: list[SkillInsight] = Field(default_factory=list)


app = FastAPI(title="AI Job Analyzer Qwen Service")
_extractor: QwenLoRAExtractor | None = None

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen Service Tester</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-zinc-950 text-zinc-100">
  <main class="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-5 px-5 py-6">
    <header class="flex flex-col gap-1 border-b border-zinc-800 pb-4">
      <h1 class="text-2xl font-semibold">Qwen Service Tester</h1>
      <p class="text-sm text-zinc-400">POST /extract · standalone model service</p>
    </header>

    <section class="grid gap-4 lg:grid-cols-[1fr_1fr]">
      <form id="form" class="flex flex-col gap-4">
        <label class="flex flex-col gap-2">
          <span class="text-sm font-medium text-zinc-300">Job title</span>
          <input id="jobTitle" class="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 outline-none focus:border-sky-500" value="Applied AI Engineer" />
        </label>

        <label class="flex flex-col gap-2">
          <span class="text-sm font-medium text-zinc-300">Posting text</span>
          <textarea id="description" class="min-h-72 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 outline-none focus:border-sky-500">Build RAG systems with PyTorch, vector databases, prompt evaluation, and production inference pipelines. Experience with Docker, Kubernetes, and FastAPI is helpful.</textarea>
        </label>

        <button id="submit" class="w-fit rounded-md bg-sky-500 px-4 py-2 font-medium text-zinc-950 hover:bg-sky-400" type="submit">Run extraction</button>
      </form>

      <section class="flex flex-col gap-2">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-medium text-zinc-300">Result</h2>
          <span id="status" class="text-xs text-zinc-500">Idle</span>
        </div>
        <pre id="output" class="min-h-96 overflow-auto rounded-md border border-zinc-800 bg-black p-4 text-xs leading-relaxed text-zinc-200"></pre>
      </section>
    </section>
  </main>

  <script>
    const form = document.querySelector("#form");
    const statusEl = document.querySelector("#status");
    const output = document.querySelector("#output");
    const submit = document.querySelector("#submit");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusEl.textContent = "Running...";
      submit.disabled = true;
      output.textContent = "";

      const payload = {
        job_title: document.querySelector("#jobTitle").value,
        postings: [{
          source: "qwen-ui",
          source_id: "manual-1",
          url: "https://example.invalid/manual-1",
          title: document.querySelector("#jobTitle").value,
          company: "Manual Test",
          location: "Remote",
          description: document.querySelector("#description").value
        }]
      };

      try {
        const response = await fetch("/extract", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        output.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = response.ok ? "Done" : `HTTP ${response.status}`;
      } catch (error) {
        output.textContent = String(error);
        statusEl.textContent = "Error";
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def _get_extractor() -> QwenLoRAExtractor:
    global _extractor
    if _extractor is None:
        _extractor = QwenLoRAExtractor(get_settings())
    return _extractor


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Small manual test UI for local model checks."""
    return _INDEX_HTML


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    """Return service liveness without forcing model loading."""
    settings = get_settings()
    model_device = None
    torch_cuda_available = None
    torch_cuda_device = None
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_available = torch.cuda.is_available()
        if torch_cuda_available:
            torch_cuda_device = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    if _extractor is not None:
        model = _extractor._model
        if hasattr(model, "hf_device_map"):
            model_device = model.hf_device_map
        elif hasattr(model, "device"):
            model_device = str(model.device)
    return {
        "status": "ok",
        "model_loaded": _extractor is not None,
        "model_device": model_device,
        "torch_version": torch_version,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_device": torch_cuda_device,
        "base_model": settings.qwen_base_model,
        "adapter_path": settings.qwen_adapter_path,
    }


@app.post("/extract", response_model=AnalysisResult)
async def extract(request: QwenExtractionRequest) -> AnalysisResult:
    """Extract skills and responsibilities with the local Qwen model."""
    extractor = _get_extractor()
    return await extractor.extract(request.postings, request.job_title)


@app.post("/synthesise", response_model=QwenSynthesisResponse)
async def synthesise(request: QwenSynthesisRequest) -> QwenSynthesisResponse:
    """Generate final cross-posting summary fields after skill aggregation."""
    extractor = _get_extractor()
    response = await extractor.synthesise(
        request.postings,
        request.job_title,
        request.top_skills,
        request.nice_to_have,
    )
    return response


@app.post("/canonicalise-skills")
async def canonicalise_skills(request: QwenSkillCanonicalizationRequest) -> dict[str, object]:
    """Clean and merge extracted skill candidates."""
    extractor = _get_extractor()
    response = await extractor.canonicalise_skills(request.job_title, request.skills)
    return response.model_dump(mode="json")
