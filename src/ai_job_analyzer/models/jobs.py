"""Domain models for job postings and analysis results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class AnalysisRequest(BaseModel):
    """Input from the user / API caller."""

    job_title: str = Field(
        ..., min_length=2, description="Target job title, e.g. 'Senior ML Engineer'."
    )
    location: str | None = Field(
        None, description="Optional location filter, e.g. 'Remote', 'Berlin'."
    )
    sources: list[str] | None = Field(
        None,
        description="Optional explicit list of scraper IDs to use. None = all enabled.",
    )
    max_per_source: int | None = Field(
        None, ge=1, le=200, description="Override the per-source posting cap."
    )
    language: Literal["en", "zh", "auto"] = Field(
        "auto", description="Preferred language for the LLM summary output."
    )


class JobPosting(BaseModel):
    """Normalised representation of a single job posting."""

    source: str = Field(..., description="Scraper / site identifier, e.g. 'linkedin', 'lagou'.")
    source_id: str | None = Field(None, description="Site-native ID for deduplication.")
    url: HttpUrl
    title: str
    company: str | None = None
    location: str | None = None
    posted_at: datetime | None = None
    description: str = Field(
        "", description="Raw JD / responsibilities text (cleaned but unsummarised)."
    )
    raw_html_path: str | None = Field(
        None, description="Optional path to the cached raw HTML for debugging."
    )


class SkillInsight(BaseModel):
    """A single extracted skill or competency."""

    name: str
    category: Literal[
        "language",
        "framework",
        "tool",
        "platform",
        "domain",
        "soft_skill",
        "other",
    ] = "other"
    frequency: int = Field(1, ge=1, description="How many postings mention this skill.")
    importance: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="LLM-assigned importance score in [0, 1] for the role overall.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Short verbatim snippets from postings supporting this skill.",
    )


class AnalysisResult(BaseModel):
    """End-to-end output of the agentic workflow."""

    request: AnalysisRequest
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    postings_analysed: int = 0
    top_skills: list[SkillInsight] = Field(default_factory=list)
    core_responsibilities: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    summary: str = Field("", description="One-paragraph narrative summary written by the LLM.")
    postings: list[JobPosting] = Field(
        default_factory=list,
        description="The underlying postings used to compute the analysis.",
    )
