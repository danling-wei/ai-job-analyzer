"""Skill / responsibility extraction.

Two implementations:

* :class:`MockExtractor` — deterministic, offline, used when no LLM provider
  is configured. Pure keyword frequency over a small allow-list.
* :class:`LLMExtractor` — calls a chat model via ``langchain-openai`` with
  ``with_structured_output(_LLMOutput)`` so the model returns a typed JSON
  payload that we then merge into :class:`AnalysisResult`.

``build_extractor`` picks the right one based on settings, and silently
degrades to ``MockExtractor`` if the configured provider is unusable
(missing API key, missing optional dependency, etc.).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, Protocol, cast

import httpx
from loguru import logger
from pydantic import BaseModel, Field, SecretStr

from ..core.config import Settings, get_settings
from ..models import AnalysisRequest, AnalysisResult, JobPosting, SkillInsight


class Extractor(Protocol):
    """Anything that can turn a list of postings into an :class:`AnalysisResult`."""

    async def extract(self, postings: list[JobPosting], job_title: str) -> AnalysisResult: ...


# ---------------------------------------------------------------------------
# Mock extractor — token-frequency heuristic so the walking skeleton works
# ---------------------------------------------------------------------------

# Very small allow-list of well-known skill tokens. Used only as a fallback
# when no LLM provider is configured.
_KNOWN_SKILLS: dict[str, str] = {
    "python": "language",
    "java": "language",
    "go": "language",
    "typescript": "language",
    "javascript": "language",
    "sql": "language",
    "fastapi": "framework",
    "django": "framework",
    "react": "framework",
    "next.js": "framework",
    "postgresql": "tool",
    "mysql": "tool",
    "mongodb": "tool",
    "redis": "tool",
    "kubernetes": "platform",
    "docker": "tool",
    "aws": "platform",
    "gcp": "platform",
    "azure": "platform",
    "terraform": "tool",
    "llm": "domain",
    "llms": "domain",
    "rag": "domain",
    "pytorch": "framework",
    "tensorflow": "framework",
    "scikit-learn": "framework",
    "spark": "framework",
    "airflow": "tool",
    "dbt": "tool",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+#-]{1,}")
_CONCRETE_SKILL_CATEGORIES = frozenset({"language", "framework", "tool", "platform", "domain"})
_GENERIC_SKILL_TERMS = frozenset(
    {
        "ai/ml development",
        "ai",
        "api integration",
        "apis",
        "automation",
        "cloud technologies",
        "cloud platforms",
        "collaboration",
        "communication",
        "communication skills",
        "customer engagement",
        "data analysis",
        "generative ai",
        "machine learning",
        "problem solving",
        "project management",
        "software development",
        "technical leadership",
    }
)
_GENERIC_PHRASES = (
    "ability to",
    "best practices",
    "cross-functional",
    "development",
    "engagement",
    "experience with",
    "knowledge of",
    "leadership",
    "management",
    "problem solving",
    "skills",
)
_ORG_OR_CONTEXT_PATTERNS = (
    r"\bpublic schools?\b",
    r"\buniversity\b",
    r"\bcollege\b",
    r"\binc\.?\b",
    r"\bllc\b",
    r"\bcorp\.?\b",
    r"\bclinical care\b",
)
_NICE_TO_HAVE_NOISE = (
    "no explicit",
    "not mentioned",
    "none mentioned",
    "no nice",
)
_NICE_TO_HAVE_DUTY_PREFIXES = (
    "assist ",
    "accurately ",
    "partner ",
    "partnership ",
    "maintain ",
    "manage ",
    "track ",
    "document ",
)


class MockExtractor:
    """Heuristic extractor used when no LLM provider is configured."""

    async def extract(self, postings: list[JobPosting], job_title: str) -> AnalysisResult:
        counter: Counter[str] = Counter()
        evidence: dict[str, list[str]] = {}
        responsibilities: list[str] = []

        for posting in postings:
            text = posting.description.lower()
            for token in _TOKEN_RE.findall(text):
                if token in _KNOWN_SKILLS:
                    counter[token] += 1
                    evidence.setdefault(token, []).append(posting.title)
            for sentence in re.split(r"(?<=[.!?])\s+", posting.description):
                if any(k in sentence.lower() for k in ("responsib", "you will", "drive ")):
                    responsibilities.append(sentence.strip())

        skills = [
            SkillInsight(
                name=name,
                category=_KNOWN_SKILLS[name],  # type: ignore[arg-type]
                frequency=freq,
                importance=min(1.0, freq / max(1, len(postings))),
                evidence=evidence.get(name, [])[:3],
            )
            for name, freq in counter.most_common(15)
        ]

        return AnalysisResult(
            request=AnalysisRequest(
                job_title=job_title,
                location=None,
                sources=None,
                max_per_source=None,
                language="auto",
            ),
            postings_analysed=len(postings),
            top_skills=skills,
            core_responsibilities=responsibilities[:8],
            nice_to_have=[],
            summary=(
                f"(mock) Analysed {len(postings)} posting(s) for '{job_title}'. "
                "Configure an LLM provider in .env to get a real narrative summary."
            ),
            postings=postings,
        )


# ---------------------------------------------------------------------------
# LLM extractor — langchain-openai with structured output
# ---------------------------------------------------------------------------


class _LLMSkill(BaseModel):
    """Schema the LLM is asked to emit for each skill."""

    name: str = Field(..., description="Canonical skill name, e.g. 'Python', 'Kubernetes'.")
    category: Literal[
        "language", "framework", "tool", "platform", "domain", "soft_skill", "other"
    ] = "other"
    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Importance for the role overall in [0, 1].",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Up to 3 short verbatim snippets from the postings supporting this skill.",
    )


class _LLMOutput(BaseModel):
    """Top-level structured output we ask the LLM to produce."""

    top_skills: list[_LLMSkill] = Field(
        default_factory=list,
        description="Most important skills, ordered by importance, max 15.",
    )
    core_responsibilities: list[str] = Field(
        default_factory=list,
        description="Distinct, deduplicated core responsibilities (max 10).",
    )
    nice_to_have: list[str] = Field(
        default_factory=list,
        description="Things that are explicitly bonus / nice-to-have (max 10).",
    )
    summary: str = Field(
        ...,
        description=(
            "One short paragraph (3-5 sentences) describing what this role is "
            "about, who it's for, and the key technologies."
        ),
    )


class _SynthesisOutput(BaseModel):
    """Final cross-posting narrative output."""

    summary: str
    core_responsibilities: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)


class _SkillCanonicalizationItem(BaseModel):
    """A model-proposed skill canonicalization mapping."""

    original: str
    canonical: str | None = None
    keep: bool = True
    category: Literal[
        "language", "framework", "tool", "platform", "domain", "soft_skill", "other"
    ] = "other"


class _SkillCanonicalizationOutput(BaseModel):
    """Canonical names for extracted skills."""

    skills: list[_SkillCanonicalizationItem] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You are an expert recruiter and career coach across technical, product, "
    "operations, instructional design, and business roles. "
    "You will receive several job postings for the same target role. "
    "Carefully read all of them and produce a single consolidated analysis. "
    "Rules:\n"
    "- 'top_skills' is a job-seeker skill-gap checklist: every item should be "
    "something a candidate can deliberately learn, practice, or add to a "
    "portfolio before applying.\n"
    "- Prefer concrete, role-specific items: tools, platforms, standards, "
    "methods, workflows, frameworks, domain techniques, programming languages, "
    "libraries, databases, analytics/product tools, and named technical methods "
    "when they are relevant to the target role.\n"
    "- Do not force software-engineering skills into non-engineering roles. "
    "Extract role-specific tools, standards, methods, and workflows from the "
    "posting text itself rather than from a fixed list.\n"
    "- Do not include broad capabilities or work-style terms such as "
    "'AI', 'Machine Learning', 'Automation', 'Software Development', "
    "'AI/ML Development', 'Cloud Platforms', 'APIs', 'User Experience', "
    "'Collaboration', 'Communication Skills', 'Project Management', "
    "'Data Analysis', 'Customer Engagement', or 'Technical Leadership'. "
    "Put those in summary or responsibilities instead.\n"
    "- Prefer exact names and phrases as they appear in the postings over broad "
    "umbrella labels. For example, keep the named tool, platform, standard, "
    "method, or framework that the employer mentions.\n"
    "- Merge synonyms (e.g. 'JS' and 'JavaScript' -> 'JavaScript').\n"
    "- 'top_skills' should include the most frequently supported concrete "
    "skill-gap items, max 15.\n"
    "- 'evidence' for each skill must be verbatim snippets from the postings, "
    "max 3, max ~120 characters each. Do not invent skills or evidence; if the "
    "posting does not support a skill directly, omit it.\n"
    "- 'importance' is a float in [0, 1] reflecting how essential the skill is.\n"
    "- 'core_responsibilities' are distinct duties (not skills); deduplicate.\n"
    "- 'nice_to_have' lists explicit bonus/nice-to-have items only.\n"
    "- 'summary' is one short paragraph (3-5 sentences) for a job seeker.\n"
    "Respond strictly in the requested structured format."
)


def _normalise_skill_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def _canonical_skill_key(name: str) -> str:
    """Return a conservative merge key for obvious skill-name variants."""
    key = _normalise_skill_name(name)
    key = key.replace("&", " and ")
    key = re.sub(r"[/_]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()

    # "Quality Matters (QM)" and "Quality Matters" should merge; "QM" alone
    # still stays separate unless the model emits the full phrase somewhere.
    key = re.sub(r"\s*\(([a-z0-9.+#-]{1,8})\)\s*$", "", key).strip()

    versionless_aliases = {
        "articulate 360": "articulate",
        "articulate storyline": "articulate",
        "storyline 360": "articulate",
        "rise 360": "articulate rise",
        "react.js": "react",
        "reactjs": "react",
        "node.js": "node",
        "nodejs": "node",
        "scikit learn": "scikit-learn",
        "scikit learn.": "scikit-learn",
    }
    return versionless_aliases.get(key, key)


def _is_role_relevant_skill(skill: _LLMSkill) -> bool:
    """Return True for concrete role-specific skill-gap items."""
    name = _normalise_skill_name(skill.name)
    if not name or name in _GENERIC_SKILL_TERMS:
        return False
    if any(re.search(pattern, name) for pattern in _ORG_OR_CONTEXT_PATTERNS):
        return False
    if name in {"api", "apis"}:
        return False
    if any(phrase in name for phrase in _GENERIC_PHRASES):
        return False
    if skill.category in _CONCRETE_SKILL_CATEGORIES:
        return True
    # Smaller/local models sometimes use "other" for concrete named items.
    # Keep name-like candidates, but evidence validation still decides whether
    # they survive into the final result.
    return bool(
        re.search(r"\b[A-Z]{2,}\b", skill.name)
        or re.search(r"\b\w+(\.js|db|sql|mlops|api|ai)\b", name)
        or re.search(r"\d", skill.name)
        or name in {"c++", "c#", "node.js"}
    )


def _clean_nice_to_have(items: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        value = re.sub(r"\s+", " ", item.strip())
        lower = value.lower()
        if not value:
            continue
        if any(noise in lower for noise in _NICE_TO_HAVE_NOISE):
            continue
        if any(lower.startswith(prefix) for prefix in _NICE_TO_HAVE_DUTY_PREFIXES):
            continue
        _unique_extend(cleaned, [value], limit=10)
    return cleaned


def _snippet_supported_by_postings(snippet: str, postings_text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", snippet.strip()).lower()
    if not cleaned:
        return False
    if cleaned in postings_text:
        return True
    words = [word for word in re.findall(r"[a-z0-9.+#-]+", cleaned) if len(word) > 2]
    return bool(words) and all(word in postings_text for word in words[:6])


def _skill_supported_by_postings(skill: _LLMSkill, postings_text: str) -> bool:
    name = _normalise_skill_name(skill.name)
    if name and _count_skill_mentions(postings_text, skill.name) > 0:
        return True
    return any(_snippet_supported_by_postings(snippet, postings_text) for snippet in skill.evidence)


def _count_skill_mentions(text: str, skill_name: str) -> int:
    cleaned = skill_name.strip()
    escaped = re.escape(cleaned)
    if not escaped:
        return 0
    if cleaned[0].isalnum() and cleaned[-1].isalnum():
        pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    else:
        pattern = escaped
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _format_postings(postings: list[JobPosting], *, max_chars: int = 2500) -> str:
    chunks = []
    for i, p in enumerate(postings, 1):
        body = p.description.strip()
        if len(body) > max_chars:
            body = body[:max_chars] + "..."
        head = f"[{i}] {p.title} @ {p.company or 'Unknown'} ({p.location or 'n/a'}) — {p.url}"
        chunks.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(chunks)


def _messages_for_postings(postings: list[JobPosting], job_title: str) -> list[tuple[str, str]]:
    user_prompt = (
        f"Target role: {job_title}\n\n"
        f"Number of postings: {len(postings)}\n\n"
        f"Postings:\n\n{_format_postings(postings)}"
    )
    return [
        ("system", _SYSTEM_PROMPT),
        ("human", user_prompt),
    ]


def _build_result_from_output(
    output: _LLMOutput,
    postings: list[JobPosting],
    job_title: str,
) -> AnalysisResult:
    postings_text = "\n".join(posting.description for posting in postings).lower()
    stack_skills = [
        skill
        for skill in output.top_skills
        if _is_role_relevant_skill(skill) and _skill_supported_by_postings(skill, postings_text)
    ]

    skill_freq: Counter[str] = Counter()
    for p in postings:
        for skill in stack_skills:
            skill_freq[skill.name] += _count_skill_mentions(p.description, skill.name)

    top_skills = [
        SkillInsight(
            name=s.name,
            category=s.category,
            frequency=max(1, skill_freq.get(s.name, 1)),
            importance=s.importance,
            evidence=s.evidence[:3],
        )
        for s in stack_skills[:15]
    ]
    top_skills.sort(key=lambda s: (-s.frequency, -s.importance, s.name.lower()))

    return AnalysisResult(
        request=AnalysisRequest(
            job_title=job_title,
            location=None,
            sources=None,
            max_per_source=None,
            language="auto",
        ),
        postings_analysed=len(postings),
        top_skills=top_skills,
        core_responsibilities=output.core_responsibilities[:10],
        nice_to_have=_clean_nice_to_have(output.nice_to_have),
        summary=output.summary,
        postings=postings,
    )


def _unique_extend(target: list[str], values: Sequence[str], *, limit: int) -> None:
    seen = {value.strip().lower() for value in target}
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        target.append(cleaned)
        seen.add(cleaned.lower())
        if len(target) >= limit:
            return


def _merge_analysis_results(
    results: Sequence[AnalysisResult],
    postings: list[JobPosting],
    job_title: str,
) -> AnalysisResult:
    skills: dict[str, SkillInsight] = {}
    for result in results:
        result_skill_names: set[str] = set()
        for skill in result.top_skills:
            key = _canonical_skill_key(skill.name)
            if not key or key in result_skill_names:
                continue
            result_skill_names.add(key)
            existing = skills.get(key)
            if existing is None:
                skills[key] = SkillInsight(
                    name=skill.name,
                    category=skill.category,
                    frequency=1,
                    importance=skill.importance,
                    evidence=skill.evidence[:3],
                )
                continue
            if len(skill.name) > len(existing.name):
                existing.name = skill.name
                existing.category = skill.category
            existing.frequency += 1
            existing.importance = max(existing.importance, skill.importance)
            _unique_extend(existing.evidence, skill.evidence, limit=3)

    top_skills = sorted(
        skills.values(),
        key=lambda skill: (-skill.frequency, -skill.importance, skill.name.lower()),
    )[:15]

    responsibilities: list[str] = []
    nice_to_have: list[str] = []
    summaries: list[str] = []
    for result in results:
        _unique_extend(responsibilities, result.core_responsibilities, limit=10)
        _unique_extend(nice_to_have, _clean_nice_to_have(result.nice_to_have), limit=10)
        _unique_extend(summaries, [result.summary], limit=3)

    return AnalysisResult(
        request=AnalysisRequest(
            job_title=job_title,
            location=None,
            sources=None,
            max_per_source=None,
            language="auto",
        ),
        postings_analysed=len(postings),
        top_skills=top_skills,
        core_responsibilities=responsibilities,
        nice_to_have=nice_to_have,
        summary=" ".join(summaries),
        postings=postings,
    )


def _apply_skill_canonicalization(
    result: AnalysisResult,
    mappings: Sequence[_SkillCanonicalizationItem],
) -> AnalysisResult:
    by_original = {
        _normalise_skill_name(mapping.original): mapping
        for mapping in mappings
        if mapping.original.strip()
    }
    merged: dict[str, SkillInsight] = {}
    for skill in result.top_skills:
        mapping = by_original.get(_normalise_skill_name(skill.name))
        if mapping is not None and not mapping.keep:
            continue
        canonical_name = (
            mapping.canonical.strip()
            if mapping is not None and mapping.canonical and mapping.canonical.strip()
            else skill.name
        )
        canonical_key = _canonical_skill_key(canonical_name)
        if not canonical_key:
            continue
        category = mapping.category if mapping is not None else skill.category
        existing = merged.get(canonical_key)
        if existing is None:
            merged[canonical_key] = SkillInsight(
                name=canonical_name,
                category=category,
                frequency=skill.frequency,
                importance=skill.importance,
                evidence=skill.evidence[:3],
            )
            continue
        if len(canonical_name) > len(existing.name):
            existing.name = canonical_name
            existing.category = category
        existing.frequency += skill.frequency
        existing.importance = max(existing.importance, skill.importance)
        _unique_extend(existing.evidence, skill.evidence, limit=3)

    result.top_skills = sorted(
        merged.values(),
        key=lambda skill: (-skill.frequency, -skill.importance, skill.name.lower()),
    )[:15]
    result.nice_to_have = _clean_nice_to_have(result.nice_to_have)
    return result


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return cleaned[start : end + 1]


def _synthesis_output_from_text(text: str) -> _SynthesisOutput:
    """Best-effort fallback when a small model writes Markdown instead of JSON."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    cleaned = re.sub(r"(?im)^#{1,6}\s*(consolidated job-market analysis|summary)\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^#{1,6}\s*", "", cleaned).strip()
    responsibilities: list[str] = []
    nice_to_have: list[str] = []

    section_match = re.search(
        r"(?is)(?:core responsibilities|responsibilities)\s*:?\s*(.*?)(?:nice[- ]to[- ]have|bonus|$)",
        cleaned,
    )
    if section_match:
        responsibilities = [
            re.sub(r"^[-*\d.\s]+", "", line).strip()
            for line in section_match.group(1).splitlines()
            if re.sub(r"^[-*\d.\s]+", "", line).strip()
        ][:10]

    nice_match = re.search(r"(?is)(?:nice[- ]to[- ]have|bonus)\s*:?\s*(.*)$", cleaned)
    if nice_match:
        nice_to_have = _clean_nice_to_have(
            [
                re.sub(r"^[-*\d.\s]+", "", line).strip()
                for line in nice_match.group(1).splitlines()
            ]
        )

    summary = re.split(
        r"(?im)^\s*(?:core responsibilities|responsibilities|nice[- ]to[- ]have|bonus)\s*:?\s*$",
        cleaned,
    )[0]
    summary = re.sub(r"\s+", " ", summary).strip()
    if not summary:
        summary = re.sub(r"\s+", " ", cleaned).strip()
    if not summary:
        raise ValueError("Model response did not contain usable synthesis text.")
    return _SynthesisOutput(
        summary=summary,
        core_responsibilities=responsibilities,
        nice_to_have=nice_to_have,
    )


_CATEGORY_ALIASES = {
    "programming language": "language",
    "programming languages": "language",
    "programming_language": "language",
    "programming_languages": "language",
    "language programming": "language",
    "languages": "language",
    "library": "framework",
    "libraries": "framework",
    "frameworks": "framework",
    "package": "framework",
    "packages": "framework",
    "technology": "tool",
    "technologies": "tool",
    "tools": "tool",
    "platforms": "platform",
    "cloud": "platform",
    "cloud platform": "platform",
    "cloud platforms": "platform",
    "technical domain": "domain",
    "technique": "domain",
    "techniques": "domain",
    "method": "domain",
    "methods": "domain",
    "methodology": "domain",
    "methodologies": "domain",
    "concept": "domain",
    "concepts": "domain",
    "process": "domain",
    "processes": "domain",
    "practice": "domain",
    "practices": "domain",
    "skill": "other",
    "skills": "other",
    "other skills": "other",
    "soft skill": "soft_skill",
    "soft skills": "soft_skill",
}
_VALID_SKILL_CATEGORIES = {
    "language",
    "framework",
    "tool",
    "platform",
    "domain",
    "soft_skill",
    "other",
}


def _normalise_llm_payload(raw_json: str) -> str:
    """Coerce small-model JSON shape drift before strict Pydantic validation."""
    payload = json.loads(raw_json)
    for skill in payload.get("top_skills", []):
        if isinstance(skill.get("category"), str):
            category = re.sub(r"[\s-]+", "_", skill["category"].strip().lower())
            category = _CATEGORY_ALIASES.get(category, _CATEGORY_ALIASES.get(category.replace("_", " "), category))
            if category not in _VALID_SKILL_CATEGORIES:
                category = "other"
            skill["category"] = category
        if isinstance(skill.get("evidence"), str):
            skill["evidence"] = [skill["evidence"]]
    return json.dumps(payload)


class LLMExtractor:
    """Calls a chat model with a structured-output schema."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._chain = self._build_chain()

    def _build_chain(self) -> object:
        provider = self.settings.llm_provider
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                api_key=(
                    SecretStr(self.settings.openai_api_key)
                    if self.settings.openai_api_key
                    else None
                ),
                base_url=self.settings.openai_base_url,
            )
        elif provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "langchain-anthropic is not installed. Install with "
                    "`uv pip install ai-job-analyzer[anthropic]`."
                ) from exc
            llm = ChatAnthropic(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                api_key=self.settings.anthropic_api_key,
            )
        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError(
                    "langchain-ollama is not installed. Install with "
                    "`uv pip install ai-job-analyzer[ollama]`."
                ) from exc
            llm = ChatOllama(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                base_url=self.settings.ollama_base_url,
            )
        else:  # pragma: no cover - guarded by build_extractor
            raise ValueError(f"Unsupported llm_provider: {provider!r}")

        return llm.with_structured_output(_LLMOutput)

    async def extract(self, postings: list[JobPosting], job_title: str) -> AnalysisResult:
        if not postings:
            return await MockExtractor().extract(postings, job_title)

        messages = _messages_for_postings(postings, job_title)

        try:
            output: _LLMOutput = await self._chain.ainvoke(messages)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.exception("LLM extraction failed, falling back to MockExtractor: {}", exc)
            return await MockExtractor().extract(postings, job_title)

        return _build_result_from_output(output, postings, job_title)


class QwenLoRAExtractor:
    """Runs a local Qwen instruct model, optionally with a PEFT LoRA adapter."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._tokenizer, self._model = self._load_model()

    def _torch_dtype(self) -> object:
        import torch

        if self.settings.qwen_torch_dtype == "float16":
            return torch.float16
        if self.settings.qwen_torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.settings.qwen_torch_dtype == "float32":
            return torch.float32
        return "auto"

    def _load_model(self) -> tuple[object, object]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen LoRA extraction requires optional dependencies. "
                "Install them with `uv sync --extra qwen-lora`."
            ) from exc

        tokenizer: Any = AutoTokenizer.from_pretrained(self.settings.qwen_base_model)
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.settings.qwen_base_model,
            device_map=self.settings.qwen_device_map,
            torch_dtype=self._torch_dtype(),
        )
        if self.settings.qwen_adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "QWEN_ADAPTER_PATH is set, but `peft` is not installed. "
                    "Install with `uv sync --extra qwen-lora`."
                ) from exc
            model = PeftModel.from_pretrained(model, self.settings.qwen_adapter_path)
        model.eval()
        return tokenizer, model

    def _render_prompt(self, messages: Sequence[tuple[str, str]]) -> str:
        tokenizer = self._tokenizer
        chat_messages = [
            {"role": "system" if role == "system" else "user", "content": content}
            for role, content in messages
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return str(
                tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return "\n\n".join(f"{role.upper()}:\n{content}" for role, content in messages)

    def _invoke(self, messages: Sequence[tuple[str, str]]) -> str:
        import torch

        tokenizer: Any = self._tokenizer
        model: Any = self._model
        prompt = self._render_prompt(messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.settings.qwen_max_new_tokens,
                do_sample=self.settings.llm_temperature > 0,
                temperature=max(self.settings.llm_temperature, 0.01),
            )
        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_len:]
        return str(tokenizer.decode(generated_ids, skip_special_tokens=True))

    async def extract(self, postings: list[JobPosting], job_title: str) -> AnalysisResult:
        if not postings:
            return await MockExtractor().extract(postings, job_title)

        messages = _messages_for_postings(postings, job_title)
        messages.append(
            (
                "human",
                "Return only one valid JSON object. Do not wrap it in markdown. "
                "Do not return a JSON schema. Use exactly these top-level keys: "
                "top_skills, core_responsibilities, nice_to_have, summary. "
                "Each top_skills item must have: name, category, importance, evidence. "
                "Only include skills directly supported by the posting text. "
                "Do not add software-engineering skills to non-software roles "
                "unless those exact skills appear in the posting. "
                "For every role, extract the concrete tools, methods, standards, "
                "platforms, workflows, and domain techniques supported by the posting. "
                'Example: {"top_skills":[{"name":"RAG","category":"domain",'
                '"importance":0.9,"evidence":["Build RAG systems"]}],'
                '"core_responsibilities":["Build applied AI features."],'
                '"nice_to_have":["LoRA fine-tuning"],'
                '"summary":"This role focuses on applied AI systems."}',
            )
        )
        try:
            response_text = await asyncio.to_thread(self._invoke, messages)
            raw_json = _extract_json_object(response_text)
            output = _LLMOutput.model_validate_json(_normalise_llm_payload(raw_json))
        except Exception as exc:
            logger.exception("Qwen LoRA extraction failed: {}", exc)
            raise

        return _build_result_from_output(output, postings, job_title)

    async def synthesise(
        self,
        postings: list[JobPosting],
        job_title: str,
        top_skills: list[SkillInsight],
        nice_to_have: list[str],
    ) -> _SynthesisOutput:
        skills_text = "\n".join(
            f"- {skill.name} ({skill.category}, frequency={skill.frequency}, importance={skill.importance:.2f})"
            for skill in top_skills[:15]
        )
        posting_text = _format_postings(postings, max_chars=900)
        prompt = (
            "Return only one valid JSON object. Do not write Markdown headings, "
            "bullets outside JSON, or explanatory text.\n\n"
            f"Target role: {job_title}\n\n"
            f"Aggregated top skills:\n{skills_text or '- none'}\n\n"
            f"Nice-to-have candidates:\n"
            + "\n".join(f"- {item}" for item in nice_to_have[:20])
            + "\n\n"
            f"Source postings:\n{posting_text}\n\n"
            "Write a consolidated job-market analysis for a job seeker. "
            "Do not write one sentence per posting. Merge repeated ideas. "
            "Use only information supported by the postings or aggregated skills. "
            "Return only one valid JSON object with exactly these keys: "
            "summary, core_responsibilities, nice_to_have. "
            "summary must be one cohesive paragraph of 3-5 sentences. "
            "core_responsibilities must contain 6-10 deduplicated responsibilities. "
            "nice_to_have must contain explicit bonus preferences only. "
            'Example shape: {"summary":"...","core_responsibilities":["..."],'
            '"nice_to_have":["..."]}'
        )
        try:
            response_text = await asyncio.to_thread(
                self._invoke,
                [
                    ("system", _SYSTEM_PROMPT),
                    ("human", prompt),
                ],
            )
            try:
                raw_json = _extract_json_object(response_text)
                return _SynthesisOutput.model_validate_json(raw_json)
            except ValueError:
                logger.warning("Qwen synthesis returned non-JSON; using text fallback.")
                return _synthesis_output_from_text(response_text)
        except Exception as exc:
            logger.exception("Qwen synthesis failed: {}", exc)
            raise

    async def canonicalise_skills(
        self,
        job_title: str,
        skills: list[SkillInsight],
    ) -> _SkillCanonicalizationOutput:
        skills_text = "\n".join(
            (
                f"- {skill.name} | category={skill.category} | frequency={skill.frequency} | "
                f"evidence={'; '.join(skill.evidence[:2])}"
            )
            for skill in skills[:40]
        )
        prompt = (
            f"Target role: {job_title}\n\n"
            f"Extracted skill candidates:\n{skills_text or '- none'}\n\n"
            "Clean and canonicalize the skill list. Merge synonyms and version/name variants. "
            "Remove company names, employer names, industries, departments, duties, and broad "
            "context labels that are not job-seeker skill-gap items. Keep emerging or unfamiliar "
            "tools if the evidence supports them; do not rely on a fixed taxonomy. "
            "Return only one valid JSON object with key 'skills'. Each item must contain: "
            "original, canonical, keep, category. Use canonical=null when keep=false."
        )
        try:
            response_text = await asyncio.to_thread(
                self._invoke,
                [
                    ("system", _SYSTEM_PROMPT),
                    ("human", prompt),
                ],
            )
            raw_json = _extract_json_object(response_text)
            return _SkillCanonicalizationOutput.model_validate_json(
                _normalise_llm_payload(raw_json)
            )
        except Exception as exc:
            logger.exception("Qwen skill canonicalization failed: {}", exc)
            raise


class OpenAIFinalizer:
    """Uses OpenAI only for low-frequency final cleanup/synthesis tasks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("FINALIZER_PROVIDER=openai requires OPENAI_API_KEY.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - required dependency
            raise RuntimeError("langchain-openai is required for OpenAI finalization.") from exc

        llm = ChatOpenAI(
            model=self.settings.finalizer_model,
            temperature=self.settings.finalizer_temperature,
            api_key=SecretStr(self.settings.openai_api_key),
            base_url=self.settings.openai_base_url,
        )
        self._canonicalizer = llm.with_structured_output(_SkillCanonicalizationOutput)
        self._synthesizer = llm.with_structured_output(_SynthesisOutput)

    async def canonicalise_skills(
        self,
        job_title: str,
        skills: list[SkillInsight],
    ) -> _SkillCanonicalizationOutput:
        skills_text = "\n".join(
            (
                f"- {skill.name} | category={skill.category} | frequency={skill.frequency} | "
                f"importance={skill.importance:.2f} | evidence={'; '.join(skill.evidence[:2])}"
            )
            for skill in skills[:60]
        )
        messages = [
            (
                "system",
                "You clean extracted job-skill candidates. Merge synonyms and naming variants. "
                "Remove company names, employer names, industries, departments, duties, and "
                "broad context labels that are not candidate skill-gap items. Keep unfamiliar "
                "or emerging tools when evidence supports them. Return structured output only.",
            ),
            (
                "human",
                f"Target role: {job_title}\n\nSkill candidates:\n{skills_text or '- none'}",
            ),
        ]
        return cast(_SkillCanonicalizationOutput, await self._canonicalizer.ainvoke(messages))

    async def synthesise(
        self,
        postings: list[JobPosting],
        job_title: str,
        top_skills: list[SkillInsight],
        nice_to_have: list[str],
    ) -> _SynthesisOutput:
        skills_text = "\n".join(
            f"- {skill.name} ({skill.category}, frequency={skill.frequency}, importance={skill.importance:.2f})"
            for skill in top_skills[:15]
        )
        posting_text = _format_postings(postings, max_chars=700)
        nice_text = "\n".join(f"- {item}" for item in nice_to_have[:20])
        messages = [
            (
                "system",
                "You write concise final job-market analysis for job seekers. "
                "Merge repeated ideas; do not write one sentence per posting. "
                "Use only information supported by the provided postings and skills. "
                "Return structured output only.",
            ),
            (
                "human",
                f"Target role: {job_title}\n\n"
                f"Aggregated top skills:\n{skills_text or '- none'}\n\n"
                f"Nice-to-have candidates:\n{nice_text or '- none'}\n\n"
                f"Source postings:\n{posting_text}\n\n"
                "Produce one cohesive 3-5 sentence summary, 6-10 deduplicated core "
                "responsibilities, and explicit nice-to-have items only.",
            ),
        ]
        return cast(_SynthesisOutput, await self._synthesizer.ainvoke(messages))


class QwenServiceExtractor:
    """Calls a separately running Qwen LoRA extraction service over HTTP."""

    def __init__(
        self,
        settings: Settings | None = None,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.qwen_service_url.rstrip("/")
        self.progress_callback = progress_callback
        self._openai_finalizer: OpenAIFinalizer | None = None

    def _get_openai_finalizer(self) -> OpenAIFinalizer:
        if self._openai_finalizer is None:
            self._openai_finalizer = OpenAIFinalizer(self.settings)
        return self._openai_finalizer

    def _posting_batches(self, postings: list[JobPosting]) -> list[list[JobPosting]]:
        batch_size = max(1, self.settings.qwen_service_batch_size)
        return [postings[i : i + batch_size] for i in range(0, len(postings), batch_size)]

    async def _extract_batch(
        self,
        client: httpx.AsyncClient,
        postings: list[JobPosting],
        job_title: str,
    ) -> AnalysisResult:
        payload = {
            "job_title": job_title,
            "postings": [posting.model_dump(mode="json") for posting in postings],
        }
        response = await client.post(f"{self.base_url}/extract", json=payload)
        response.raise_for_status()
        return AnalysisResult.model_validate(response.json())

    async def _synthesise_final(
        self,
        client: httpx.AsyncClient,
        postings: list[JobPosting],
        job_title: str,
        merged: AnalysisResult,
    ) -> AnalysisResult:
        if self.settings.finalizer_provider == "openai":
            data = await self._get_openai_finalizer().synthesise(
                postings,
                job_title,
                merged.top_skills,
                merged.nice_to_have,
            )
            merged.summary = data.summary
            merged.core_responsibilities = data.core_responsibilities[:10]
            merged.nice_to_have = _clean_nice_to_have(data.nice_to_have)
            return merged
        if self.settings.finalizer_provider == "mock":
            merged.nice_to_have = _clean_nice_to_have(merged.nice_to_have)
            return merged

        payload = {
            "job_title": job_title,
            "postings": [posting.model_dump(mode="json") for posting in postings],
            "top_skills": [skill.model_dump(mode="json") for skill in merged.top_skills],
            "nice_to_have": merged.nice_to_have,
        }
        response = await client.post(f"{self.base_url}/synthesise", json=payload)
        response.raise_for_status()
        data = response.json()
        merged.summary = str(data.get("summary") or merged.summary)
        merged.core_responsibilities = [
            str(item) for item in data.get("core_responsibilities", merged.core_responsibilities)
        ][:10]
        merged.nice_to_have = [str(item) for item in data.get("nice_to_have", merged.nice_to_have)][
            :10
        ]
        return merged

    async def _canonicalise_final_skills(
        self,
        client: httpx.AsyncClient,
        job_title: str,
        merged: AnalysisResult,
    ) -> AnalysisResult:
        if self.settings.finalizer_provider == "openai":
            output = await self._get_openai_finalizer().canonicalise_skills(
                job_title,
                merged.top_skills,
            )
            return _apply_skill_canonicalization(merged, output.skills)
        if self.settings.finalizer_provider == "mock":
            merged.nice_to_have = _clean_nice_to_have(merged.nice_to_have)
            return merged

        payload = {
            "job_title": job_title,
            "skills": [skill.model_dump(mode="json") for skill in merged.top_skills],
        }
        response = await client.post(f"{self.base_url}/canonicalise-skills", json=payload)
        response.raise_for_status()
        output = _SkillCanonicalizationOutput.model_validate(response.json())
        return _apply_skill_canonicalization(merged, output.skills)

    async def extract(self, postings: list[JobPosting], job_title: str) -> AnalysisResult:
        if not postings:
            return await MockExtractor().extract(postings, job_title)

        batches = self._posting_batches(postings)
        max_concurrency = max(1, self.settings.qwen_service_max_concurrency)
        semaphore = asyncio.Semaphore(max_concurrency)

        completed = 0

        async def _emit_progress(batch: list[JobPosting], result: AnalysisResult) -> None:
            nonlocal completed
            completed += len(batch)
            if self.progress_callback is None:
                return
            await self.progress_callback(
                {
                    "phase": "per_posting",
                    "completed": min(completed, len(postings)),
                    "total": len(postings),
                    "batch_size": len(batch),
                    "skills": len(result.top_skills),
                    "title": batch[0].title if batch else None,
                    "source": batch[0].source if batch else None,
                }
            )

        async def _bounded_extract(
            client: httpx.AsyncClient,
            batch: list[JobPosting],
        ) -> AnalysisResult:
            async with semaphore:
                result = await self._extract_batch(client, batch, job_title)
                await _emit_progress(batch, result)
                return result

        try:
            async with httpx.AsyncClient(timeout=self.settings.qwen_service_timeout_seconds) as client:
                results = await asyncio.gather(
                    *(_bounded_extract(client, batch) for batch in batches)
                )
                merged = results[0] if len(results) == 1 else _merge_analysis_results(
                    results, postings, job_title
                )
                if self.progress_callback is not None:
                    await self.progress_callback(
                        {
                            "phase": "skill_cleanup",
                            "completed": len(postings),
                            "total": len(postings),
                            "batch_size": 0,
                            "skills": len(merged.top_skills),
                            "title": None,
                            "source": None,
                        }
                    )
                merged = await self._canonicalise_final_skills(client, job_title, merged)
                if self.progress_callback is not None:
                    await self.progress_callback(
                        {
                            "phase": "synthesis",
                            "completed": len(postings),
                            "total": len(postings),
                            "batch_size": 0,
                            "skills": len(merged.top_skills),
                            "title": None,
                            "source": None,
                        }
                    )
                return await self._synthesise_final(client, postings, job_title, merged)
        except Exception as exc:
            logger.exception("Qwen service extraction failed: {}", exc)
            raise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _provider_is_usable(settings: Settings) -> bool:
    if settings.llm_provider == "openai":
        return bool(settings.openai_api_key)
    if settings.llm_provider == "anthropic":
        return bool(settings.anthropic_api_key)
    if settings.llm_provider in {"qwen_lora", "qwen_service"}:
        return True
    # ollama runs locally; we don't probe here.
    return settings.llm_provider == "ollama"


def build_extractor(settings: Settings | None = None) -> Extractor:
    """Pick an extractor implementation based on configuration.

    Falls back to :class:`MockExtractor` if the configured provider is not
    usable (missing API key, missing optional dep, etc.).
    """
    settings = settings or get_settings()
    if settings.llm_provider == "mock":
        return MockExtractor()
    if not _provider_is_usable(settings):
        logger.warning(
            "LLM provider {!r} configured but unusable (missing key?); using MockExtractor.",
            settings.llm_provider,
        )
        return MockExtractor()
    try:
        if settings.llm_provider == "qwen_lora":
            return QwenLoRAExtractor(settings)
        if settings.llm_provider == "qwen_service":
            return QwenServiceExtractor(settings)
        return LLMExtractor(settings)
    except Exception as exc:
        logger.exception("Failed to initialise LLMExtractor, using MockExtractor: {}", exc)
        return MockExtractor()
