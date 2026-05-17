"""Tests for the LLM extractor wiring (no real network calls)."""

from __future__ import annotations

import httpx
import pytest
from respx import MockRouter

from ai_job_analyzer.core.config import Settings
from ai_job_analyzer.extractors.llm import (
    LLMExtractor,
    MockExtractor,
    QwenLoRAExtractor,
    QwenServiceExtractor,
    _apply_skill_canonicalization,
    _clean_nice_to_have,
    _LLMOutput,
    _LLMSkill,
    _normalise_llm_payload,
    _SkillCanonicalizationItem,
    build_extractor,
)
from ai_job_analyzer.models import AnalysisRequest, AnalysisResult, JobPosting, SkillInsight


def _posting(idx: int) -> JobPosting:
    extra = " Python." if idx == 2 else ""
    return JobPosting(
        source="mock",
        source_id=f"p-{idx}",
        url=f"https://example.invalid/{idx}",
        title=f"Backend Engineer #{idx}",
        company="Example",
        location="Remote",
        description=(
            "We are hiring a Backend Engineer. Python, FastAPI, and PostgreSQL "
            f"are required. Nice to have: Kubernetes.{extra}"
        ),
    )


def test_build_extractor_defaults_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert isinstance(build_extractor(), MockExtractor)
    finally:
        get_settings.cache_clear()


def test_build_extractor_falls_back_when_openai_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    # Override any developer-local .env value so this test stays offline.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from ai_job_analyzer.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert isinstance(build_extractor(), MockExtractor)
    finally:
        get_settings.cache_clear()


async def test_llm_extractor_uses_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMExtractor merges the structured chain output into AnalysisResult."""

    fake_output = _LLMOutput(
        top_skills=[
            _LLMSkill(name="Kubernetes", category="platform", importance=1.0, evidence=["nice"]),
            _LLMSkill(
                name="FastAPI", category="framework", importance=0.8, evidence=["FastAPI..."]
            ),
            _LLMSkill(name="Python", category="language", importance=0.4, evidence=["Python..."]),
            _LLMSkill(
                name="AI",
                category="domain",
                importance=1.0,
                evidence=["AI product"],
            ),
            _LLMSkill(
                name="Machine Learning",
                category="domain",
                importance=0.9,
                evidence=["ML"],
            ),
            _LLMSkill(
                name="Cloud Platforms",
                category="platform",
                importance=0.8,
                evidence=["cloud"],
            ),
            _LLMSkill(
                name="User Experience (UX)",
                category="domain",
                importance=0.8,
                evidence=["UX"],
            ),
            _LLMSkill(
                name="APIs",
                category="tool",
                importance=0.7,
                evidence=["APIs"],
            ),
            _LLMSkill(
                name="Collaboration",
                category="soft_skill",
                importance=0.9,
                evidence=["cross-functional"],
            ),
            _LLMSkill(
                name="Automation",
                category="domain",
                importance=0.9,
                evidence=["automate workflows"],
            ),
        ],
        core_responsibilities=["Ship production code.", "Design APIs."],
        nice_to_have=["Kubernetes"],
        summary="A backend engineering role focused on Python + FastAPI.",
    )

    class _FakeChain:
        async def ainvoke(self, _messages: object) -> _LLMOutput:
            return fake_output

    extractor = LLMExtractor.__new__(LLMExtractor)  # bypass __init__
    extractor.settings = None  # type: ignore[assignment]
    extractor._chain = _FakeChain()  # type: ignore[attr-defined]

    result = await extractor.extract([_posting(1), _posting(2)], "Backend Engineer")

    assert result.postings_analysed == 2
    assert result.summary == fake_output.summary
    assert [s.name for s in result.top_skills] == ["Python", "Kubernetes", "FastAPI"]
    filtered_names = {s.name for s in result.top_skills}
    assert "AI" not in filtered_names
    assert "Machine Learning" not in filtered_names
    assert "Cloud Platforms" not in filtered_names
    assert "User Experience (UX)" not in filtered_names
    assert "APIs" not in filtered_names
    assert "Collaboration" not in filtered_names
    assert "Automation" not in filtered_names
    # Frequency is recomputed from posting text, so 'Python' should appear in both.
    python_skill = next(s for s in result.top_skills if s.name == "Python")
    assert python_skill.frequency == 3
    assert [s.frequency for s in result.top_skills] == sorted(
        (s.frequency for s in result.top_skills),
        reverse=True,
    )
    assert "Ship production code." in result.core_responsibilities
    assert "Kubernetes" in result.nice_to_have


async def test_short_acronym_mentions_do_not_match_inside_words() -> None:
    fake_output = _LLMOutput(
        top_skills=[
            _LLMSkill(name="AI", category="domain", importance=1.0, evidence=["AI"]),
            _LLMSkill(name="RAG", category="domain", importance=0.8, evidence=["RAG"]),
        ],
        core_responsibilities=[],
        nice_to_have=[],
        summary="A role using retrieval augmented generation.",
    )

    class _FakeChain:
        async def ainvoke(self, _messages: object) -> _LLMOutput:
            return fake_output

    posting = JobPosting(
        source="mock",
        source_id="p-1",
        url="https://example.invalid/1",
        title="AI Product Manager",
        company="Example",
        location="Remote",
        description=(
            "Maintain training pipelines and availability dashboards. "
            "Build RAG evaluations for customer-facing assistants."
        ),
    )
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor.settings = None  # type: ignore[assignment]
    extractor._chain = _FakeChain()  # type: ignore[attr-defined]

    result = await extractor.extract([posting], "AI Product Manager")

    names = {s.name for s in result.top_skills}
    assert "AI" not in names
    rag = next(s for s in result.top_skills if s.name == "RAG")
    assert rag.frequency == 1


async def test_role_relevant_instructional_design_skills_are_kept() -> None:
    fake_output = _LLMOutput(
        top_skills=[
            _LLMSkill(
                name="Articulate 360",
                category="tool",
                importance=0.9,
                evidence=["Articulate 360"],
            ),
            _LLMSkill(
                name="SCORM",
                category="tool",
                importance=0.8,
                evidence=["SCORM-compliant modules"],
            ),
            _LLMSkill(
                name="WCAG 2.0 AA",
                category="tool",
                importance=0.7,
                evidence=["WCAG 2.0 AA accessibility standards"],
            ),
            _LLMSkill(
                name="Python",
                category="language",
                importance=0.9,
                evidence=["Python"],
            ),
        ],
        core_responsibilities=["Design e-learning modules."],
        nice_to_have=[],
        summary="Instructional design role.",
    )

    class _FakeChain:
        async def ainvoke(self, _messages: object) -> _LLMOutput:
            return fake_output

    posting = JobPosting(
        source="mock",
        source_id="p-1",
        url="https://example.invalid/1",
        title="Instructional Designer",
        company="Example",
        location="Remote",
        description=(
            "Design SCORM-compliant modules in Articulate 360 and ensure "
            "WCAG 2.0 AA accessibility standards."
        ),
    )
    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor.settings = None  # type: ignore[assignment]
    extractor._chain = _FakeChain()  # type: ignore[attr-defined]

    result = await extractor.extract([posting], "Instructional Designer")

    names = {skill.name for skill in result.top_skills}
    assert {"Articulate 360", "SCORM", "WCAG 2.0 AA"} <= names
    assert "Python" not in names


async def test_qwen_lora_extractor_parses_json_response() -> None:
    class _FakeQwen(QwenLoRAExtractor):
        def __init__(self) -> None:
            pass

        def _invoke(self, _messages: object) -> str:
            return """
            ```json
            {
              "top_skills": [
                {"name": "RAG", "category": "domain", "importance": 0.9, "evidence": ["Build RAG systems"]},
                {"name": "PyTorch", "category": "framework", "importance": 0.8, "evidence": ["PyTorch models"]},
                {"name": "AI", "category": "domain", "importance": 1.0, "evidence": ["AI role"]}
              ],
              "core_responsibilities": ["Build AI product features."],
              "nice_to_have": ["LoRA fine-tuning"],
              "summary": "A role focused on applied AI systems."
            }
            ```
            """

    posting = JobPosting(
        source="mock",
        source_id="p-1",
        url="https://example.invalid/1",
        title="Applied AI Engineer",
        company="Example",
        location="Remote",
        description="Build RAG systems with PyTorch models and evaluation pipelines.",
    )

    result = await _FakeQwen().extract([posting], "Applied AI Engineer")

    assert [s.name for s in result.top_skills] == ["RAG", "PyTorch"]
    assert result.summary == "A role focused on applied AI systems."


async def test_qwen_lora_extractor_normalises_small_shape_drift() -> None:
    class _FakeQwen(QwenLoRAExtractor):
        def __init__(self) -> None:
            pass

        def _invoke(self, _messages: object) -> str:
            return """
            {
              "top_skills": [
                {"name": "Python", "category": "programming languages", "importance": 0.8, "evidence": "Python"},
                {"name": "FastAPI", "category": "programming_language", "importance": 0.8, "evidence": "FastAPI"},
                {"name": "prompt evaluation", "category": "methodology", "importance": 0.7, "evidence": "prompt evaluation"}
              ],
              "core_responsibilities": ["Build APIs."],
              "nice_to_have": [],
              "summary": "A role focused on Python APIs."
            }
            """

    posting = JobPosting(
        source="mock",
        source_id="p-1",
        url="https://example.invalid/1",
        title="Backend Engineer",
        company="Example",
        location="Remote",
        description="Use Python, FastAPI, and prompt evaluation.",
    )

    result = await _FakeQwen().extract([posting], "Backend Engineer")

    python_skill = next(skill for skill in result.top_skills if skill.name == "Python")
    assert python_skill.category == "language"
    assert python_skill.evidence == ["Python"]
    categories = {skill.name: skill.category for skill in result.top_skills}
    assert categories["FastAPI"] == "language"
    assert categories["prompt evaluation"] == "domain"


def test_normalise_llm_payload_defaults_unknown_categories_to_other() -> None:
    payload = _normalise_llm_payload(
        """
        {
          "top_skills": [
            {"name": "Mystery Skill", "category": "made_up_category", "importance": 0.6, "evidence": "Mystery Skill"}
          ],
          "core_responsibilities": [],
          "nice_to_have": [],
          "summary": "x"
        }
        """
    )

    output = _LLMOutput.model_validate_json(payload)

    assert output.top_skills[0].category == "other"
    assert output.top_skills[0].evidence == ["Mystery Skill"]


def test_skill_cleanup_removes_context_items_and_noisy_nice_to_haves() -> None:
    result = AnalysisResult(
        request=AnalysisRequest(job_title="Instructional Designer"),
        postings_analysed=2,
        top_skills=[
            SkillInsight(
                name="IDEA Public Schools",
                category="other",
                frequency=1,
                importance=0.7,
                evidence=["IDEA Public Schools"],
            ),
            SkillInsight(
                name="Learning Outcomes",
                category="domain",
                frequency=1,
                importance=0.8,
                evidence=["learning outcomes"],
            ),
            SkillInsight(
                name="Learning Objectives",
                category="domain",
                frequency=1,
                importance=0.8,
                evidence=["learning objectives"],
            ),
        ],
        nice_to_have=["No explicit nice-to-haves mentioned", "Experience with Canvas"],
    )

    cleaned = _apply_skill_canonicalization(
        result,
        [
            _SkillCanonicalizationItem(
                original="IDEA Public Schools",
                canonical=None,
                keep=False,
            ),
            _SkillCanonicalizationItem(
                original="Learning Outcomes",
                canonical="Learning Objectives",
                keep=True,
                category="domain",
            ),
            _SkillCanonicalizationItem(
                original="Learning Objectives",
                canonical="Learning Objectives",
                keep=True,
                category="domain",
            ),
        ],
    )

    assert [skill.name for skill in cleaned.top_skills] == ["Learning Objectives"]
    assert cleaned.top_skills[0].frequency == 2
    assert _clean_nice_to_have(result.nice_to_have) == ["Experience with Canvas"]


async def test_qwen_lora_extractor_does_not_fallback_on_invalid_output() -> None:
    class _FakeQwen(QwenLoRAExtractor):
        def __init__(self) -> None:
            pass

        def _invoke(self, _messages: object) -> str:
            return "not json"

    with pytest.raises(ValueError, match="Model response did not contain a JSON object"):
        await _FakeQwen().extract([_posting(1)], "Backend Engineer")


async def test_qwen_lora_synthesise_parses_final_narrative() -> None:
    class _FakeQwen(QwenLoRAExtractor):
        def __init__(self) -> None:
            pass

        def _invoke(self, _messages: object) -> str:
            return """
            {
              "summary": "Applied AI roles emphasize production RAG and model evaluation.",
              "core_responsibilities": ["Build RAG systems.", "Evaluate model quality."],
              "nice_to_have": ["LoRA fine-tuning"]
            }
            """

    output = await _FakeQwen().synthesise(
        [_posting(1)],
        "Applied AI Engineer",
        [
            SkillInsight(
                name="RAG",
                category="domain",
                frequency=1,
                importance=0.9,
                evidence=["RAG"],
            )
        ],
        [],
    )

    assert output.summary.startswith("Applied AI roles")
    assert output.core_responsibilities == ["Build RAG systems.", "Evaluate model quality."]


async def test_qwen_lora_synthesise_falls_back_from_markdown_text() -> None:
    class _FakeQwen(QwenLoRAExtractor):
        def __init__(self) -> None:
            pass

        def _invoke(self, _messages: object) -> str:
            return """
            ### Consolidated Job-Market Analysis

            #### Summary
            Applied AI Engineer roles focus on production AI systems, evaluation, and deployment.

            Core responsibilities:
            - Build AI features.
            - Evaluate model quality.

            Nice-to-have:
            - LoRA fine-tuning
            """

    output = await _FakeQwen().synthesise(
        [_posting(1)],
        "Applied AI Engineer",
        [],
        [],
    )

    assert "production AI systems" in output.summary
    assert "Build AI features." in output.core_responsibilities
    assert output.nice_to_have == ["LoRA fine-tuning"]


async def test_qwen_service_extractor_calls_http(respx_mock: MockRouter) -> None:
    posting = _posting(1)
    service_result = await MockExtractor().extract([posting], "Backend Engineer")
    route = respx_mock.post("http://qwen.local/extract").mock(
        return_value=httpx.Response(200, json=service_result.model_dump(mode="json"))
    )
    respx_mock.post("http://qwen.local/synthesise").mock(
        return_value=httpx.Response(
            200,
            json={
                "summary": "Synthesised backend role.",
                "core_responsibilities": ["Build APIs."],
                "nice_to_have": ["Kubernetes"],
            },
        )
    )
    respx_mock.post("http://qwen.local/canonicalise-skills").mock(
        return_value=httpx.Response(
            200,
            json={
                "skills": [
                    {
                        "original": skill.name,
                        "canonical": skill.name,
                        "keep": True,
                        "category": skill.category,
                    }
                    for skill in service_result.top_skills
                ]
            },
        )
    )
    extractor = QwenServiceExtractor(
        Settings(
            llm_provider="qwen_service",
            qwen_service_url="http://qwen.local",
            finalizer_provider="qwen_service",
        )
    )

    result = await extractor.extract([posting], "Backend Engineer")

    assert route.called
    assert result.summary == "Synthesised backend role."


async def test_qwen_service_extractor_batches_requests(respx_mock: MockRouter) -> None:
    postings = [_posting(1), _posting(2), _posting(3)]
    results = [
        await MockExtractor().extract([posting], "Backend Engineer") for posting in postings
    ]
    route = respx_mock.post("http://qwen.local/extract").mock(
        side_effect=[
            httpx.Response(200, json=result.model_dump(mode="json")) for result in results
        ]
    )
    respx_mock.post("http://qwen.local/synthesise").mock(
        return_value=httpx.Response(
            200,
            json={
                "summary": "Synthesised backend role.",
                "core_responsibilities": ["Build APIs."],
                "nice_to_have": [],
            },
        )
    )
    respx_mock.post("http://qwen.local/canonicalise-skills").mock(
        return_value=httpx.Response(
            200,
            json={
                "skills": [
                    {
                        "original": "python",
                        "canonical": "python",
                        "keep": True,
                        "category": "language",
                    }
                ]
            },
        )
    )
    extractor = QwenServiceExtractor(
        Settings(
            llm_provider="qwen_service",
            qwen_service_url="http://qwen.local",
            qwen_service_batch_size=1,
            qwen_service_max_concurrency=2,
            finalizer_provider="qwen_service",
        )
    )

    result = await extractor.extract(postings, "Backend Engineer")

    assert route.call_count == 3
    assert result.postings_analysed == 3
    assert result.postings == postings
    python_skill = next(skill for skill in result.top_skills if skill.name == "python")
    assert python_skill.frequency == 3


async def test_qwen_service_extractor_frequency_counts_posting_results(
    respx_mock: MockRouter,
) -> None:
    postings = [_posting(1), _posting(2)]
    batch_result_1 = AnalysisResult(
        request=AnalysisRequest(job_title="Backend Engineer"),
        postings_analysed=1,
        top_skills=[
            SkillInsight(
                name="Python",
                category="language",
                frequency=4,
                importance=0.8,
                evidence=["Python"],
            )
        ],
        summary="Python role.",
        postings=[postings[0]],
    )
    batch_result_2 = AnalysisResult(
        request=AnalysisRequest(job_title="Backend Engineer"),
        postings_analysed=1,
        top_skills=[
            SkillInsight(
                name="Python",
                category="language",
                frequency=7,
                importance=0.9,
                evidence=["Python again"],
            )
        ],
        summary="Python role.",
        postings=[postings[1]],
    )
    route = respx_mock.post("http://qwen.local/extract").mock(
        side_effect=[
            httpx.Response(200, json=batch_result_1.model_dump(mode="json")),
            httpx.Response(200, json=batch_result_2.model_dump(mode="json")),
        ]
    )
    respx_mock.post("http://qwen.local/synthesise").mock(
        return_value=httpx.Response(
            200,
            json={
                "summary": "Synthesised Python role.",
                "core_responsibilities": ["Build Python services."],
                "nice_to_have": [],
            },
        )
    )
    respx_mock.post("http://qwen.local/canonicalise-skills").mock(
        return_value=httpx.Response(
            200,
            json={
                "skills": [
                    {
                        "original": "Python",
                        "canonical": "Python",
                        "keep": True,
                        "category": "language",
                    }
                ]
            },
        )
    )
    extractor = QwenServiceExtractor(
        Settings(
            llm_provider="qwen_service",
            qwen_service_url="http://qwen.local",
            qwen_service_batch_size=1,
            qwen_service_max_concurrency=1,
            finalizer_provider="qwen_service",
        )
    )

    result = await extractor.extract(postings, "Backend Engineer")

    assert route.call_count == 2
    python_skill = next(skill for skill in result.top_skills if skill.name == "Python")
    assert python_skill.frequency == 2
    assert python_skill.importance == 0.9


async def test_qwen_service_extractor_canonicalises_obvious_skill_variants(
    respx_mock: MockRouter,
) -> None:
    postings = [_posting(1), _posting(2), _posting(3), _posting(4)]
    batch_results = [
        AnalysisResult(
            request=AnalysisRequest(job_title="Instructional Designer"),
            postings_analysed=1,
            top_skills=[
                SkillInsight(
                    name="Articulate",
                    category="tool",
                    frequency=1,
                    importance=0.8,
                    evidence=["Articulate"],
                )
            ],
            summary="x",
            postings=[postings[0]],
        ),
        AnalysisResult(
            request=AnalysisRequest(job_title="Instructional Designer"),
            postings_analysed=1,
            top_skills=[
                SkillInsight(
                    name="Articulate 360",
                    category="tool",
                    frequency=1,
                    importance=0.9,
                    evidence=["Articulate 360"],
                )
            ],
            summary="x",
            postings=[postings[1]],
        ),
        AnalysisResult(
            request=AnalysisRequest(job_title="Instructional Designer"),
            postings_analysed=1,
            top_skills=[
                SkillInsight(
                    name="Canvas",
                    category="platform",
                    frequency=1,
                    importance=0.9,
                    evidence=["Canvas"],
                )
            ],
            summary="x",
            postings=[postings[2]],
        ),
        AnalysisResult(
            request=AnalysisRequest(job_title="Instructional Designer"),
            postings_analysed=1,
            top_skills=[
                SkillInsight(
                    name="Canva",
                    category="tool",
                    frequency=1,
                    importance=0.9,
                    evidence=["Canva"],
                )
            ],
            summary="x",
            postings=[postings[3]],
        ),
    ]
    respx_mock.post("http://qwen.local/extract").mock(
        side_effect=[
            httpx.Response(200, json=result.model_dump(mode="json"))
            for result in batch_results
        ]
    )
    respx_mock.post("http://qwen.local/synthesise").mock(
        return_value=httpx.Response(
            200,
            json={
                "summary": "Synthesised instructional design role.",
                "core_responsibilities": [],
                "nice_to_have": [],
            },
        )
    )
    respx_mock.post("http://qwen.local/canonicalise-skills").mock(
        return_value=httpx.Response(
            200,
            json={
                "skills": [
                    {
                        "original": "Articulate",
                        "canonical": "Articulate 360",
                        "keep": True,
                        "category": "tool",
                    },
                    {
                        "original": "Articulate 360",
                        "canonical": "Articulate 360",
                        "keep": True,
                        "category": "tool",
                    },
                    {
                        "original": "Canvas",
                        "canonical": "Canvas",
                        "keep": True,
                        "category": "platform",
                    },
                    {
                        "original": "Canva",
                        "canonical": "Canva",
                        "keep": True,
                        "category": "tool",
                    },
                ]
            },
        )
    )
    extractor = QwenServiceExtractor(
        Settings(
            llm_provider="qwen_service",
            qwen_service_url="http://qwen.local",
            qwen_service_batch_size=1,
            qwen_service_max_concurrency=1,
            finalizer_provider="qwen_service",
        )
    )

    result = await extractor.extract(postings, "Instructional Designer")

    skills = {skill.name: skill.frequency for skill in result.top_skills}
    assert skills["Articulate 360"] == 2
    assert skills["Canvas"] == 1
    assert skills["Canva"] == 1


async def test_qwen_service_extractor_can_use_openai_finalizer(
    respx_mock: MockRouter,
) -> None:
    class _FakeFinalizer:
        async def canonicalise_skills(
            self,
            _job_title: str,
            skills: list[SkillInsight],
        ) -> object:
            return type(
                "_Output",
                (),
                {
                    "skills": [
                        _SkillCanonicalizationItem(
                            original=skill.name,
                            canonical=skill.name,
                            keep=True,
                            category=skill.category,
                        )
                        for skill in skills
                    ]
                },
            )()

        async def synthesise(
            self,
            _postings: list[JobPosting],
            _job_title: str,
            _top_skills: list[SkillInsight],
            _nice_to_have: list[str],
        ) -> object:
            return type(
                "_Output",
                (),
                {
                    "summary": "OpenAI final summary.",
                    "core_responsibilities": ["Build services."],
                    "nice_to_have": ["Kubernetes"],
                },
            )()

    posting = _posting(1)
    service_result = await MockExtractor().extract([posting], "Backend Engineer")
    respx_mock.post("http://qwen.local/extract").mock(
        return_value=httpx.Response(200, json=service_result.model_dump(mode="json"))
    )
    extractor = QwenServiceExtractor(
        Settings(
            llm_provider="qwen_service",
            qwen_service_url="http://qwen.local",
            finalizer_provider="openai",
        )
    )
    extractor._openai_finalizer = _FakeFinalizer()  # type: ignore[assignment]

    result = await extractor.extract([posting], "Backend Engineer")

    assert result.summary == "OpenAI final summary."
    assert result.core_responsibilities == ["Build services."]


async def test_llm_extractor_falls_back_on_chain_error() -> None:
    class _BoomChain:
        async def ainvoke(self, _messages: object) -> _LLMOutput:
            raise RuntimeError("rate limited")

    extractor = LLMExtractor.__new__(LLMExtractor)
    extractor.settings = None  # type: ignore[assignment]
    extractor._chain = _BoomChain()  # type: ignore[attr-defined]

    result = await extractor.extract([_posting(1)], "Backend Engineer")

    assert result.summary.startswith("(mock)")
    assert result.postings_analysed == 1
