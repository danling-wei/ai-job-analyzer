"""Generate Qwen fine-tuning JSONL labels with a closed-source OpenAI model.

Run from the repository root:

    uv run python model_lab/scripts/generate_qwen_training_data.py \
      --input model_lab/data/examples/qwen_skill_seed.example.jsonl \
      --output model_lab/data/qwen_skill_train.generated.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from ai_job_analyzer.core.config import get_settings


class SkillLabel(BaseModel):
    """A single concrete skill label for supervised Qwen training."""

    name: str = Field(..., description="Canonical concrete skill name, e.g. PyTorch.")
    category: Literal[
        "language", "framework", "tool", "platform", "domain", "soft_skill", "other"
    ] = "other"
    importance: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(
        default_factory=list,
        description="Up to 3 short verbatim snippets from the input postings.",
    )


class TrainingOutput(BaseModel):
    """Output object stored in each supervised training JSONL row."""

    top_skills: list[SkillLabel] = Field(default_factory=list)
    core_responsibilities: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    summary: str


SYSTEM_PROMPT = """You create supervised fine-tuning labels for a Qwen job-skill extractor.

Return a precise JSON object with:
- top_skills: concrete learnable tools, platforms, frameworks, languages, methods, and role-specific domains.
- core_responsibilities: distinct responsibilities across the postings.
- nice_to_have: only explicitly bonus/preferred qualifications.
- summary: one concise paragraph.

Rules:
- Prefer job-seeker skill-gap items that can be learned and verified in a portfolio.
- Do not include generic labels such as AI, machine learning, cloud platforms, APIs, collaboration, communication, project management, or software development unless a posting names a concrete subskill.
- Evidence must be copied from the provided postings and kept short.
- Deduplicate near-synonyms into one canonical skill name.
- If a skill is not supported by the postings, omit it.
"""


def _posting_text(posting: dict[str, Any], index: int) -> str:
    return "\n".join(
        [
            f"[{index}] {posting.get('title', 'Untitled')}",
            f"Company: {posting.get('company', 'Unknown')}",
            f"Location: {posting.get('location', 'n/a')}",
            str(posting.get("description", "")),
        ]
    )


def _messages_for_row(row: dict[str, Any], max_skills: int) -> list[tuple[str, str]]:
    postings = row.get("postings")
    if not isinstance(postings, list) or not postings:
        raise ValueError("Each input row must include a non-empty postings list.")

    posting_blocks = [
        _posting_text(posting, index)
        for index, posting in enumerate(postings, 1)
        if isinstance(posting, dict)
    ]
    user_prompt = (
        f"Target role: {row.get('job_title', 'Unknown')}\n"
        f"Maximum top_skills: {max_skills}\n\n"
        f"Postings:\n\n{chr(10).join(posting_blocks)}"
    )
    return [("system", SYSTEM_PROMPT), ("user", user_prompt)]


def _batched_posting_rows(row: dict[str, Any], postings_per_request: int) -> list[dict[str, Any]]:
    postings = row.get("postings")
    if not isinstance(postings, list) or not postings:
        raise ValueError("Each input row must include a non-empty postings list.")
    if postings_per_request <= 0:
        return [row]

    rows: list[dict[str, Any]] = []
    for start in range(0, len(postings), postings_per_request):
        batch = postings[start : start + postings_per_request]
        rows.append(
            {
                "job_title": row.get("job_title", "Unknown"),
                "postings": batch,
                "batch_index": start // postings_per_request + 1,
            }
        )
    return rows


def _normalise_output(output: TrainingOutput, max_skills: int) -> dict[str, Any]:
    payload = output.model_dump()
    payload["top_skills"] = payload["top_skills"][:max_skills]
    for skill in payload["top_skills"]:
        skill["evidence"] = skill.get("evidence", [])[:3]
    return payload


def _training_row(row: dict[str, Any], output: TrainingOutput, max_skills: int) -> dict[str, Any]:
    return {
        "job_title": row.get("job_title", "Unknown"),
        "postings": row["postings"],
        "output": _normalise_output(output, max_skills),
    }


async def _generate(args: argparse.Namespace) -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in .env or the environment.")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        api_key=SecretStr(settings.openai_api_key),
        base_url=settings.openai_base_url,
    )
    chain = llm.with_structured_output(TrainingOutput)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request_rows = [
        batch
        for row in rows
        for batch in _batched_posting_rows(row, args.postings_per_request)
    ]

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(request_rows, 1):
            output = await chain.ainvoke(_messages_for_row(row, args.max_skills))
            handle.write(
                json.dumps(
                    _training_row(row, output, args.max_skills),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            batch_index = row.get("batch_index")
            suffix = f" batch {batch_index}" if batch_index else ""
            print(f"Wrote {index}/{len(request_rows)}: {row.get('job_title', 'Unknown')}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Seed JSONL with job_title and postings.")
    parser.add_argument(
        "--output",
        default="model_lab/data/qwen_skill_train.generated.jsonl",
        help="Generated supervised training JSONL path.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Closed-source teacher model. Use gpt-5.2-pro if your account has access.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-skills", type=int, default=15)
    parser.add_argument(
        "--postings-per-request",
        type=int,
        default=1,
        help="How many postings to label per teacher-model request. Use 1 for one label per posting; use 0 to keep each input row whole.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asyncio.run(_generate(args))


if __name__ == "__main__":
    main()
