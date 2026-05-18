"""Evaluate skill extraction models against generated teacher labels.

Examples:

    # Pure Qwen3-1.7B. Start qwen service separately without QWEN_ADAPTER_PATH.
    uv run python model_lab/scripts/evaluate_skill_extractors.py \
      --dataset model_lab/data/eval/test.jsonl \
      --provider qwen_service \
      --run-name qwen3-1_7b-base

    # GPT-5 mini.
    uv run python model_lab/scripts/evaluate_skill_extractors.py \
      --dataset model_lab/data/eval/test.jsonl \
      --provider openai \
      --openai-model gpt-5-mini \
      --run-name gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ai_job_analyzer.models import AnalysisResult, JobPosting


class EvalSkill(BaseModel):
    name: str
    category: Literal[
        "language", "framework", "tool", "platform", "domain", "soft_skill", "other"
    ] = "other"
    importance: float = Field(0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class EvalOutput(BaseModel):
    top_skills: list[EvalSkill] = Field(default_factory=list)
    core_responsibilities: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    summary: str = ""


SYSTEM_PROMPT = """You extract a job-seeker skill-gap checklist from one job posting.

Return a precise JSON object with:
- top_skills: concrete learnable tools, platforms, frameworks, languages, methods, and role-specific domains.
- core_responsibilities: distinct responsibilities in the posting.
- nice_to_have: only explicitly bonus/preferred qualifications.
- summary: one concise paragraph.

Rules:
- Prefer job-seeker skill-gap items that can be learned and verified in a portfolio.
- Do not include generic labels such as AI, machine learning, cloud platforms, APIs, collaboration, communication, project management, or software development unless the posting names a concrete subskill.
- Evidence must be copied from the posting and kept short.
- Deduplicate near-synonyms into one canonical skill name.
- If a skill is not supported by the posting, omit it.
"""

SKILL_ALIASES = {
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node",
    "nodejs": "node",
    "scikit learn": "scikit-learn",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "large language models": "llm",
    "llms": "llm",
}


def _dotenv_value(name: str) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _skill_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("&", " and ")
    key = re.sub(r"\s*\(([a-z0-9.+#-]{1,12})\)\s*$", "", key)
    key = re.sub(r"[/_]+", " ", key)
    key = re.sub(r"[^a-z0-9.+# -]+", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return SKILL_ALIASES.get(key, key)


def _skill_keys(skills: list[dict[str, Any]]) -> set[str]:
    return {
        key
        for skill in skills
        if isinstance(skill, dict)
        for key in [_skill_key(str(skill.get("name") or ""))]
        if key
    }


def _count_supported_skills(skills: list[dict[str, Any]], posting_text: str) -> int:
    supported = 0
    text = posting_text.lower()
    for skill in skills:
        name = str(skill.get("name") or "")
        key = _skill_key(name)
        evidence = [str(item) for item in skill.get("evidence") or []]
        if key and re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text):
            supported += 1
            continue
        if any(_evidence_supported(snippet, text) for snippet in evidence):
            supported += 1
    return supported


def _evidence_supported(snippet: str, posting_text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", snippet.strip()).lower()
    if not cleaned:
        return False
    if cleaned in posting_text:
        return True
    words = [word for word in re.findall(r"[a-z0-9.+#-]+", cleaned) if len(word) > 2]
    return bool(words) and all(word in posting_text for word in words[:6])


def _posting_from_row(row: dict[str, Any]) -> JobPosting:
    posting = row["postings"][0]
    return JobPosting.model_validate(posting)


def _eval_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["top_skills", "core_responsibilities", "nice_to_have", "summary"],
        "properties": {
            "top_skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "category", "importance", "evidence"],
                    "properties": {
                        "name": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "language",
                                "framework",
                                "tool",
                                "platform",
                                "domain",
                                "soft_skill",
                                "other",
                            ],
                        },
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "core_responsibilities": {"type": "array", "items": {"type": "string"}},
            "nice_to_have": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
    }


def _row_id(row: dict[str, Any], index: int) -> str:
    posting = row["postings"][0]
    return str(posting.get("url") or posting.get("source_id") or index)


async def _extract_openai(
    client: httpx.AsyncClient,
    *,
    row: dict[str, Any],
    model: str,
    api_key: str,
    base_url: str,
) -> EvalOutput:
    posting = row["postings"][0]
    user = "\n".join(
        [
            f"Target role: {row.get('job_title', 'Unknown')}",
            "Maximum top_skills: 15",
            "",
            "Posting:",
            "",
            f"[1] {posting.get('title', 'Untitled')}",
            f"Company: {posting.get('company', 'Unknown')}",
            f"Location: {posting.get('location', 'n/a')}",
            str(posting.get("description", "")),
        ]
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "skill_extraction_eval_output",
                "strict": True,
                "schema": _eval_output_schema(),
            },
        },
    }
    response = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return EvalOutput.model_validate_json(content)


async def _extract_qwen_service(
    client: httpx.AsyncClient,
    *,
    row: dict[str, Any],
    qwen_service_url: str,
) -> EvalOutput:
    posting = _posting_from_row(row)
    payload = {
        "job_title": row.get("job_title", "Unknown"),
        "postings": [posting.model_dump(mode="json")],
    }
    response = await client.post(f"{qwen_service_url.rstrip('/')}/extract", json=payload)
    response.raise_for_status()
    result = AnalysisResult.model_validate(response.json())
    return EvalOutput(
        top_skills=[
            EvalSkill(
                name=skill.name,
                category=skill.category,
                importance=skill.importance,
                evidence=skill.evidence,
            )
            for skill in result.top_skills
        ],
        core_responsibilities=result.core_responsibilities,
        nice_to_have=result.nice_to_have,
        summary=result.summary,
    )


async def _run_prediction(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    prediction_path: Path,
) -> list[dict[str, Any]]:
    api_key = _dotenv_value("OPENAI_API_KEY")
    if args.provider == "openai" and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --provider openai.")

    done_ids: set[str] = set()
    predictions: list[dict[str, Any]] = []
    error_path = prediction_path.with_suffix(".errors.jsonl")
    if prediction_path.exists() and not args.overwrite_predictions:
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            predictions.append(record)
            done_ids.add(str(record["id"]))
    elif prediction_path.exists():
        prediction_path.unlink()
    if args.overwrite_predictions and error_path.exists():
        error_path.unlink()

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def predict_one(index: int, row: dict[str, Any]) -> dict[str, Any] | None:
            sample_id = _row_id(row, index)
            if sample_id in done_ids:
                return None
            async with semaphore:
                started = time.perf_counter()
                try:
                    if args.provider == "openai":
                        output = await _extract_openai(
                            client,
                            row=row,
                            model=args.openai_model,
                            api_key=str(api_key),
                            base_url=args.openai_base_url,
                        )
                    else:
                        output = await _extract_qwen_service(
                            client,
                            row=row,
                            qwen_service_url=args.qwen_service_url,
                        )
                except Exception as exc:
                    error_record = {
                        "id": sample_id,
                        "index": index,
                        "run_name": args.run_name,
                        "provider": args.provider,
                        "model": (
                            args.openai_model if args.provider == "openai" else args.qwen_model_name
                        ),
                        "job_title": row.get("job_title", "Unknown"),
                        "error": repr(exc),
                    }
                    _append_jsonl(error_path, error_record)
                    print(f"FAILED {args.run_name}: {row.get('job_title')} :: {exc!r}")
                    done_ids.add(sample_id)
                    return None
                record = {
                    "id": sample_id,
                    "index": index,
                    "run_name": args.run_name,
                    "provider": args.provider,
                    "model": args.openai_model
                    if args.provider == "openai"
                    else args.qwen_model_name,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "job_title": row.get("job_title", "Unknown"),
                    "prediction": output.model_dump(mode="json"),
                    "gold": row["output"],
                    "postings": row["postings"],
                }
                _append_jsonl(prediction_path, record)
                print(
                    f"Wrote {len(done_ids) + 1}/{len(rows)} {args.run_name}: {row.get('job_title')}"
                )
                done_ids.add(sample_id)
                return record

        tasks = [predict_one(index, row) for index, row in enumerate(rows)]
        for result in await asyncio.gather(*tasks):
            if result is not None:
                predictions.append(result)
    return predictions


def _metrics_from_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = total_pred = total_gold = supported = support_den = 0
    per_role: dict[str, Counter[str]] = defaultdict(Counter)
    for record in predictions:
        pred_skills = record["prediction"].get("top_skills") or []
        gold_skills = record["gold"].get("top_skills") or []
        pred_keys = _skill_keys(pred_skills)
        gold_keys = _skill_keys(gold_skills)
        tp = len(pred_keys & gold_keys)
        total_tp += tp
        total_pred += len(pred_keys)
        total_gold += len(gold_keys)
        posting_text = "\n".join(str(p.get("description") or "") for p in record["postings"])
        supported_count = _count_supported_skills(pred_skills, posting_text)
        supported += supported_count
        support_den += len(pred_skills)

        role = str(record.get("job_title") or "Unknown")
        per_role[role]["tp"] += tp
        per_role[role]["pred"] += len(pred_keys)
        per_role[role]["gold"] += len(gold_keys)
        per_role[role]["supported"] += supported_count
        per_role[role]["support_den"] += len(pred_skills)
        per_role[role]["n"] += 1

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    role_rows = []
    for role, counts in sorted(per_role.items()):
        role_precision = counts["tp"] / counts["pred"] if counts["pred"] else 0.0
        role_recall = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
        role_f1 = (
            2 * role_precision * role_recall / (role_precision + role_recall)
            if role_precision + role_recall
            else 0.0
        )
        role_rows.append(
            {
                "job_title": role,
                "samples": counts["n"],
                "skill_precision": role_precision,
                "skill_recall": role_recall,
                "skill_f1": role_f1,
                "faithfulness_proxy": (
                    counts["supported"] / counts["support_den"] if counts["support_den"] else 0.0
                ),
            }
        )
    return {
        "samples": len(predictions),
        "skill_precision": precision,
        "skill_recall": recall,
        "skill_f1": f1,
        "faithfulness_proxy": supported / support_den if support_den else 0.0,
        "predicted_skills": total_pred,
        "gold_skills": total_gold,
        "matched_skills": total_tp,
        "per_role": role_rows,
    }


async def _main_async(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.dataset))
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{args.run_name}.predictions.jsonl"
    metrics_path = output_dir / f"{args.run_name}.metrics.json"

    predictions = await _run_prediction(rows=rows, args=args, prediction_path=prediction_path)
    metrics = _metrics_from_predictions(predictions)
    error_path = prediction_path.with_suffix(".errors.jsonl")
    errors = (
        [line for line in error_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if error_path.exists()
        else []
    )
    metrics.update(
        {
            "run_name": args.run_name,
            "provider": args.provider,
            "model": args.openai_model if args.provider == "openai" else args.qwen_model_name,
            "dataset": args.dataset,
            "prediction_path": str(prediction_path),
            "requested_samples": len(rows),
            "failed_samples": len(errors),
            "failure_rate": len(errors) / len(rows) if rows else 0.0,
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="model_lab/data/eval/test.jsonl")
    parser.add_argument("--output-dir", default="model_lab/eval_runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--provider", choices=["openai", "qwen_service"], required=True)
    parser.add_argument("--openai-model", default="gpt-5-mini")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--qwen-service-url", default="http://127.0.0.1:8010")
    parser.add_argument("--qwen-model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
