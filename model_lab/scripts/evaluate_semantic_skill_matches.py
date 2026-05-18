"""Evaluate skill extraction with cached semantic skill-name matching.

This sits after the global canonicalization pass. It first counts exact
canonical-name matches, then asks an LLM judge only about plausible unmatched
prediction/gold pairs. Judgments are cached and reused across runs.

Examples:

    uv run python model_lab/scripts/evaluate_semantic_skill_matches.py \
      model_lab/eval_runs/canonicalized/gpt-5-mini.predictions.jsonl

    uv run python model_lab/scripts/evaluate_semantic_skill_matches.py \
      model_lab/eval_runs/canonicalized/*.predictions.jsonl \
      --judge-model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

SYSTEM_PROMPT = """You judge whether two extracted job skills should count as the same skill for evaluation.

Return JSON only.

Count as MATCH when:
- They are aliases, spelling variants, acronym expansions, or product-name variants of the same skill.
- One name is a slightly more specific phrasing of the same skill, without changing the underlying technology/method.
- A bundled prediction reasonably covers the gold item, e.g. "C/C++" covers "C++".

Count as NOT MATCH when:
- One is only a broad parent category of the other, e.g. "AWS" vs "AWS Lambda", "data analysis" vs "pandas".
- They are different tools, different products, or different skill levels.
- An acronym is ambiguous and the names do not clarify the same meaning, e.g. React vs ReAct.

Be conservative. The goal is fair evaluation, not inflating scores.
"""


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
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _skill_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("&", " and ")
    key = re.sub(r"\s*\(([a-z0-9.+# /-]{1,40})\)\s*$", "", key)
    key = re.sub(r"[/_]+", " ", key)
    key = re.sub(r"[^a-z0-9.+# -]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _tokens(name: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9.+#-]+", _skill_key(name)) if len(word) > 2}


def _similarity(left: str, right: str) -> float:
    left_key = _skill_key(left)
    right_key = _skill_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    jaccard = (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens and right_tokens
        else 0.0
    )
    sequence = SequenceMatcher(None, left_key, right_key).ratio()
    contained = (left_key in right_key or right_key in left_key) and min(
        len(left_key), len(right_key)
    ) >= 5
    return max(jaccard, sequence, 0.86 if contained else 0.0)


def _judgment_key(pred_name: str, gold_name: str) -> str:
    return f"{_skill_key(pred_name)} ||| {_skill_key(gold_name)}"


def _load_judgment_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    for row in _read_jsonl(path):
        key = str(row.get("key") or "")
        if key:
            cache[key] = row
    return cache


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["judgments"],
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "match", "confidence", "relation", "reason"],
                    "properties": {
                        "id": {"type": "integer"},
                        "match": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "relation": {
                            "type": "string",
                            "enum": [
                                "alias",
                                "same_skill",
                                "covers_gold",
                                "too_broad",
                                "different",
                                "ambiguous",
                            ],
                        },
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    }


@dataclass(frozen=True)
class Candidate:
    key: str
    pred_name: str
    pred_category: str
    gold_name: str
    gold_category: str
    similarity: float


async def _judge_batch(
    client: httpx.AsyncClient,
    *,
    candidates: list[Candidate],
    model: str,
    api_key: str,
    base_url: str,
) -> list[dict[str, Any]]:
    items = [
        {
            "id": index,
            "predicted_skill": candidate.pred_name,
            "predicted_category": candidate.pred_category,
            "gold_skill": candidate.gold_name,
            "gold_category": candidate.gold_category,
            "string_similarity": round(candidate.similarity, 3),
        }
        for index, candidate in enumerate(candidates)
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Judge these candidate pairs. Return one judgment per id.\n\n"
                    + json.dumps({"pairs": items}, ensure_ascii=False)
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_skill_match_judgments",
                "strict": True,
                "schema": _schema(),
            },
        },
    }
    response = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"},
        content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    response.raise_for_status()
    body = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(body)
    judgments = parsed["judgments"]
    by_id = {int(item["id"]): item for item in judgments}
    return [by_id[index] for index in range(len(candidates))]


def _candidate_pairs_for_row(
    pred_skills: list[dict[str, Any]],
    gold_skills: list[dict[str, Any]],
    *,
    threshold: float,
    max_candidates_per_pred: int,
) -> tuple[int, list[Candidate]]:
    pred_items = [
        (_skill_key(str(skill.get("name") or "")), skill)
        for skill in pred_skills
        if _skill_key(str(skill.get("name") or ""))
    ]
    gold_items = [
        (_skill_key(str(skill.get("name") or "")), skill)
        for skill in gold_skills
        if _skill_key(str(skill.get("name") or ""))
    ]
    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    exact = 0
    for pred_index, (pred_key, _pred_skill) in enumerate(pred_items):
        for gold_index, (gold_key, _gold_skill) in enumerate(gold_items):
            if (
                pred_index not in matched_pred
                and gold_index not in matched_gold
                and pred_key == gold_key
            ):
                matched_pred.add(pred_index)
                matched_gold.add(gold_index)
                exact += 1

    candidates: list[Candidate] = []
    for pred_index, (_pred_key, pred_skill) in enumerate(pred_items):
        if pred_index in matched_pred:
            continue
        scored: list[Candidate] = []
        pred_name = str(pred_skill.get("name") or "")
        for gold_index, (_gold_key, gold_skill) in enumerate(gold_items):
            if gold_index in matched_gold:
                continue
            gold_name = str(gold_skill.get("name") or "")
            similarity = _similarity(pred_name, gold_name)
            if similarity < threshold:
                continue
            scored.append(
                Candidate(
                    key=_judgment_key(pred_name, gold_name),
                    pred_name=pred_name,
                    pred_category=str(pred_skill.get("category") or "other"),
                    gold_name=gold_name,
                    gold_category=str(gold_skill.get("category") or "other"),
                    similarity=similarity,
                )
            )
        scored.sort(key=lambda item: item.similarity, reverse=True)
        candidates.extend(scored[:max_candidates_per_pred])
    return exact, candidates


def _evidence_supported(snippet: str, posting_text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", snippet.strip()).lower()
    if not cleaned:
        return False
    if cleaned in posting_text:
        return True
    words = [word for word in re.findall(r"[a-z0-9.+#-]+", cleaned) if len(word) > 2]
    return bool(words) and all(word in posting_text for word in words[:6])


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


async def _ensure_judgments(
    candidates: dict[str, Candidate],
    *,
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing = [candidate for key, candidate in candidates.items() if key not in cache]
    if args.limit_judgments is not None:
        missing = missing[: args.limit_judgments]
    if not missing:
        return cache
    api_key = _dotenv_value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for semantic LLM judge.")

    timeout = httpx.Timeout(args.timeout_seconds)
    cache_path = Path(args.judgment_cache)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start : start + args.batch_size]
            judgments = await _judge_batch(
                client,
                candidates=batch,
                model=args.judge_model,
                api_key=api_key,
                base_url=args.openai_base_url,
            )
            for candidate, judgment in zip(batch, judgments, strict=True):
                row = {
                    "key": candidate.key,
                    "predicted_skill": candidate.pred_name,
                    "gold_skill": candidate.gold_name,
                    "predicted_category": candidate.pred_category,
                    "gold_category": candidate.gold_category,
                    "string_similarity": candidate.similarity,
                    "judge_model": args.judge_model,
                    "match": bool(judgment["match"]),
                    "confidence": float(judgment["confidence"]),
                    "relation": judgment["relation"],
                    "reason": judgment["reason"],
                }
                cache[candidate.key] = row
                _append_jsonl(cache_path, row)
            print(f"Judged {min(start + len(batch), len(missing))}/{len(missing)} new pairs")
    return cache


def _semantic_matches_for_row(
    candidates: list[Candidate],
    cache: dict[str, dict[str, Any]],
    *,
    min_confidence: float,
) -> int:
    accepted = [
        candidate
        for candidate in candidates
        if candidate.key in cache
        and cache[candidate.key].get("match")
        and float(cache[candidate.key].get("confidence") or 0.0) >= min_confidence
    ]
    accepted.sort(
        key=lambda candidate: (
            float(cache[candidate.key].get("confidence") or 0.0),
            candidate.similarity,
        ),
        reverse=True,
    )
    used_pred: set[str] = set()
    used_gold: set[str] = set()
    matches = 0
    for candidate in accepted:
        pred_key = _skill_key(candidate.pred_name)
        gold_key = _skill_key(candidate.gold_name)
        if pred_key in used_pred or gold_key in used_gold:
            continue
        used_pred.add(pred_key)
        used_gold.add(gold_key)
        matches += 1
    return matches


def _load_original_metrics(path: Path) -> dict[str, Any]:
    run_name = path.name.removesuffix(".predictions.jsonl")
    original_path = Path("model_lab/eval_runs") / f"{run_name}.metrics.json"
    if original_path.exists():
        return json.loads(original_path.read_text(encoding="utf-8"))
    canonical_path = path.with_suffix(".canonical.metrics.json")
    if canonical_path.exists():
        return json.loads(canonical_path.read_text(encoding="utf-8"))
    return {}


def _metrics_for_path(
    path: Path,
    *,
    cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Candidate]]:
    rows = _read_jsonl(path)
    original = _load_original_metrics(path)
    all_candidates: dict[str, Candidate] = {}
    row_candidates: list[tuple[int, list[Candidate], dict[str, Any]]] = []
    for row in rows:
        exact, candidates = _candidate_pairs_for_row(
            row.get("prediction", {}).get("top_skills") or [],
            row.get("gold", {}).get("top_skills") or [],
            threshold=args.candidate_threshold,
            max_candidates_per_pred=args.max_candidates_per_pred,
        )
        for candidate in candidates:
            all_candidates.setdefault(candidate.key, candidate)
        row_candidates.append((exact, candidates, row))

    tp = exact_tp = semantic_tp = pred_total = gold_total = supported = support_den = 0
    per_role: dict[str, Counter[str]] = defaultdict(Counter)
    for exact, candidates, row in row_candidates:
        pred_skills = row.get("prediction", {}).get("top_skills") or []
        gold_skills = row.get("gold", {}).get("top_skills") or []
        pred_keys = {_skill_key(str(skill.get("name") or "")) for skill in pred_skills}
        gold_keys = {_skill_key(str(skill.get("name") or "")) for skill in gold_skills}
        pred_keys.discard("")
        gold_keys.discard("")
        semantic = _semantic_matches_for_row(candidates, cache, min_confidence=args.min_confidence)
        matched = exact + semantic
        exact_tp += exact
        semantic_tp += semantic
        tp += matched
        pred_total += len(pred_keys)
        gold_total += len(gold_keys)

        posting_text = "\n".join(
            str(posting.get("description") or "") for posting in row.get("postings") or []
        )
        supported_count = _count_supported_skills(pred_skills, posting_text)
        supported += supported_count
        support_den += len(pred_skills)

        role = str(row.get("job_title") or "Unknown")
        per_role[role]["tp"] += matched
        per_role[role]["exact_tp"] += exact
        per_role[role]["semantic_tp"] += semantic
        per_role[role]["pred"] += len(pred_keys)
        per_role[role]["gold"] += len(gold_keys)
        per_role[role]["supported"] += supported_count
        per_role[role]["support_den"] += len(pred_skills)
        per_role[role]["n"] += 1

    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_precision = exact_tp / pred_total if pred_total else 0.0
    exact_recall = exact_tp / gold_total if gold_total else 0.0
    exact_f1 = (
        2 * exact_precision * exact_recall / (exact_precision + exact_recall)
        if exact_precision + exact_recall
        else 0.0
    )
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
                "exact_matches": counts["exact_tp"],
                "semantic_matches": counts["semantic_tp"],
            }
        )

    run_name = path.name.removesuffix(".predictions.jsonl")
    metrics = {
        "run_name": f"{run_name}-semantic",
        "provider": original.get("provider", ""),
        "model": original.get("model", ""),
        "samples": len(rows),
        "skill_precision": precision,
        "skill_recall": recall,
        "skill_f1": f1,
        "faithfulness_proxy": supported / support_den if support_den else 0.0,
        "predicted_skills": pred_total,
        "gold_skills": gold_total,
        "matched_skills": tp,
        "exact_matches": exact_tp,
        "semantic_matches": semantic_tp,
        "exact_skill_precision": exact_precision,
        "exact_skill_recall": exact_recall,
        "exact_skill_f1": exact_f1,
        "candidate_pairs": len(all_candidates),
        "judge_model": args.judge_model,
        "judgment_cache": args.judgment_cache,
        "per_role": role_rows,
        "requested_samples": original.get("requested_samples", len(rows)),
        "failed_samples": original.get("failed_samples", 0),
        "failure_rate": original.get("failure_rate", 0.0),
        "prediction_path": str(path),
    }
    return metrics, all_candidates


async def _main_async(args: argparse.Namespace) -> None:
    paths = [Path(path) for path in args.predictions]
    cache_path = Path(args.judgment_cache)
    cache = _load_judgment_cache(cache_path)

    metrics_by_path: dict[Path, dict[str, Any]] = {}
    all_candidates: dict[str, Candidate] = {}
    for path in paths:
        metrics, candidates = _metrics_for_path(path, cache=cache, args=args)
        metrics_by_path[path] = metrics
        all_candidates.update(candidates)

    print(f"Candidate pairs: {len(all_candidates)}; cached judgments: {len(cache)}")
    if not args.no_judge:
        cache = await _ensure_judgments(all_candidates, args=args, cache=cache)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        metrics, _candidates = _metrics_for_path(path, cache=cache, args=args)
        output_path = (
            output_dir / f"{path.name.removesuffix('.predictions.jsonl')}.semantic.metrics.json"
        )
        output_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--output-dir", default="model_lab/eval_runs/semantic")
    parser.add_argument(
        "--judgment-cache",
        default="model_lab/data/skill_canonicalization/semantic_match_judgments.jsonl",
    )
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--candidate-threshold", type=float, default=0.80)
    parser.add_argument("--max-candidates-per-pred", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.72)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--limit-judgments", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
