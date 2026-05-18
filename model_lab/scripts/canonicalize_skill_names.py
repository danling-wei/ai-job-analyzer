"""Build and apply a global skill-name canonicalization mapping.

This script is intentionally global: it collects skill names from training
labels and model predictions, creates one reusable alias -> canonical mapping,
then rewrites datasets/predictions with that same mapping.

Examples:

    uv run python model_lab/scripts/canonicalize_skill_names.py

    uv run python model_lab/scripts/canonicalize_skill_names.py \
      --mapping model_lab/data/skill_canonicalization/skill_canonicalization.mapping.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

Category = Literal["language", "framework", "tool", "platform", "domain", "soft_skill", "other"]

SKILL_ALIASES = {
    ".net": ".NET",
    "ab testing": "A/B testing",
    "a b testing": "A/B testing",
    "a/b testing": "A/B testing",
    "amazon web services": "Amazon Web Services (AWS)",
    "amazon web services aws": "Amazon Web Services (AWS)",
    "aws": "Amazon Web Services (AWS)",
    "aws cloud": "Amazon Web Services (AWS)",
    "azure": "Microsoft Azure",
    "gcp": "Google Cloud Platform (GCP)",
    "google cloud": "Google Cloud Platform (GCP)",
    "google cloud platform": "Google Cloud Platform (GCP)",
    "ibm mainframe environment": "IBM Mainframe",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "large language model": "Large Language Models (LLMs)",
    "large language models": "Large Language Models (LLMs)",
    "llm": "Large Language Models (LLMs)",
    "llms": "Large Language Models (LLMs)",
    "microsoft azure": "Microsoft Azure",
    "microsoft power bi": "Microsoft Power BI",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "power bi": "Microsoft Power BI",
    "powerbi": "Microsoft Power BI",
    "react.js": "React",
    "reactjs": "React",
    "retrieval augmented generation": "Retrieval-Augmented Generation (RAG)",
    "retrieval-augmented generation": "Retrieval-Augmented Generation (RAG)",
    "rag": "Retrieval-Augmented Generation (RAG)",
    "scikit learn": "scikit-learn",
    "test automation": "Automation Testing",
    "typescript": "TypeScript",
    "ts": "TypeScript",
}

LOWERCASE_CANONICALS = {"dbt", "scikit-learn"}
GENERIC_SUFFIXES = (
    "architecture",
    "best practices",
    "capabilities",
    "concepts",
    "development",
    "environment",
    "experience",
    "framework",
    "integration",
    "integrations",
    "management",
    "methodology",
    "platform",
    "processes",
    "reporting",
    "strategy",
    "tooling",
    "tools",
    "usage",
    "workflows",
)
GENERIC_PREFIXES = (
    "advanced",
    "basic",
    "certified",
    "enterprise",
    "hands-on",
    "microsoft",
)


@dataclass
class SkillOccurrence:
    name: str
    category: str
    source_kind: str
    source_path: str
    source_field: str
    row_id: str


@dataclass
class SkillCluster:
    key: str
    names: Counter[str] = field(default_factory=Counter)
    categories: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)

    def add(self, occurrence: SkillOccurrence) -> None:
        self.names[occurrence.name] += 1
        self.categories[occurrence.category or "other"] += 1
        self.sources[f"{occurrence.source_kind}:{occurrence.source_field}"] += 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(
                f"Skipping invalid JSONL row in {path} line {line_number}: {exc}",
                file=sys.stderr,
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _normalise(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("&", " and ")
    key = re.sub(r"\bpowerbi\b", "power bi", key)
    key = re.sub(r"\s*\(([a-z0-9.+# /-]{1,40})\)\s*$", "", key)
    key = re.sub(r"[/_]+", " ", key)
    key = re.sub(r"[^a-z0-9.+# -]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _base_key(name: str) -> str:
    if re.search(r"\bgood clinical practice\b", name, flags=re.IGNORECASE):
        return "good clinical practice"
    if re.fullmatch(r"\s*ReAct\s*", name) or (
        re.search(r"\bReAct\b", name)
        and not re.search(r"\b(chain|tool use|multi-agent)\b", name, flags=re.IGNORECASE)
        and re.search(
            r"\b(agent|reason|reasoning|acting)\b",
            name,
            flags=re.IGNORECASE,
        )
    ):
        return "react agent pattern"
    key = _normalise(name)
    if key in SKILL_ALIASES:
        return _normalise(SKILL_ALIASES[key])
    key = re.sub(r"\b(technologies|technology|skills|skill)\b$", "", key).strip()
    for suffix in GENERIC_SUFFIXES:
        if key.endswith(f" {suffix}") and len(key.split()) > 2:
            key = key[: -len(suffix)].strip()
            break
    for prefix in GENERIC_PREFIXES:
        if key.startswith(f"{prefix} ") and len(key.split()) > 2:
            key = key[len(prefix) :].strip()
            break
    return SKILL_ALIASES.get(key, key)


def _tokens(name: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9.+#-]+", _base_key(name)) if len(word) > 2}


def _similarity(left: str, right: str) -> float:
    left_key = _base_key(left)
    right_key = _base_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    left_tokens = _tokens(left_key)
    right_tokens = _tokens(right_key)
    jaccard = (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens and right_tokens
        else 0.0
    )
    sequence = SequenceMatcher(None, left_key, right_key).ratio()
    contained = (
        (left_key in right_key or right_key in left_key)
        and min(len(left_key), len(right_key)) >= 5
        and _same_skill_level(left_key, right_key)
    )
    return max(jaccard, sequence, 0.88 if contained else 0.0)


def _same_skill_level(left: str, right: str) -> bool:
    """Avoid merging broad platforms with specific sub-services."""
    broad_to_specific = (
        ("amazon web services", "lambda"),
        ("amazon web services", "s3"),
        ("amazon web services", "ec2"),
        ("amazon web services", "glue"),
        ("microsoft azure", "functions"),
        ("microsoft azure", "data factory"),
        ("google cloud platform", "bigquery"),
        ("sql", "server"),
        ("python", "pandas"),
        ("java", "javascript"),
    )
    pair = " ".join(sorted([left, right]))
    return all(not (broad in pair and specific in pair) for broad, specific in broad_to_specific)


def _canonical_display_name(cluster: SkillCluster) -> str:
    if cluster.key == "good clinical practice":
        return "Good Clinical Practice (GCP)"
    if cluster.key == "react agent pattern":
        return "ReAct (agent pattern)"
    alias_votes: Counter[str] = Counter()
    for name, count in cluster.names.items():
        if _base_key(name) in {"good clinical practice", "react agent pattern"}:
            continue
        alias = SKILL_ALIASES.get(_normalise(name))
        if alias:
            alias_votes[alias] += count + 1000
    if alias_votes:
        return alias_votes.most_common(1)[0][0]

    def score(item: tuple[str, int]) -> tuple[int, int, int, int]:
        name, count = item
        is_clean = int(
            not any(token in _normalise(name).split() for token in ("environment", "experience"))
        )
        has_acronym = int(bool(re.search(r"\b[A-Z]{2,}\b", name)))
        return (count, is_clean, has_acronym, -len(name))

    chosen = max(cluster.names.items(), key=score)[0]
    if _normalise(chosen) in LOWERCASE_CANONICALS:
        return _normalise(chosen)
    return chosen.strip()


def _collect_dataset_skills(paths: list[Path]) -> list[SkillOccurrence]:
    occurrences: list[SkillOccurrence] = []
    for path in paths:
        if not path.exists():
            continue
        for index, row in enumerate(_read_jsonl(path)):
            row_id = str(row.get("id") or row.get("job_title") or index)
            for skill in row.get("output", {}).get("top_skills", []) or []:
                name = str(skill.get("name") or "").strip()
                if name:
                    occurrences.append(
                        SkillOccurrence(
                            name=name,
                            category=str(skill.get("category") or "other"),
                            source_kind="dataset",
                            source_path=str(path),
                            source_field="output",
                            row_id=row_id,
                        )
                    )
    return occurrences


def _collect_prediction_skills(paths: list[Path]) -> list[SkillOccurrence]:
    occurrences: list[SkillOccurrence] = []
    for path in paths:
        if not path.exists():
            continue
        for index, row in enumerate(_read_jsonl(path)):
            row_id = str(row.get("id") or index)
            for field_name in ("prediction", "gold"):
                for skill in row.get(field_name, {}).get("top_skills", []) or []:
                    name = str(skill.get("name") or "").strip()
                    if name:
                        occurrences.append(
                            SkillOccurrence(
                                name=name,
                                category=str(skill.get("category") or "other"),
                                source_kind="prediction",
                                source_path=str(path),
                                source_field=field_name,
                                row_id=row_id,
                            )
                        )
    return occurrences


def _build_clusters(occurrences: list[SkillOccurrence], *, threshold: float) -> list[SkillCluster]:
    clusters_by_key: dict[str, SkillCluster] = {}
    for occurrence in occurrences:
        key = _base_key(occurrence.name)
        if not key:
            continue
        clusters_by_key.setdefault(key, SkillCluster(key=key)).add(occurrence)

    clusters = list(clusters_by_key.values())
    if threshold >= 1.0:
        return clusters

    parents = list(range(len(clusters)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    token_index: dict[str, set[int]] = defaultdict(set)
    for index, cluster in enumerate(clusters):
        for token in _tokens(cluster.key):
            token_index[token].add(index)

    compared: set[tuple[int, int]] = set()
    for index, cluster in enumerate(clusters):
        candidates: set[int] = set()
        for token in _tokens(cluster.key):
            if len(token_index[token]) <= 250:
                candidates.update(token_index[token])
        for candidate in candidates:
            if candidate <= index:
                continue
            pair = (index, candidate)
            if pair in compared:
                continue
            compared.add(pair)
            if _similarity(cluster.key, clusters[candidate].key) >= threshold:
                union(index, candidate)

    merged_by_root: dict[int, SkillCluster] = {}
    for index, cluster in enumerate(clusters):
        root = find(index)
        existing = merged_by_root.get(root)
        if existing is None:
            merged_by_root[root] = cluster
            continue
        existing.names.update(cluster.names)
        existing.categories.update(cluster.categories)
        existing.sources.update(cluster.sources)
        existing.key = _base_key(_canonical_display_name(existing))
    return list(merged_by_root.values())


def _mapping_from_clusters(clusters: list[SkillCluster]) -> dict[str, Any]:
    entries = []
    for cluster in sorted(clusters, key=lambda item: _canonical_display_name(item).lower()):
        canonical = _canonical_display_name(cluster)
        aliases = sorted(
            cluster.names,
            key=lambda name: (_normalise(name) != _normalise(canonical), name.lower()),
        )
        category = cluster.categories.most_common(1)[0][0] if cluster.categories else "other"
        confidence = 1.0 if len({_base_key(alias) for alias in aliases}) == 1 else 0.88
        entries.append(
            {
                "canonical_name": canonical,
                "category": category,
                "aliases": aliases,
                "confidence": confidence,
                "method": "heuristic_global",
                "occurrences": sum(cluster.names.values()),
                "source_counts": dict(sorted(cluster.sources.items())),
            }
        )
    return {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "description": (
            "Global skill-name canonicalization mapping shared by training labels "
            "and model predictions."
        ),
        "entry_count": len(entries),
        "entries": entries,
    }


def _alias_lookup(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for entry in mapping.get("entries", []):
        for alias in entry.get("aliases", []):
            lookup[_normalise(str(alias))] = entry
            lookup[_base_key(str(alias))] = entry
    return lookup


def _canonicalize_skills(
    skills: list[dict[str, Any]], lookup: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for skill in skills:
        name = str(skill.get("name") or "").strip()
        if not name:
            continue
        entry = lookup.get(_normalise(name)) or lookup.get(_base_key(name))
        canonical = str(entry.get("canonical_name") if entry else name)
        category = str(entry.get("category") if entry else skill.get("category") or "other")
        key = _base_key(canonical)
        existing = merged.get(key)
        if existing is None:
            new_skill = dict(skill)
            new_skill["name"] = canonical
            new_skill["category"] = category
            new_skill["aliases"] = sorted({name, *new_skill.get("aliases", [])})
            merged[key] = new_skill
            continue
        existing["importance"] = max(
            float(existing.get("importance") or 0.0),
            float(skill.get("importance") or 0.0),
        )
        existing["evidence"] = _merge_strings(
            existing.get("evidence") or [], skill.get("evidence") or [], 3
        )
        existing["aliases"] = _merge_strings(existing.get("aliases") or [], [name], 20)
    return sorted(
        merged.values(),
        key=lambda item: (
            -float(item.get("importance") or 0.0),
            str(item.get("name") or "").lower(),
        ),
    )


def _merge_strings(left: list[str], right: list[str], limit: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in [*left, *right]:
        cleaned = str(value).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            values.append(cleaned)
            seen.add(key)
        if len(values) >= limit:
            break
    return values


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
        key = _base_key(name)
        evidence = [str(item) for item in skill.get("evidence") or []]
        if key and re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text):
            supported += 1
            continue
        if any(_evidence_supported(snippet, text) for snippet in evidence):
            supported += 1
    return supported


def _apply_to_dataset(path: Path, output_path: Path, lookup: dict[str, dict[str, Any]]) -> None:
    rows = _read_jsonl(path)
    for row in rows:
        if "output" in row:
            row["output"]["top_skills"] = _canonicalize_skills(
                row["output"].get("top_skills") or [],
                lookup,
            )
    _write_jsonl(output_path, rows)


def _apply_to_predictions(path: Path, output_path: Path, lookup: dict[str, dict[str, Any]]) -> None:
    rows = _read_jsonl(path)
    for row in rows:
        for field_name in ("prediction", "gold"):
            if field_name in row:
                row[field_name]["top_skills"] = _canonicalize_skills(
                    row[field_name].get("top_skills") or [],
                    lookup,
                )
    _write_jsonl(output_path, rows)


def _skill_keys(skills: list[dict[str, Any]]) -> set[str]:
    return {
        _base_key(str(skill.get("name") or ""))
        for skill in skills
        if _base_key(str(skill.get("name") or ""))
    }


def _load_original_metrics(path: Path) -> dict[str, Any]:
    run_name = path.name.removesuffix(".predictions.jsonl")
    metrics_path = Path("model_lab/eval_runs") / f"{run_name}.metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def _metrics_from_predictions(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    original = _load_original_metrics(path)
    tp = pred_total = gold_total = supported = support_den = 0
    per_role: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        pred = _skill_keys(row.get("prediction", {}).get("top_skills") or [])
        gold = _skill_keys(row.get("gold", {}).get("top_skills") or [])
        matched = len(pred & gold)
        tp += matched
        pred_total += len(pred)
        gold_total += len(gold)

        pred_skills = row.get("prediction", {}).get("top_skills") or []
        posting_text = "\n".join(
            str(posting.get("description") or "") for posting in row.get("postings") or []
        )
        supported_count = _count_supported_skills(pred_skills, posting_text)
        supported += supported_count
        support_den += len(pred_skills)

        role = str(row.get("job_title") or "Unknown")
        per_role[role]["tp"] += matched
        per_role[role]["pred"] += len(pred)
        per_role[role]["gold"] += len(gold)
        per_role[role]["supported"] += supported_count
        per_role[role]["support_den"] += len(pred_skills)
        per_role[role]["n"] += 1

    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
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
    run_name = path.name.removesuffix(".predictions.jsonl")
    return {
        "run_name": f"{run_name}-canonicalized",
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
        "per_role": role_rows,
        "requested_samples": original.get("requested_samples", len(rows)),
        "failed_samples": original.get("failed_samples", 0),
        "failure_rate": original.get("failure_rate", 0.0),
        "prediction_path": str(path),
    }


def _default_existing(paths: list[str]) -> list[Path]:
    return [Path(path) for path in paths if Path(path).exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help="Training/eval JSONL file containing output.top_skills. Can be repeated.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        dest="predictions",
        default=None,
        help="Prediction JSONL file containing prediction/gold top_skills. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_lab/data/skill_canonicalization",
        help="Directory for the mapping and inventory.",
    )
    parser.add_argument(
        "--canonicalized-data-dir",
        default="model_lab/data/canonicalized",
        help="Directory for canonicalized dataset copies.",
    )
    parser.add_argument(
        "--canonicalized-eval-dir",
        default="model_lab/eval_runs/canonicalized",
        help="Directory for canonicalized prediction copies and metrics.",
    )
    parser.add_argument(
        "--mapping",
        default="model_lab/data/skill_canonicalization/skill_canonicalization.mapping.json",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=1.01,
        help=(
            "Use >=1.0 for deterministic alias/key grouping only. Lower values "
            "enable slower fuzzy merging."
        ),
    )
    args = parser.parse_args()

    datasets = (
        [Path(path) for path in args.datasets]
        if args.datasets
        else _default_existing(
            [
                "model_lab/data/eval/train.jsonl",
                "model_lab/data/eval/dev.jsonl",
                "model_lab/data/eval/test.jsonl",
            ]
        )
    )
    predictions = (
        [Path(path) for path in args.predictions]
        if args.predictions
        else _default_existing(
            [
                "model_lab/eval_runs/gpt-5-mini.predictions.jsonl",
                "model_lab/eval_runs/qwen3-1_7b-base.predictions.jsonl",
                "model_lab/eval_runs/qwen3-1_7b-qlora-r16a32-compact.predictions.jsonl",
            ]
        )
    )

    occurrences = [*_collect_dataset_skills(datasets), *_collect_prediction_skills(predictions)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "skill_occurrences.inventory.jsonl"
    _write_jsonl(inventory_path, [occurrence.__dict__ for occurrence in occurrences])

    clusters = _build_clusters(occurrences, threshold=args.similarity_threshold)
    mapping = _mapping_from_clusters(clusters)
    mapping_path = Path(args.mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lookup = _alias_lookup(mapping)
    canonicalized_data_dir = Path(args.canonicalized_data_dir)
    for dataset in datasets:
        output_path = canonicalized_data_dir / "eval" / dataset.name
        _apply_to_dataset(dataset, output_path, lookup)

    canonicalized_eval_dir = Path(args.canonicalized_eval_dir)
    for prediction in predictions:
        output_path = canonicalized_eval_dir / prediction.name
        _apply_to_predictions(prediction, output_path, lookup)
        metrics = _metrics_from_predictions(output_path)
        metrics["canonicalization_mapping"] = str(mapping_path)
        metrics_path = output_path.with_suffix(".canonical.metrics.json")
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "occurrences": len(occurrences),
                "mapping_entries": mapping["entry_count"],
                "mapping": str(mapping_path),
                "inventory": str(inventory_path),
                "canonicalized_datasets": str(canonicalized_data_dir),
                "canonicalized_predictions": str(canonicalized_eval_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
