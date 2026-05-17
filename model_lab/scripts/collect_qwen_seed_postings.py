"""Collect unlabeled job postings for Qwen training seed JSONL.

This script writes rows shaped for ``generate_qwen_training_data.py``:

    {"job_title": "...", "postings": [{...}, {...}]}

It uses the project's configured scraper interface. The default source is
SerpApi Google Jobs because it is API-backed and avoids brittle/login-gated
job-board scraping.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

from ai_job_analyzer.scrapers.base import ScrapeQuery
from ai_job_analyzer.scrapers.registry import get_scraper

DEFAULT_ROLES = [
    "Software Engineer",
    "Backend Engineer",
    "Frontend Engineer",
    "Full Stack Engineer",
    "DevOps Engineer",
    "Site Reliability Engineer",
    "Cloud Engineer",
    "Platform Engineer",
    "Security Engineer",
    "Data Engineer",
    "Analytics Engineer",
    "Data Analyst",
    "Business Intelligence Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "Applied AI Engineer",
    "AI Engineer",
    "MLOps Engineer",
    "NLP Engineer",
    "Computer Vision Engineer",
    "Research Scientist",
    "Product Manager",
    "Technical Product Manager",
    "Product Analyst",
    "UX Designer",
    "UX Researcher",
    "QA Engineer",
    "Automation Test Engineer",
    "Solutions Architect",
    "Sales Engineer",
    "Technical Account Manager",
    "Customer Success Manager",
    "Project Manager",
    "Program Manager",
    "Scrum Master",
    "Digital Marketing Manager",
    "Growth Marketing Manager",
    "Operations Analyst",
    "Financial Analyst",
    "Cybersecurity Analyst",
]


def _register_scrapers() -> None:
    # Import scraper modules so their @register_scraper decorators run.
    importlib.import_module("ai_job_analyzer.scrapers.mock")
    importlib.import_module("ai_job_analyzer.scrapers.serpapi")


def _load_roles(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_ROLES
    role_path = Path(path)
    if role_path.suffix.lower() == ".json":
        data = json.loads(role_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Role JSON must be a list of strings.")
        return [item.strip() for item in data if item.strip()]
    return [
        line.strip()
        for line in role_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _posting_for_seed(posting: Any) -> dict[str, Any]:
    payload = posting.model_dump(mode="json")
    return {
        "source": payload.get("source"),
        "source_id": payload.get("source_id"),
        "url": payload.get("url"),
        "title": payload.get("title"),
        "company": payload.get("company"),
        "location": payload.get("location"),
        "posted_at": payload.get("posted_at"),
        "description": payload.get("description"),
    }


async def _collect_one(
    *,
    role: str,
    source: str,
    location: str | None,
    per_role: int,
    min_description_chars: int,
) -> dict[str, Any]:
    scraper = get_scraper(source)
    configured, reason = scraper.is_configured()
    if not configured:
        raise RuntimeError(f"Scraper {source!r} is not configured: {reason}")

    postings = await scraper.search(
        ScrapeQuery(job_title=role, location=location, max_results=per_role)
    )
    seen: set[str] = set()
    seed_postings: list[dict[str, Any]] = []
    for posting in postings:
        description = posting.description.strip()
        if len(description) < min_description_chars:
            continue
        key = str(posting.url).lower().rstrip("/") or posting.source_id or posting.title
        if key in seen:
            continue
        seen.add(key)
        seed_postings.append(_posting_for_seed(posting))
        if len(seed_postings) >= per_role:
            break
    return {"job_title": role, "postings": seed_postings}


async def _collect(args: argparse.Namespace) -> None:
    roles = _load_roles(args.roles)
    if args.limit_roles is not None:
        roles = roles[: args.limit_roles]

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(role: str) -> dict[str, Any]:
        async with semaphore:
            row = await _collect_one(
                role=role,
                source=args.source,
                location=args.location,
                per_role=args.per_role,
                min_description_chars=args.min_description_chars,
            )
            print(f"{role}: collected {len(row['postings'])}/{args.per_role}")
            return row

    rows = await asyncio.gather(*(guarded(role) for role in roles))
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if row["postings"] or args.keep_empty:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Wrote seed data for {len(rows)} roles to {output_path}")


def main() -> None:
    _register_scrapers()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="model_lab/data/qwen_skill_seed.generated.jsonl",
        help="Seed JSONL output path.",
    )
    parser.add_argument(
        "--source",
        default="serpapi",
        choices=["serpapi", "mock"],
        help="Scraper source. SerpApi requires SERPAPI_API_KEY.",
    )
    parser.add_argument("--roles", default=None, help="Optional .txt or .json role list.")
    parser.add_argument("--location", default="United States")
    parser.add_argument("--per-role", type=int, default=100)
    parser.add_argument("--limit-roles", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--min-description-chars", type=int, default=120)
    parser.add_argument("--keep-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asyncio.run(_collect(args))


if __name__ == "__main__":
    main()
