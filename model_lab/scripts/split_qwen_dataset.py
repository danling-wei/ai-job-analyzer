"""Split generated Qwen training JSONL into train/dev/test files.

Run from the repository root:

    uv run python model_lab/scripts/split_qwen_dataset.py \
      --input model_lab/data/qwen_skill_train.generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def _stratified_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_role[str(row.get("job_title") or "Unknown")].append(row)

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for role_rows in by_role.values():
        rng.shuffle(role_rows)
        train_end = int(len(role_rows) * train_ratio)
        dev_end = train_end + int(len(role_rows) * dev_ratio)
        if len(role_rows) >= 10:
            train_end = max(1, train_end)
            dev_end = max(train_end + 1, dev_end)
        train.extend(role_rows[:train_end])
        dev.extend(role_rows[train_end:dev_end])
        test.extend(role_rows[dev_end:])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="model_lab/data/qwen_skill_train.generated.jsonl")
    parser.add_argument("--output-dir", default="model_lab/data/eval")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input))
    train, dev, test = _stratified_split(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )
    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "dev.jsonl", dev)
    _write_jsonl(output_dir / "test.jsonl", test)
    manifest = {
        "source": args.input,
        "seed": args.seed,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "test_rows": len(test),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
