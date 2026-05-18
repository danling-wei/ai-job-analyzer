"""Render a self-contained HTML report for model extraction eval runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bar(value: float, *, color: str) -> str:
    width = max(0.0, min(100.0, value * 100))
    return (
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>'
        "</div>"
    )


def _load_metrics(paths: list[str]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def _metric_cards(metrics: list[dict[str, Any]]) -> str:
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    blocks = []
    for index, metric in enumerate(metrics):
        color = colors[index % len(colors)]
        blocks.append(
            f"""
            <section class="card">
              <h2>{html.escape(metric["run_name"])}</h2>
              <p class="meta">{html.escape(metric.get("model", ""))} · {metric["samples"]}/{metric.get("requested_samples", metric["samples"])} samples</p>
              <div class="metric"><span>Skill F1</span><strong>{_pct(metric["skill_f1"])}</strong></div>
              {_bar(metric["skill_f1"], color=color)}
              <div class="metric"><span>Precision</span><strong>{_pct(metric["skill_precision"])}</strong></div>
              {_bar(metric["skill_precision"], color=color)}
              <div class="metric"><span>Recall</span><strong>{_pct(metric["skill_recall"])}</strong></div>
              {_bar(metric["skill_recall"], color=color)}
              <div class="metric"><span>Faithfulness Proxy</span><strong>{_pct(metric["faithfulness_proxy"])}</strong></div>
              {_bar(metric["faithfulness_proxy"], color=color)}
              <div class="metric"><span>Failure Rate</span><strong>{_pct(metric.get("failure_rate", 0.0))}</strong></div>
              {_bar(metric.get("failure_rate", 0.0), color="#991b1b")}
            </section>
            """
        )
    return "\n".join(blocks)


def _comparison_table(metrics: list[dict[str, Any]]) -> str:
    rows = []
    for metric in metrics:
        rows.append(
            "<tr>"
            f"<td>{html.escape(metric['run_name'])}</td>"
            f"<td>{html.escape(metric.get('model', ''))}</td>"
            f"<td>{metric['samples']} / {metric.get('requested_samples', metric['samples'])}</td>"
            f"<td>{_pct(metric['skill_precision'])}</td>"
            f"<td>{_pct(metric['skill_recall'])}</td>"
            f"<td>{_pct(metric['skill_f1'])}</td>"
            f"<td>{_pct(metric['faithfulness_proxy'])}</td>"
            f"<td>{_pct(metric.get('failure_rate', 0.0))}</td>"
            f"<td>{metric['matched_skills']} / {metric['gold_skills']}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _role_table(metrics: list[dict[str, Any]]) -> str:
    rows = []
    for metric in metrics:
        for role in metric.get("per_role", []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(metric['run_name'])}</td>"
                f"<td>{html.escape(role['job_title'])}</td>"
                f"<td>{role['samples']}</td>"
                f"<td>{_pct(role['skill_f1'])}</td>"
                f"<td>{_pct(role['skill_precision'])}</td>"
                f"<td>{_pct(role['skill_recall'])}</td>"
                f"<td>{_pct(role['faithfulness_proxy'])}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def render(metrics: list[dict[str, Any]]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Skill Extraction Evaluation</title>
  <style>
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #f8fafc; color: #111827; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 0 0 6px; }}
    p {{ color: #475569; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 24px 0; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }}
    .meta {{ margin: 0 0 16px; font-size: 13px; }}
    .metric {{ display: flex; justify-content: space-between; gap: 12px; margin-top: 12px; font-size: 14px; }}
    .bar-track {{ height: 9px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-top: 6px; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin: 16px 0 28px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }}
    th {{ background: #f1f5f9; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #475569; }}
    tr:last-child td {{ border-bottom: 0; }}
    .note {{ font-size: 13px; color: #64748b; }}
  </style>
</head>
<body>
<main>
  <h1>Qwen Skill Extraction Evaluation</h1>
  <p>Compares model outputs against GPT-teacher labels. Faithfulness proxy counts predicted skills supported by the posting text or model evidence.</p>
  <section class="grid">
    {_metric_cards(metrics)}
  </section>
  <h2>Run Comparison</h2>
  <table>
    <thead><tr><th>Run</th><th>Model</th><th>Samples</th><th>Precision</th><th>Recall</th><th>F1</th><th>Faithfulness</th><th>Failures</th><th>Matched / Gold</th></tr></thead>
    <tbody>{_comparison_table(metrics)}</tbody>
  </table>
  <h2>By Role</h2>
  <table>
    <thead><tr><th>Run</th><th>Role</th><th>Samples</th><th>F1</th><th>Precision</th><th>Recall</th><th>Faithfulness</th></tr></thead>
    <tbody>{_role_table(metrics)}</tbody>
  </table>
  <p class="note">Use the same test split for Qwen3-1.7B base, GPT-5 mini, and later Qwen3-1.7B QLoRA to make the comparison defensible.</p>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", help="One or more *.metrics.json files.")
    parser.add_argument("--output", default="model_lab/eval_runs/report.html")
    args = parser.parse_args()

    metrics = _load_metrics(args.metrics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(metrics), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
