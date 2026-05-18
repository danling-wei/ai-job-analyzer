# QLoRA Training Decisions

Last updated: 2026-05-18

## Current Run: Qwen3-1.7B Skill Extractor

Planned adapter output:

`model_lab/models/qwen3-1_7b-skill-extractor-qlora-r16a32`

Training data:

- Use `model_lab/data/eval/train.jsonl` for fine-tuning.
- Keep `model_lab/data/eval/test.jsonl` held out for final comparison against
  `qwen3-1_7b-base` and `gpt-5-mini`.
- Rationale: this avoids train/test leakage and makes the QLoRA improvement
  defensible in interviews.

Base model:

- `Qwen/Qwen3-1.7B`
- Rationale: 8B was too slow for practical local evaluation on the current
  workstation. 1.7B lets the project demonstrate an end-to-end, reproducible,
  locally deployable QLoRA workflow.

QLoRA / LoRA choices:

- Quantization: 4-bit NF4 with double quantization.
- LoRA rank `r=16`.
- LoRA alpha `32`.
- LoRA dropout `0.05`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`.

Rationale:

- `r=16` is a conservative first run for a 1.7B model: enough adapter capacity
  to learn structured JSON skill extraction and domain vocabulary without
  making training unnecessarily heavy.
- `alpha=32` gives an alpha/rank ratio of 2, a common stable starting point
  for QLoRA that keeps adapter updates meaningful but not too aggressive.
- `dropout=0.05` adds light regularization because the generated labels are
  teacher-produced and may contain style bias; higher dropout would risk
  underfitting the desired JSON schema.
- Including attention and MLP projection modules gives the adapter enough
  control over both instruction following and domain-specific token selection.

Optimization choices:

- Epochs: `2.0`.
- Learning rate: `2e-4`.
- Per-device batch size: `1`.
- Gradient accumulation: `8`.
- Max sequence length: `4096`.

Rationale:

- `2` epochs is enough for a first supervised adapter on about two thousand
  generated examples while limiting overfitting.
- `2e-4` is a standard QLoRA learning rate for small adapter training.
- Effective batch size is `8`, which keeps memory use manageable while making
  optimization less noisy than pure batch size 1.
- `4096` tokens preserves most job descriptions without forcing excessive
  memory use.

Evaluation plan:

1. Train the adapter on `train.jsonl`.
2. Restart Qwen service with `QWEN_ADAPTER_PATH` set to the adapter directory.
3. Run `evaluate_skill_extractors.py` on `test.jsonl` with run name
   `qwen3-1_7b-qlora-r16a32`.
4. Render a three-way report comparing:
   - `qwen3-1_7b-base`
   - `gpt-5-mini`
   - `qwen3-1_7b-qlora-r16a32`


## Correction: Strict Evaluation and Compact Retraining

After the first adapter finished training, a strict smoke test failed. The
adapter produced useful `top_skills`, but the JSON was incomplete/malformed and
did not reliably include `summary` and `core_responsibilities`.

Decision:

- Do not relax evaluation for the LoRA run.
- Keep `summary` required, just like the GPT-5 mini and Qwen3-1.7B base
  baselines.
- Do not use JSON repair or empty-summary fallback in the final comparison.
- Retrain with format-aligned compact targets instead.

Rationale:

- The first training target was too verbose for the 900-token inference budget:
  teacher labels can contain up to 15 skills and multiple evidence snippets per
  skill, so the adapter learned to emit long JSON and sometimes failed before
  completing all required fields.
- The compact retraining target keeps the same semantic task but caps output at
  10 skills and 1 short evidence snippet per skill, matching the inference
  prompt used by the Qwen service.
- This is not a scoring advantage for LoRA; it is a training-format correction
  so the adapter is judged by the same strict schema as the previous baselines.

Second adapter output:

`model_lab/models/qwen3-1_7b-skill-extractor-qlora-r16a32-compact`

Result:

- Strict smoke test passed on 1 held-out sample with no schema failure.
- Full strict evaluation on `model_lab/data/eval/test.jsonl` completed with:
  - successful samples: `275 / 299`
  - failed samples: `24`
  - failure rate: `0.080`
  - skill precision: `0.409`
  - skill recall: `0.235`
  - skill F1: `0.299`
  - faithfulness proxy: `0.999`

Comparison:

- The compact adapter improved over the base Qwen3-1.7B run on skill precision
  (`0.409` vs `0.324`), recall (`0.235` vs `0.171`), and F1 (`0.299` vs
  `0.224`).
- It did not beat `gpt-5-mini` on recall or F1 (`0.334` F1), and it had a much
  higher strict-schema failure rate (`24 / 299` vs `0 / 299`).
- The main remaining issue is not skill faithfulness; it is output reliability
  for the full four-field schema.

Decision for next experiment:

- Treat this as a useful but not final adapter.
- If the project needs a more convincing interview demo, train a narrower
  skills-only QLoRA adapter next.
- Keep summary and responsibility synthesis in the product pipeline, but route
  that heavier synthesis step to a stronger finalizer rather than forcing the
  1.7B adapter to own the entire structured output.


## Global Skill Canonicalization

Added a shared canonicalization pass:

`model_lab/scripts/canonicalize_skill_names.py`

Artifacts:

- Mapping:
  `model_lab/data/skill_canonicalization/skill_canonicalization.mapping.json`
- Skill occurrence inventory:
  `model_lab/data/skill_canonicalization/skill_occurrences.inventory.jsonl`
- Canonicalized train/dev/test copies:
  `model_lab/data/canonicalized/eval/`
- Canonicalized prediction copies and metrics:
  `model_lab/eval_runs/canonicalized/`

Decision:

- Use one global mapping for training labels and all model predictions.
- Preserve aliases in each canonicalized skill so the product can show clean
  canonical skill names without losing original wording.
- Keep this first pass conservative and deterministic. It merges obvious
  aliases and naming variants, while avoiding known acronym collisions such as
  Google Cloud Platform (GCP) vs Good Clinical Practice (GCP), and React vs
  ReAct.

Result:

- `gpt-5-mini` strict-match F1 improved from `0.334` to `0.380`.
- `qwen3-1_7b-base` strict-match F1 improved from `0.224` to `0.245`.
- `qwen3-1_7b-qlora-r16a32-compact` strict-match F1 improved from `0.299`
  to `0.320`.

Interpretation:

- A significant portion of the low exact-match score was caused by duplicate
  skill naming and teacher/prediction wording differences, not unsupported
  predictions.
- The next stronger version should use an LLM-assisted review step on top of
  the saved mapping, but the deterministic mapping is already useful for the
  Job Analyzer application because it stabilizes skill frequency aggregation.


## Semantic Skill Matching Evaluation

Added cached LLM-as-judge evaluation:

`model_lab/scripts/evaluate_semantic_skill_matches.py`

Artifacts:

- Judgment cache:
  `model_lab/data/skill_canonicalization/semantic_match_judgments.jsonl`
- Semantic metrics:
  `model_lab/eval_runs/semantic/*.semantic.metrics.json`
- Semantic visual report:
  `model_lab/eval_runs/semantic/report.html`

Method:

- Run after global canonicalization.
- Count exact canonical skill-name matches first.
- Generate only plausible unmatched prediction/gold skill pairs.
- Ask `gpt-5-mini` to judge whether each pair is an alias, the same skill, or
  a reasonable coverage match.
- Cache all pair judgments so the same mapping is reused across Qwen base,
  GPT-5 mini, and QLoRA evaluations.

Result:

| Run | Semantic precision | Semantic recall | Semantic F1 | Exact F1 before semantic judge |
| --- | ---: | ---: | ---: | ---: |
| `qwen3-1_7b-base-semantic` | `0.449` | `0.236` | `0.310` | `0.245` |
| `gpt-5-mini-semantic` | `0.559` | `0.587` | `0.572` | `0.380` |
| `qwen3-1_7b-qlora-r16a32-compact-semantic` | `0.538` | `0.309` | `0.392` | `0.320` |

Interpretation:

- GPT-5 mini's low exact-match score was mostly an evaluation artifact from
  skill naming differences. Semantic F1 rises to `0.572`.
- Qwen3-1.7B base still has a real recall problem. Semantic matching helps, but
  it remains far behind GPT-5 mini.
- The compact QLoRA adapter improves over base Qwen under both exact and
  semantic evaluation, but it still has lower recall than GPT-5 mini and a
  strict-schema failure rate of `24 / 299`.


## Skills-Only Qwen Adapter

Current run:

`model_lab/models/qwen3-1_7b-skill-extractor-qlora-r16a32-skills-only`

Decision:

- Simplify the small Qwen adapter task to skills-only extraction.
- Require only one top-level JSON key during Qwen inference: `top_skills`.
- Move `summary`, `core_responsibilities`, and final `nice_to_have`
  synthesis to the low-volume finalizer stage.
- Train on the canonicalized training split:
  `model_lab/data/canonicalized/eval/train.jsonl`.

Rationale:

- The previous compact adapter showed that the 1.7B model can extract faithful
  skills, but the full four-field output is too brittle for strict production
  evaluation.
- Skills are the high-volume, per-JD task where local Qwen saves cost and
  latency. Summary/responsibility synthesis is a lower-volume aggregation task
  and is a better fit for the configured finalizer.
- Training on canonicalized labels teaches the adapter stable skill names
  rather than forcing downstream code to repair as many naming variants.
- The hyperparameters stay the same as the compact run (`r=16`, `alpha=32`,
  dropout `0.05`, learning rate `2e-4`, 2 epochs) so the experiment isolates
  the task simplification instead of mixing in a hyperparameter search.
- `max_skills=20` is used to improve recall. Evidence is capped at one short
  snippet per skill so outputs stay compact and JSON completion is more likely.

Business pipeline decision:

- With `QWEN_EXTRACTION_TASK=skills_only`, Qwen extracts per-posting skills.
- The application then merges skill candidates, applies final skill
  canonicalization, and asks the finalizer to generate summary/responsibility
  fields from aggregated skills plus compact responsibility/nice-to-have
  candidates extracted from the postings.
- This keeps the app output complete without sending all raw job descriptions
  to the finalizer for large batches.
