# Model Lab: Qwen LoRA Skill Extractor

This folder contains model-training and model-evaluation assets for the AI Job
Analyzer. The main app stays in `src/ai_job_analyzer/`; this folder is for
datasets, LoRA fine-tuning, evaluation, and adapter artifacts.

## Recommended QLoRA Model

The default local fine-tuning target is now:

```text
Qwen/Qwen3-1.7B
```

Use QLoRA for this model unless you have enough VRAM for full-precision LoRA.
If you need a lighter smoke-test model, try:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The Qwen service defaults to Qwen3 1.7B. The main app can either load Qwen
in-process with `LLM_PROVIDER=qwen_lora` or call the separate service with
`LLM_PROVIDER=qwen_service`.

## Install

From the repo root:

```bash
uv sync --extra qwen-lora
```

## Data Format

Training data is JSONL. One row:

```json
{"job_title":"Applied AI Engineer","postings":[{"title":"Applied AI Engineer","company":"Example","location":"Remote","description":"Build RAG systems with PyTorch and evaluation pipelines."}],"output":{"top_skills":[{"name":"RAG","category":"domain","importance":0.9,"evidence":["Build RAG systems"]},{"name":"PyTorch","category":"framework","importance":0.8,"evidence":["PyTorch"]}],"core_responsibilities":["Build applied AI product features."],"nice_to_have":["LoRA fine-tuning"],"summary":"Applied AI roles emphasize RAG, evaluation, and production ML tooling."}}
```

See `data/examples/qwen_skill_train.example.jsonl`.

## Generate Training Labels With A Closed Teacher Model

First collect seed rows with `job_title` and `postings`, without `output`. See
`data/examples/qwen_skill_seed.example.jsonl` for the shape.

```bash
uv run python model_lab/scripts/collect_qwen_seed_postings.py \
  --output model_lab/data/qwen_skill_seed.generated.jsonl \
  --source serpapi \
  --location "United States" \
  --per-role 100
```

The default role catalog covers common software, data, AI, product, design,
security, operations, marketing, sales, and finance roles. You can pass
`--roles path/to/roles.txt` to use one role per line.

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
uv run python model_lab/scripts/generate_qwen_training_data.py \
  --input model_lab/data/qwen_skill_seed.generated.jsonl \
  --output model_lab/data/qwen_skill_train.generated.jsonl \
  --model gpt-5.2
```

`gpt-5.2` is the default teacher model. If your OpenAI account has access and
you want the highest-quality labels over speed/cost, pass `--model gpt-5.2-pro`.

## Fine-Tune

```bash
uv run python model_lab/scripts/finetune_qwen_lora.py \
  --train model_lab/data/qwen_skill_train.generated.jsonl \
  --output model_lab/models/qwen-job-keyword-lora \
  --base-model Qwen/Qwen3-1.7B
```

## Evaluate Base Qwen, GPT-5 Mini, And QLoRA

Create a stable split:

```bash
uv run python model_lab/scripts/split_qwen_dataset.py \
  --input model_lab/data/qwen_skill_train.generated.jsonl
```

Run pure Qwen3 1.7B first. Start the Qwen service with no adapter:

```bash
QWEN_BASE_MODEL=Qwen/Qwen3-1.7B QWEN_ADAPTER_PATH= uv run ai-job-analyzer serve-qwen
```

Then evaluate:

```bash
uv run python model_lab/scripts/evaluate_skill_extractors.py \
  --dataset model_lab/data/eval/test.jsonl \
  --provider qwen_service \
  --run-name qwen3-1_7b-base \
  --qwen-model-name Qwen/Qwen3-1.7B
```

Evaluate GPT-5 mini on the same split:

```bash
uv run python model_lab/scripts/evaluate_skill_extractors.py \
  --dataset model_lab/data/eval/test.jsonl \
  --provider openai \
  --openai-model gpt-5-mini \
  --run-name gpt-5-mini
```

After QLoRA, restart the Qwen service with `QWEN_ADAPTER_PATH` set and run the
same command with `--run-name qwen3-1_7b-qlora`.

Render a visual report:

```bash
uv run python model_lab/scripts/render_eval_report.py \
  model_lab/eval_runs/qwen3-1_7b-base.metrics.json \
  model_lab/eval_runs/gpt-5-mini.metrics.json \
  --output model_lab/eval_runs/report.html
```

The report compares skill precision, recall, F1, and a deterministic
faithfulness proxy against the GPT-teacher labels.

## Run Qwen As A Separate Service

Terminal 1:

```bash
uv run ai-job-analyzer serve-qwen
```

Terminal 2:

```bash
LLM_PROVIDER=qwen_service uv run ai-job-analyzer serve --reload
```

The application calls `QWEN_SERVICE_URL` (default `http://127.0.0.1:8010`) and
does not load `torch`, `transformers`, or the model weights in the app process.

## Use The Adapter

Set `.env`:

```env
QWEN_BASE_MODEL=Qwen/Qwen3-1.7B
QWEN_ADAPTER_PATH=model_lab/models/qwen-job-keyword-lora
QWEN_DEVICE_MAP=auto
QWEN_TORCH_DTYPE=auto
QWEN_MAX_NEW_TOKENS=900
QWEN_SERVICE_URL=http://127.0.0.1:8010
```

Use `LLM_PROVIDER=qwen_service` for the split-service setup. Use
`LLM_PROVIDER=qwen_lora` only if you intentionally want the main app process to
load Qwen directly.

Then run the Qwen service first:

```bash
uv run ai-job-analyzer serve-qwen
```

## AWS / Azure Notes

AWS:

- Start with a GPU host with enough VRAM for Qwen3 1.7B QLoRA experiments.
- Store adapter artifacts in S3 and sync them into `model_lab/models/` during deployment.

Azure:

- Start with NCasT4_v3 or NC A10 v5 family.
- Store adapter artifacts in Azure Blob Storage and sync them into `model_lab/models/`.

For production, you can either run this FastAPI app on the GPU host or deploy a
separate Qwen inference service and keep the main app CPU-only.
