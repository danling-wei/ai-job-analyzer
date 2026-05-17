"""Fine-tune a Qwen instruct model with QLoRA/LoRA for skill extraction.

Run from the repository root:

    uv run python model_lab/scripts/finetune_qwen_lora.py \
      --train model_lab/data/examples/qwen_skill_train.example.jsonl \
      --output model_lab/models/qwen-job-keyword-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You extract a job-seeker skill-gap checklist from job postings. "
    "Return only JSON with top_skills, core_responsibilities, nice_to_have, and summary. "
    "Top skills must be concrete learnable skills/tools/platforms/methods, not generic labels."
)


def _format_example(row: dict[str, Any]) -> dict[str, str]:
    postings = row.get("postings")
    if not isinstance(postings, list):
        raise ValueError("Each row must include a postings list.")
    chunks: list[str] = []
    for i, posting in enumerate(postings, 1):
        if not isinstance(posting, dict):
            continue
        chunks.append(
            "\n".join(
                [
                    f"[{i}] {posting.get('title', 'Untitled')}",
                    f"Company: {posting.get('company', 'Unknown')}",
                    f"Location: {posting.get('location', 'n/a')}",
                    str(posting.get("description", "")),
                ]
            )
        )
    output = row.get("output")
    if not isinstance(output, dict):
        raise ValueError("Each row must include an output object.")
    user = f"Target role: {row.get('job_title', 'Unknown')}\n\nPostings:\n\n{chr(10).join(chunks)}"
    assistant = json.dumps(output, ensure_ascii=False)
    return {"system": SYSTEM_PROMPT, "user": user, "assistant": assistant}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training JSONL path.")
    parser.add_argument("--output", required=True, help="LoRA adapter output directory.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--qlora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model in 4-bit NF4 and train LoRA adapters.",
    )
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    rows = [
        _format_example(json.loads(line))
        for line in Path(args.train).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def to_text(row: dict[str, str]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(to_text)
    model_kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
