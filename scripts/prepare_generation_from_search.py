from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from utils import build_chat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build generation prompts from a TransferAttack search log and benchmark CSV."
    )
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--suffix-file", type=Path, default=None)
    parser.add_argument("--model-name", default="llama31")
    parser.add_argument("--control-placement", choices=("prefix", "suffix"), default="prefix")
    parser.add_argument("--control-index", type=int, default=-1)
    return parser


def _extract_controls(log_text: str) -> list[str]:
    controls = re.findall(r"control='(.*?)===========================.*?", log_text, re.DOTALL)
    out: list[str] = []
    for control in controls:
        item = control
        if item.endswith("\\n'"):
            item = item[:-3]
        if item.endswith("'\n"):
            item = item[:-2]
        item = item.strip()
        if item:
            out.append(item)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _combine(control: str, goal: str, placement: str) -> str:
    if placement == "prefix":
        return f"{control} {goal}".strip()
    return f"{goal} {control}".strip()


def main() -> None:
    args = build_parser().parse_args()
    log_text = args.log_file.read_text(encoding="utf-8", errors="replace")
    controls = _extract_controls(log_text)
    if not controls:
        raise ValueError(f"no control prompt found in {args.log_file}")
    control = controls[int(args.control_index)]

    rows = _read_csv(args.benchmark_csv)
    prompts: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        goal = str(row.get("goal") or row.get("question") or "").strip()
        if not goal:
            continue
        combined = _combine(control, goal, str(args.control_placement))
        prompts.append(
            {
                "id": idx,
                "prompt": build_chat(combined, str(args.model_name)),
                "question_idx": idx,
                "question": str(row.get("question", "")),
                "goal": goal,
                "target": str(row.get("target", "")),
                "benchmark": str(row.get("benchmark", "")),
                "record_id": str(row.get("record_id", "")),
                "category": str(row.get("category", "")),
                "source": str(row.get("source", "")),
            }
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.suffix_file is not None:
        args.suffix_file.parent.mkdir(parents=True, exist_ok=True)
        args.suffix_file.write_text(
            json.dumps(
                {
                    "control": control,
                    "control_index": int(args.control_index),
                    "control_placement": str(args.control_placement),
                    "benchmark_csv": str(args.benchmark_csv),
                    "log_file": str(args.log_file),
                    "num_prompts": len(prompts),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"[prepare] wrote {len(prompts)} prompts to {args.output_file}")


if __name__ == "__main__":
    main()
