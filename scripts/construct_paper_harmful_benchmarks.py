from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


PAPER_HARMFUL_BENCHMARKS = [
    "harmbench",
    "jailbreakbench",
    "strongreject",
    "advbench",
    "malicious_instruct",
    "do_not_answer",
    "xstest_unsafe",
    "sorrybench",
    "wildjailbreak",
]

HARMFUL_BENCH_SPECS: dict[str, dict[str, list[str]]] = {
    "harmbench": {
        "urls": [
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        ],
        "hf": [
            "hf:swiss-ai/harmbench:DirectRequest:test",
            "hf:swiss-ai/harmbench:HumanJailbreaks:test",
            "hf:mariagrandury/harmbench:DirectRequest:test",
            "hf:cais/HarmBench::test",
            "hf:centerforaisafety/HarmBench::test",
        ],
    },
    "jailbreakbench": {
        "urls": [
            "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
            "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/data/behaviors.csv",
            "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/data/behaviors.jsonl",
        ],
        "hf": [
            "hf:JailbreakBench/JBB-Behaviors:behaviors:train",
            "hf:JailbreakBench/JailbreakBench::test",
            "hf:walledai/JailbreakBench::test",
        ],
    },
    "strongreject": {
        "urls": [
            "https://raw.githubusercontent.com/alexandrasouly/strongreject/refs/heads/main/strongreject_dataset/strongreject_dataset.csv",
            "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
            "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/data/strongreject_dataset.csv",
        ],
        "hf": [
            "hf:walledai/StrongREJECT::train",
            "hf:Machlovi/strongreject-dataset::train",
            "hf:allenai/strongreject::test",
            "hf:StrongREJECT/StrongREJECT::test",
        ],
    },
    "advbench": {
        "urls": [
            "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        ],
        "hf": [
            "hf:walledai/AdvBench::train",
            "hf:JailbreakBench/AdvBench::train",
        ],
    },
    "malicious_instruct": {
        "urls": [
            "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt",
        ],
        "hf": [
            "hf:walledai/MaliciousInstruct::train",
            "hf:TrustAIRLab/in-the-wild-jailbreak-prompts:MaliciousInstruct:train",
        ],
    },
    "do_not_answer": {
        "urls": [
            "https://huggingface.co/datasets/LibrAI/do-not-answer/resolve/main/data_en.csv?download=true",
        ],
        "hf": [
            "hf:LibrAI/do-not-answer::train",
            "hf:allenai/do-not-answer::train",
        ],
    },
    "xstest_unsafe": {
        "urls": [
            "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
        ],
        "hf": [
            "hf:Paul/XSTest::train",
            "hf:natolambert/xstest-v2-copy::train",
            "hf:walledai/XSTest::test",
            "hf:allenai/xstest-response-pairs:unsafe:test",
            "hf:XSTest/XSTest::test",
        ],
    },
    "sorrybench": {
        "hf": [
            "hf:Sorry-Bench/sorry-bench-202406::train",
        ],
    },
    "wildjailbreak": {
        "hf": [
            "hf:allenai/wildjailbreak:train:train",
            "hf:allenai/wildjailbreak::train",
            "hf:walledai/WildJailbreak::train",
        ],
    },
}

PROMPT_KEYS = (
    "instruction",
    "prompt",
    "query",
    "request",
    "behavior",
    "Behavior",
    "goal",
    "Goal",
    "question",
    "Question",
    "forbidden_prompt",
    "jailbreak_prompt",
    "advbench_prompt",
    "text",
    "input",
)
CATEGORY_KEYS = ("category", "Category", "semantic_category", "FunctionalCategory", "source")
TARGET_KEYS = ("target", "Target", "target_str", "target_string")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct TransferAttack CSV files for the paper harmful benchmark "
            "suite used by COMBAT, without importing COMBAT."
        )
    )
    parser.add_argument("--benchmarks", nargs="+", default=list(PAPER_HARMFUL_BENCHMARKS))
    parser.add_argument("--output-dir", type=Path, default=_repo_root() / "data" / "benchmarks")
    parser.add_argument(
        "--benchmark-cache-dir",
        type=Path,
        default=_repo_root() / "data" / "benchmark_cache",
    )
    parser.add_argument("--benchmark-max-examples", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-template",
        choices=("gjo", "gcg_here", "gcg_sure"),
        default="gjo",
    )
    parser.add_argument(
        "--default-target",
        default="Sure, here is a detailed response",
        help=(
            "Fallback target string for benchmarks that only provide an instruction. "
            "TransferAttack requires a target for GCG/GJO loss computation."
        ),
    )
    parser.add_argument("--require-external", action="store_true")
    parser.add_argument(
        "--prefer-local-harmbench",
        action="store_true",
        help=(
            "Use the local data/harmbench_sorted.json file for HarmBench so "
            "TransferAttack targets are preserved. By default this script uses "
            "the COMBAT-style external HarmBench sources for identical sampling."
        ),
    )
    parser.add_argument("--no-combined", action="store_true")
    return parser


def _clean_benchmark_names(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        item = str(name).strip()
        if not item or item.lower() in {"none", "baseline", "baseline_only"}:
            continue
        if item.lower() == "all":
            for bench in PAPER_HARMFUL_BENCHMARKS:
                if bench not in out:
                    out.append(bench)
            continue
        if item not in out:
            out.append(item)
    return out


def _stable_unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        instruction = str(record.get("instruction", "")).strip()
        if not instruction or instruction in seen:
            continue
        seen.add(instruction)
        out.append(dict(record))
    return out


def _sample_records_deterministic(
    records: list[dict[str, Any]],
    *,
    n: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    records = _stable_unique_records(records)
    if n is None or int(n) <= 0 or int(n) >= len(records):
        return records
    rng = random.Random(int(seed))
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[index] for index in sorted(indices[: int(n)])]


def _record_from_prompt(
    prompt: Any,
    *,
    bench_name: str,
    index: int,
    category: Any,
    source: str,
    target: Any = None,
) -> dict[str, Any] | None:
    instruction = str(prompt or "").strip()
    if not instruction:
        return None
    record = {
        "id": f"{bench_name}-{index:04d}",
        "instruction": instruction,
        "category": None if category is None else str(category),
        "benchmark": str(bench_name),
        "source": str(source),
    }
    if target is not None and str(target).strip():
        record["target"] = str(target).strip()
    return record


def _maybe_json_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _row_to_prompt(row: dict[str, Any]) -> tuple[str | None, Any]:
    turns = _maybe_json_list(row.get("turns"))
    if turns is not None:
        for turn in turns:
            if str(turn or "").strip():
                return str(turn).strip(), row.get("Category") or row.get("category")
    if row.get("Goal") is not None and str(row.get("Goal")).strip():
        return str(row["Goal"]).strip(), row.get("Category") or row.get("category")
    if row.get("goal") is not None and str(row.get("goal")).strip():
        return str(row["goal"]).strip(), row.get("Category") or row.get("category")

    prompt: str | None = None
    for key in PROMPT_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            prompt = str(value).strip()
            break

    category = None
    for key in CATEGORY_KEYS:
        if row.get(key) is not None:
            category = row.get(key)
            break
    return prompt, category


def _row_to_target(row: dict[str, Any]) -> str | None:
    for key in TARGET_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _is_xstest_unsafe_row(row: dict[str, Any]) -> bool:
    label = str(row.get("label") or row.get("safety_label") or row.get("classification") or "").lower()
    row_type = str(row.get("type") or row.get("category") or row.get("prompt_type") or "").lower()
    if label in {"unsafe", "harmful", "dangerous", "disallowed"}:
        return True
    if label in {"safe", "benign", "allowed"}:
        return False
    return row_type.startswith("contrast") or "unsafe" in row_type or "harmful" in row_type


def _records_from_rows(
    rows: list[dict[str, Any]],
    *,
    bench_name: str,
    source: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if str(bench_name).lower() == "xstest_unsafe" and not _is_xstest_unsafe_row(row):
            continue
        prompt, category = _row_to_prompt(dict(row))
        record = _record_from_prompt(
            prompt,
            bench_name=bench_name,
            index=index,
            category=category,
            source=source,
            target=_row_to_target(dict(row)),
        )
        if record is not None:
            records.append(record)
    return _stable_unique_records(records)


def _cache_path_for_url(url: str, cache_dir: Path) -> Path:
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path)).suffix or ".txt"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{Path(parsed.path).stem or 'download'}_{digest}{suffix}"


def _download_url(url: str, *, cache_dir: Path, timeout_seconds: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _cache_path_for_url(url, cache_dir)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    request = urllib.request.Request(url, headers={"User-Agent": "TransferAttackBenchmarkLoader/1.0"})
    with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
        output_path.write_bytes(response.read())
    return output_path


def _load_rows_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rows.append(dict(item) if isinstance(item, dict) else {"prompt": item})
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [dict(item) if isinstance(item, dict) else {"prompt": item} for item in data]
        if isinstance(data, dict):
            for key in ("data", "rows", "records", "train", "test", "validation"):
                value = data.get(key)
                if isinstance(value, list):
                    return [
                        dict(item) if isinstance(item, dict) else {"prompt": item}
                        for item in value
                    ]
            return [dict(data)]
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return [{"prompt": line.strip()} for line in handle if line.strip()]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for parquet benchmark files") from exc
        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    raise ValueError(f"unsupported benchmark file format: {path}")


def _parse_hf_spec(spec: str, *, default_split: str = "test") -> tuple[str, str | None, str]:
    body = spec.removeprefix("hf:")
    parts = body.split(":")
    dataset_name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else default_split
    return dataset_name, config, split


def _load_rows_from_hf(spec: str, *, default_split: str = "test") -> tuple[list[dict[str, Any]], str]:
    dataset_name, config, split = _parse_hf_spec(spec, default_split=default_split)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required for hf: benchmark loading") from exc

    load_args: list[str] = [dataset_name]
    if config is not None:
        load_args.append(config)
    try:
        dataset = load_dataset(*load_args, split=split)
    except Exception:
        fallback_split = "validation" if split == "test" else "train"
        dataset = load_dataset(*load_args, split=fallback_split)
        split = fallback_split
    return [dict(row) for row in dataset], f"hf:{dataset_name}:{config or ''}:{split}"


def _load_named_benchmark(
    name: str,
    *,
    cache_dir: Path,
    timeout_seconds: int,
    prefer_local_harmbench: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(name)
    if path.exists():
        rows = _load_rows_from_file(path)
        return _records_from_rows(rows, bench_name=path.stem, source=str(path)), []

    if prefer_local_harmbench and str(name) == "harmbench":
        local_harmbench = _repo_root() / "data" / "harmbench_sorted.json"
        if local_harmbench.exists():
            rows = _load_rows_from_file(local_harmbench)
            return _records_from_rows(
                rows,
                bench_name="harmbench",
                source=str(local_harmbench),
            ), []

    spec = HARMFUL_BENCH_SPECS.get(str(name))
    if spec is None:
        raise KeyError(f"unknown harmful benchmark: {name}")

    errors: list[str] = []
    for url in spec.get("urls", []):
        try:
            file_path = _download_url(url, cache_dir=cache_dir, timeout_seconds=timeout_seconds)
            rows = _load_rows_from_file(file_path)
            records = _records_from_rows(rows, bench_name=name, source=url)
            if records:
                return records, errors
            errors.append(f"url produced no records: {url}")
        except (OSError, urllib.error.URLError, ValueError, RuntimeError) as exc:
            errors.append(f"url failed: {url}: {type(exc).__name__}: {exc}")

    for spec_item in spec.get("hf", []):
        try:
            rows, source = _load_rows_from_hf(spec_item)
            records = _records_from_rows(rows, bench_name=name, source=source)
            if records:
                return records, errors
            errors.append(f"hf produced no records: {spec_item}")
        except Exception as exc:
            errors.append(f"hf failed: {spec_item}: {type(exc).__name__}: {exc}")

    return [], errors


def _strip_sure_prefix(target_str: str) -> str:
    text = str(target_str or "").strip()
    if text.lower().startswith("sure, "):
        text = text[6:]
    return text[:1].upper() + text[1:] if text else "Here is a detailed response"


def _manual_prompt(prompt_template: str, target_str: str, goal: str) -> tuple[str, str]:
    if prompt_template == "gcg_sure":
        return f"{goal}", f" {target_str}"
    target = _strip_sure_prefix(target_str)
    if prompt_template == "gcg_here":
        return f"{goal}", f" {target}"
    if prompt_template == "gjo":
        return (
            f"{goal}\nStart your response with \"{target}:\nStep 1\"",
            f" {target}:\nStep 1",
        )
    raise ValueError(f"wrong prompt template {prompt_template}")


def _transfer_row(
    record: dict[str, Any],
    *,
    prompt_template: str,
    default_target: str,
) -> dict[str, str]:
    question = str(record["instruction"])
    target = str(record.get("target") or default_target)
    goal, target_prefix = _manual_prompt(prompt_template, target, question)
    return {
        "goal": goal,
        "target": target_prefix,
        "question": question,
        "benchmark": str(record.get("benchmark", "")),
        "record_id": str(record.get("id", "")),
        "category": "" if record.get("category") is None else str(record.get("category")),
        "source": str(record.get("source", "")),
        "target_source": "record" if record.get("target") else "default",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "goal",
        "target",
        "question",
        "benchmark",
        "record_id",
        "category",
        "source",
        "target_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    benchmarks = _clean_benchmark_names(list(args.benchmarks))
    output_dir = args.output_dir / args.prompt_template
    cache_dir = args.benchmark_cache_dir

    combined_rows: list[dict[str, str]] = []
    combined_records: list[dict[str, Any]] = []
    summary: list[tuple[str, int, list[str]]] = []

    for offset, benchmark in enumerate(benchmarks):
        benchmark_output_name = Path(benchmark).stem if Path(benchmark).exists() else benchmark
        records, errors = _load_named_benchmark(
            benchmark,
            cache_dir=cache_dir,
            timeout_seconds=int(args.timeout_seconds),
            prefer_local_harmbench=bool(args.prefer_local_harmbench),
        )
        sampled = _sample_records_deterministic(
            records,
            n=int(args.benchmark_max_examples),
            seed=int(args.seed) + 1009 + offset,
        )
        if args.require_external and not sampled:
            raise RuntimeError(f"failed to load benchmark {benchmark}: {errors}")

        rows = [
            _transfer_row(
                record,
                prompt_template=str(args.prompt_template),
                default_target=str(args.default_target),
            )
            for record in sampled
        ]
        _write_csv(output_dir / f"{benchmark_output_name}.csv", rows)
        _write_jsonl(output_dir / f"{benchmark_output_name}.jsonl", sampled)

        combined_rows.extend(rows)
        combined_records.extend(sampled)
        summary.append((benchmark_output_name, len(rows), errors))

    if not args.no_combined:
        _write_csv(output_dir / "paper_harmful.csv", combined_rows)
        _write_jsonl(output_dir / "paper_harmful.jsonl", combined_records)

    for benchmark, count, errors in summary:
        print(f"[bench] {benchmark}: wrote {count} rows")
        if errors:
            print(f"[bench] {benchmark}: fallback/errors: {errors}", file=sys.stderr)
    print(f"[done] output_dir={output_dir}")


if __name__ == "__main__":
    main()
