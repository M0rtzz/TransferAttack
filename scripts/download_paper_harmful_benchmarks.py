from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from construct_paper_harmful_benchmarks import (
    _records_from_rows,
    _write_csv,
    _write_jsonl,
    _transfer_row,
)


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

# This is copied from COMBAT/scripts/download_datasets.py, harmful section only.
DIRECT_URL_SOURCES: dict[str, list[str]] = {
    "harmbench": [
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
    ],
    "jailbreakbench": [
        "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv?download=true",
        "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/data/behaviors.csv",
    ],
    "strongreject": [
        "https://raw.githubusercontent.com/alexandrasouly/strongreject/refs/heads/main/strongreject_dataset/strongreject_dataset.csv",
        "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
    ],
    "advbench": [
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    ],
    "malicious_instruct": [
        "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt",
    ],
    "do_not_answer": [
        "https://huggingface.co/datasets/LibrAI/do-not-answer/resolve/main/data_en.csv?download=true",
    ],
    "xstest_unsafe": [
        "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    ],
    "sorrybench": [],
    "wildjailbreak": [],
}

# Fallback sources mirror COMBAT's run_refusal_direction_ablation.py. The
# downloader reaches these through COMBAT's _load_named_benchmark when direct
# URLs are unavailable or fail.
FALLBACK_SOURCES: dict[str, dict[str, list[str]]] = {
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and materialize the COMBAT paper harmful benchmark suite "
            "without importing COMBAT."
        )
    )
    parser.add_argument(
        "--harmful-benchmarks",
        nargs="+",
        default=None,
        help="Harmful benchmarks to download. Defaults to the COMBAT paper harmful suite.",
    )
    parser.add_argument(
        "--paper-full-suite",
        action="store_true",
        help="Use the COMBAT paper harmful suite.",
    )
    parser.add_argument(
        "--benchmark-cache-dir",
        type=Path,
        default=_repo_root() / "outputs" / "benchmark_cache",
    )
    parser.add_argument(
        "--write-jsonl-dir",
        type=Path,
        default=_repo_root() / "outputs" / "downloaded_benchmarks",
    )
    parser.add_argument(
        "--write-transferattack-dir",
        type=Path,
        default=_repo_root() / "outputs" / "downloaded_transferattack_benchmarks",
        help="Directory for TransferAttack goal/target/question CSV files.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-hub-download-timeout", type=int, default=300)
    parser.add_argument("--hf-hub-etag-timeout", type=int, default=60)
    parser.add_argument("--disable-hf-xet", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--prompt-template",
        choices=("gjo", "gcg_here", "gcg_sure"),
        default="gjo",
    )
    parser.add_argument("--default-target", default="Sure, here is a detailed response")
    return parser


def _configure_hf_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(int(args.hf_hub_download_timeout)))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(int(args.hf_hub_etag_timeout)))
    if args.hf_cache_dir is not None:
        hf_cache_dir = Path(args.hf_cache_dir)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache_dir))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir / "datasets"))
    if args.disable_hf_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


def _clean_names(names: list[str] | None) -> list[str]:
    if names is None:
        return list(PAPER_HARMFUL_BENCHMARKS)
    out: list[str] = []
    for name in names:
        item = str(name).strip()
        if not item or item.lower() in {"none", "baseline", "baseline_only"}:
            continue
        if item.lower() == "all":
            for bench in PAPER_HARMFUL_BENCHMARKS:
                if bench not in out:
                    out.append(bench)
        elif item not in out:
            out.append(item)
    return out


def _safe_cache_name(url: str) -> str:
    base = url.split("?", 1)[0]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    suffix = Path(base).suffix
    safe = safe[-180:]
    if suffix and not safe.endswith(suffix):
        safe = f"{safe}{suffix}"
    return safe or "downloaded_dataset"


def _download_file(url: str, *, cache_dir: Path, timeout_seconds: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _safe_cache_name(url)
    if path.exists() and path.stat().st_size > 0:
        return path
    request = urllib.request.Request(url, headers={"User-Agent": "COMBAT-dataset-downloader"})
    with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
        path.write_bytes(response.read())
    return path


def _rows_from_downloaded_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets package is required for parquet benchmark loading") from exc
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        return [dict(row) for row in dataset]
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if suffix == ".jsonl":
        return [json.loads(line) for line in stripped.splitlines() if line.strip()]
    if suffix == ".csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if suffix == ".txt":
        return [{"prompt": line.strip()} for line in stripped.splitlines() if line.strip()]
    data = json.loads(stripped)
    if isinstance(data, list):
        return [dict(row) if isinstance(row, dict) else {"instruction": row} for row in data]
    if isinstance(data, dict):
        for key in ("data", "examples", "behaviors", "prompts", "train", "validation", "test"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(row) if isinstance(row, dict) else {"instruction": row} for row in value]
    return []


def _parse_hf_spec(spec: str, *, default_split: str = "test") -> tuple[str, str | None, str]:
    body = spec.removeprefix("hf:")
    parts = body.split(":")
    dataset_name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else default_split
    return dataset_name, config, split


def _rows_from_hf(spec: str, *, default_split: str = "test") -> tuple[list[dict[str, Any]], str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets package is required for hf: benchmark loading") from exc

    dataset_name, config, split = _parse_hf_spec(spec, default_split=default_split)
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


def _records_from_url(
    *,
    name: str,
    url: str,
    cache_dir: Path,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], Path]:
    path = _download_file(url, cache_dir=cache_dir, timeout_seconds=timeout_seconds)
    rows = _rows_from_downloaded_file(path)
    return _records_from_rows(rows, bench_name=name, source=url), path


def _records_from_hf_spec(*, name: str, spec: str) -> list[dict[str, Any]]:
    rows, source = _rows_from_hf(spec)
    return _records_from_rows(rows, bench_name=name, source=source)


def _download_one(
    *,
    name: str,
    cache_dir: Path,
    timeout_seconds: int,
    write_jsonl_dir: Path,
    write_transferattack_dir: Path,
    prompt_template: str,
    default_target: str,
) -> dict[str, Any]:
    print(f"[download] harmful: {name}", flush=True)
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    selected_source: str | None = None
    raw_cache_path: str | None = None

    for url in DIRECT_URL_SOURCES.get(name, []):
        try:
            records, path = _records_from_url(
                name=name,
                url=url,
                cache_dir=cache_dir / "direct_urls",
                timeout_seconds=timeout_seconds,
            )
            if records:
                selected_source = url
                raw_cache_path = str(path)
                break
            errors.append(f"direct url produced no records: {url}")
        except (OSError, urllib.error.URLError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"direct url failed: {url}: {type(exc).__name__}: {exc}")

    if not records:
        fallback = FALLBACK_SOURCES.get(name, {})
        for url in fallback.get("urls", []):
            try:
                records, path = _records_from_url(
                    name=name,
                    url=url,
                    cache_dir=cache_dir,
                    timeout_seconds=timeout_seconds,
                )
                if records:
                    selected_source = url
                    raw_cache_path = str(path)
                    break
                errors.append(f"fallback url produced no records: {url}")
            except (OSError, urllib.error.URLError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"fallback url failed: {url}: {type(exc).__name__}: {exc}")

    if not records:
        fallback = FALLBACK_SOURCES.get(name, {})
        for hf_spec in fallback.get("hf", []):
            try:
                records = _records_from_hf_spec(name=name, spec=hf_spec)
                if records:
                    selected_source = hf_spec
                    break
                errors.append(f"hf produced no records: {hf_spec}")
            except Exception as exc:
                errors.append(f"hf failed: {hf_spec}: {type(exc).__name__}: {exc}")

    result: dict[str, Any] = {
        "kind": "harmful",
        "name": name,
        "count": len(records),
        "selected_source": selected_source,
        "raw_cache_path": raw_cache_path,
        "jsonl_path": None,
        "transferattack_csv_path": None,
        "errors": errors,
        "ok": bool(records),
    }
    if not records:
        print(f"[failed] harmful: {name}; errors={errors}", flush=True)
        return result

    jsonl_path = write_jsonl_dir / "harmful" / f"{name}.jsonl"
    _write_jsonl(jsonl_path, records)
    transfer_rows = [
        _transfer_row(record, prompt_template=prompt_template, default_target=default_target)
        for record in records
    ]
    csv_path = write_transferattack_dir / prompt_template / f"{name}.csv"
    _write_csv(csv_path, transfer_rows)

    result["jsonl_path"] = str(jsonl_path)
    result["transferattack_csv_path"] = str(csv_path)
    print(
        f"[ok] harmful: {name}; records={len(records)}; source={selected_source}; "
        f"jsonl={jsonl_path}; csv={csv_path}",
        flush=True,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    _configure_hf_env(args)
    benchmarks = _clean_names(args.harmful_benchmarks)
    if args.paper_full_suite:
        benchmarks = list(PAPER_HARMFUL_BENCHMARKS)

    args.benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
    args.write_jsonl_dir.mkdir(parents=True, exist_ok=True)
    args.write_transferattack_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _download_one(
            name=name,
            cache_dir=args.benchmark_cache_dir,
            timeout_seconds=int(args.timeout_seconds),
            write_jsonl_dir=args.write_jsonl_dir,
            write_transferattack_dir=args.write_transferattack_dir,
            prompt_template=str(args.prompt_template),
            default_target=str(args.default_target),
        )
        for name in benchmarks
    ]

    manifest = {
        "paper_harmful_benchmarks": PAPER_HARMFUL_BENCHMARKS,
        "requested_harmful_benchmarks": benchmarks,
        "direct_url_sources": DIRECT_URL_SOURCES,
        "fallback_sources": FALLBACK_SOURCES,
        "results": results,
        "benchmark_cache_dir": str(args.benchmark_cache_dir),
        "write_jsonl_dir": str(args.write_jsonl_dir),
        "write_transferattack_dir": str(args.write_transferattack_dir),
        "hf_env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"),
            "HF_HUB_ETAG_TIMEOUT": os.environ.get("HF_HUB_ETAG_TIMEOUT"),
            "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET"),
        },
    }
    manifest_path = args.write_jsonl_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [row for row in results if not row.get("ok")]
    print(f"[summary] ok={len(results) - len(failed)} failed={len(failed)} manifest={manifest_path}")
    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
