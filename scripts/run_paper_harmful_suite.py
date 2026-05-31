from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from construct_paper_harmful_benchmarks import PAPER_HARMFUL_BENCHMARKS


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return _repo_root() / "outputs" / f"paper_harmful_mistral_v03_to_llama31_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end TransferAttack suite for COMBAT-style paper harmful "
            "benchmarks: construct data, search suffixes, generate target outputs, "
            "and compute ASR."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--benchmarks", nargs="+", default=list(PAPER_HARMFUL_BENCHMARKS))
    parser.add_argument("--benchmark-max-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--prompt-template", choices=("gjo", "gcg_here", "gcg_sure"), default="gjo")
    parser.add_argument("--prefer-local-harmbench", action="store_true")

    parser.add_argument("--config-name", default="mistral_v03_to_llama31")
    parser.add_argument("--source-model-path", type=Path, default=_repo_root() / "Models" / "Mistral-7B-Instruct-v0.3")
    parser.add_argument("--target-model-path", type=Path, default=_repo_root() / "Models" / "Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target-tokenizer-path", type=Path, default=None)
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument(
        "--single-gpu-staged",
        action="store_true",
        help="Search with source model only, then load target/scorer in later processes.",
    )

    parser.add_argument("--control-len", type=int, default=100)
    parser.add_argument("--token-num", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--test-steps", type=int, default=5)
    parser.add_argument("--search-batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--system-message", default="False")

    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--asr-scorer", choices=("prefix", "harmbench"), default="harmbench")
    parser.add_argument("--asr-batch-size", type=int, default=4)
    parser.add_argument("--asr-device", default="cuda:0")
    parser.add_argument("--classifier-model-path", default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--classifier-tokenizer-path", default=None)

    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _expand_benchmarks(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if item == "all":
            for bench in PAPER_HARMFUL_BENCHMARKS:
                if bench not in out:
                    out.append(bench)
        elif item not in out:
            out.append(item)
    return out


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    printable = " ".join(cmd)
    print(f"[run] cwd={cwd} {printable}")
    if dry_run:
        return
    if log_file is None:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _row_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _write_run_config(args: argparse.Namespace, benchmarks: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    data["benchmarks"] = benchmarks
    (args.output_dir / "run_config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _construct_benchmarks(args: argparse.Namespace, benchmarks: list[str]) -> None:
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "construct_paper_harmful_benchmarks.py"),
        "--benchmarks",
        *benchmarks,
        "--output-dir",
        str(args.output_dir / "benchmarks"),
        "--benchmark-cache-dir",
        str(args.output_dir / "benchmark_cache"),
        "--benchmark-max-examples",
        str(int(args.benchmark_max_examples)),
        "--timeout-seconds",
        str(int(args.timeout_seconds)),
        "--seed",
        str(int(args.seed)),
        "--prompt-template",
        str(args.prompt_template),
    ]
    if args.prefer_local_harmbench:
        cmd.append("--prefer-local-harmbench")
    _run(cmd, cwd=_repo_root(), dry_run=bool(args.dry_run))


def _search(args: argparse.Namespace, benchmark: str, csv_path: Path, count: int) -> Path:
    search_dir = args.output_dir / "search" / benchmark
    log_file = search_dir / "search.log"
    result_prefix = search_dir / f"transfer_{args.config_name}_{benchmark}"
    control_init = "! " * int(args.control_len)
    env = dict(os.environ)
    env.update(
        {
            "SOURCE_MODEL_PATH": str(args.source_model_path),
            "TARGET_MODEL_PATH": str(args.target_model_path),
            "SOURCE_DEVICE": str(args.source_device),
            "TARGET_DEVICE": str(args.target_device),
            "WANDB_MODE": env.get("WANDB_MODE", "disabled"),
        }
    )
    if args.single_gpu_staged:
        env["SINGLE_GPU_SEARCH"] = "1"
    cmd = [
        sys.executable,
        "-u",
        "./main.py",
        f"--config=./configs/transfer_{args.config_name}.py",
        "--config.attack=gcg",
        f"--config.train_data={csv_path}",
        f"--config.result_prefix={result_prefix}",
        "--config.progressive_goals=True",
        "--config.stop_on_success=False",
        "--config.num_train_models=1",
        "--config.allow_non_ascii=False",
        f"--config.n_train_data={count}",
        f"--config.n_test_data={count}",
        f"--config.n_steps={int(args.steps)}",
        f"--config.test_steps={int(args.test_steps)}",
        f"--config.batch_size={int(args.search_batch_size)}",
        f"--config.topk={int(args.topk)}",
        "--config.ce_prefix=True",
        f"--config.system_message={args.system_message}",
        f"--config.control_init={control_init}",
        f"--config.token_num={int(args.token_num)}",
    ]
    _run(cmd, cwd=_repo_root() / "scripts", log_file=log_file, env=env, dry_run=bool(args.dry_run))
    return log_file


def _prepare_generation(args: argparse.Namespace, benchmark: str, csv_path: Path, log_file: Path) -> Path:
    prompts_file = args.output_dir / "prompts" / f"{benchmark}.json"
    suffix_file = args.output_dir / "suffixes" / f"{benchmark}.json"
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts" / "prepare_generation_from_search.py"),
        "--benchmark-csv",
        str(csv_path),
        "--log-file",
        str(log_file),
        "--output-file",
        str(prompts_file),
        "--suffix-file",
        str(suffix_file),
        "--model-name",
        "llama31",
        "--control-placement",
        "prefix",
    ]
    _run(cmd, cwd=_repo_root() / "scripts", dry_run=bool(args.dry_run))
    return prompts_file


def _generate(args: argparse.Namespace, benchmark: str, prompts_file: Path) -> Path:
    generation_file = args.output_dir / "generations" / f"{benchmark}.json"
    tokenizer_path = args.target_tokenizer_path or args.target_model_path
    cmd = [
        sys.executable,
        str(_repo_root() / "gen_code" / "generate.py"),
        "--base_model",
        str(args.target_model_path),
        "--tokenizer_path",
        str(tokenizer_path),
        "--input_file",
        str(prompts_file),
        "--output_file",
        str(generation_file),
        "--regen",
        "1",
        "--batchsize",
        str(int(args.generation_batch_size)),
    ]
    _run(cmd, cwd=_repo_root(), dry_run=bool(args.dry_run))
    return generation_file


def _evaluate(args: argparse.Namespace, benchmark: str, generation_file: Path) -> Path:
    eval_file = args.output_dir / "eval" / f"{benchmark}.json"
    summary_json = args.output_dir / "summaries" / f"{benchmark}.json"
    summary_csv = args.output_dir / "asr_summary.csv"
    cmd = [
        sys.executable,
        str(_repo_root() / "evaluation" / "paper_harmful_asr.py"),
        "--input-file",
        str(generation_file),
        "--output-file",
        str(eval_file),
        "--summary-json",
        str(summary_json),
        "--summary-csv",
        str(summary_csv),
        "--benchmark",
        benchmark,
        "--scorer",
        str(args.asr_scorer),
        "--batch-size",
        str(int(args.asr_batch_size)),
        "--device",
        str(args.asr_device),
        "--classifier-model-path",
        str(args.classifier_model_path),
    ]
    if args.classifier_tokenizer_path:
        cmd.extend(["--classifier-tokenizer-path", str(args.classifier_tokenizer_path)])
    _run(cmd, cwd=_repo_root(), dry_run=bool(args.dry_run))
    return summary_json


def main() -> None:
    args = build_parser().parse_args()
    if args.single_gpu_staged:
        args.source_device = "cuda:0"
        args.target_device = "cuda:0"
        args.asr_device = "cuda:0"
    benchmarks = _expand_benchmarks(list(args.benchmarks))
    _write_run_config(args, benchmarks)
    _construct_benchmarks(args, benchmarks)
    if args.construct_only:
        print(f"[done] constructed benchmarks under {args.output_dir / 'benchmarks'}")
        return
    summary_csv = args.output_dir / "asr_summary.csv"
    if not args.skip_eval and not args.dry_run and summary_csv.exists():
        summary_csv.unlink()

    for benchmark in benchmarks:
        csv_path = args.output_dir / "benchmarks" / args.prompt_template / f"{benchmark}.csv"
        count = _row_count(csv_path)
        if count <= 0:
            print(f"[skip] {benchmark}: no rows at {csv_path}")
            continue
        log_file = args.output_dir / "search" / benchmark / "search.log"
        if not args.skip_search:
            log_file = _search(args, benchmark, csv_path, count)
        prompts_file = args.output_dir / "prompts" / f"{benchmark}.json"
        if not args.skip_generation:
            prompts_file = _prepare_generation(args, benchmark, csv_path, log_file)
            generation_file = _generate(args, benchmark, prompts_file)
        else:
            generation_file = args.output_dir / "generations" / f"{benchmark}.json"
        if not args.skip_eval:
            _evaluate(args, benchmark, generation_file)

    print(f"[done] suite output: {args.output_dir}")
    print(f"[done] ASR summary: {args.output_dir / 'asr_summary.csv'}")


if __name__ == "__main__":
    main()
