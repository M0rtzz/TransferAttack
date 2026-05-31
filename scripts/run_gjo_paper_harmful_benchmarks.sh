#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

paper_harmful_benchmarks=(
    harmbench
    jailbreakbench
    strongreject
    advbench
    malicious_instruct
    do_not_answer
    xstest_unsafe
    sorrybench
    wildjailbreak
)

if [ "${BENCHMARKS:-}" = "all" ]; then
    benchmarks="${paper_harmful_benchmarks[*]}"
else
    benchmarks="${BENCHMARKS:-${paper_harmful_benchmarks[*]}}"
fi
prompt_template="${PROMPT_TEMPLATE:-gjo}"
benchmark_max_examples="${BENCHMARK_MAX_EXAMPLES:-100}"
seed="${SEED:-0}"

python ./construct_paper_harmful_benchmarks.py \
    --benchmarks ${benchmarks} \
    --prompt-template "${prompt_template}" \
    --benchmark-max-examples "${benchmark_max_examples}" \
    --seed "${seed}"

for benchmark in ${benchmarks}; do
    csv_path="../data/benchmarks/${prompt_template}/${benchmark}.csv"
    if [ ! -s "${csv_path}" ]; then
        echo "Skipping ${benchmark}: ${csv_path} is empty or missing."
        continue
    fi
    RUN_NAME="mistral_v03_to_llama31_${benchmark}" \
    n="${benchmark_max_examples}" \
    data_path="${csv_path}" \
    bash ./run_gjo_mistral_v03_to_llama31.sh
done
