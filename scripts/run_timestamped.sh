#!/usr/bin/env bash

set -euo pipefail
trap '' INT QUIT TSTP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat >&2 << 'EOF'
Usage:
    ./scripts/run_timestamped.sh [--log-name NAME] [--no-output-dir] <command> [args...]

Examples:
    CUDA_VISIBLE_DEVICES=0 ./scripts/run_timestamped.sh \
      --log-name single_gpu_suite \
      python scripts/run_paper_harmful_suite.py \
      --single-gpu-staged \
      --source-model-path ./Models/Mistral-7B-Instruct-v0.3 \
      --target-model-path ./Models/Meta-Llama-3.1-8B-Instruct \
      --benchmarks all \
      --benchmark-max-examples 100

    ./scripts/run_timestamped.sh \
      --no-output-dir \
      bash scripts/run_gjo_mistral_v03_to_llama31.sh

Notes:
    - By default this wrapper creates outputs/<timestamp>/ and injects
      --output-dir <that directory> when the command has not already supplied it.
    - Use --no-output-dir for commands that do not accept --output-dir.
    - stdout and stderr are both streamed to the terminal and saved to a log file.
EOF
}

log_name=""
inject_output_dir=1

while [[ $# -gt 0 ]]; do
    case "${1}" in
        --log-name)
            if [[ $# -lt 2 ]]; then
                echo "error: --log-name requires a value" >&2
                usage
                exit 1
            fi
            log_name="$2"
            shift 2
            ;;
        --no-output-dir)
            inject_output_dir=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "error: missing command" >&2
    usage
    exit 1
fi

timestamp="$(TZ=Asia/Shanghai date '+%Y-%m-%d_%H:%M:%S')"
run_dir="${ROOT_DIR}/outputs/${timestamp}"
suffix=1
while [[ -e "${run_dir}" ]]; do
    run_dir="${ROOT_DIR}/outputs/${timestamp}_${suffix}"
    suffix=$((suffix + 1))
done
mkdir -p "${run_dir}"

if [[ -z "${log_name}" ]]; then
    log_name="$(basename "${1}")"
    log_name="${log_name%.*}"
fi
log_path="${run_dir}/${log_name}.log"

cmd=("$@")
has_output_dir=0
for arg in "${cmd[@]}"; do
    if [[ "${arg}" == "--output-dir" || "${arg}" == --output-dir=* ]]; then
        has_output_dir=1
        break
    fi
done

if [[ "${inject_output_dir}" -eq 1 && "${has_output_dir}" -eq 0 ]]; then
    cmd+=("--output-dir" "${run_dir}")
fi

print_command() {
    echo "[Command]"
    local i=0
    while [[ "${i}" -lt "${#cmd[@]}" ]]; do
        local escaped=""
        printf -v escaped '%q' "${cmd[$i]}"
        if [[ "${i}" -eq 0 ]]; then
            printf '%s' "${escaped}"
        else
            printf ' %s' "${escaped}"
        fi
        i=$((i + 1))
    done
    printf '\n'
}

status=0
set +e
{
    echo "[Run Dir] ${run_dir}"
    echo "[Log Path] ${log_path}"
    print_command
    echo
    cd "${ROOT_DIR}"
    "${cmd[@]}"
} 2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}
set -e

echo "[Exit Status] ${status}" | tee -a "${log_path}"
exit "${status}"
