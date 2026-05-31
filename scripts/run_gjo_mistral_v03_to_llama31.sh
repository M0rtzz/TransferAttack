#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export n="${n:-20}"
export len="${len:-100}"
export token_num="${token_num:-2}"

steps="${steps:-500}"
system_message="${system_message:-False}"
method="${method:-gjo}"
data_path="${data_path:-../data/harmbench_${method}.csv}"
config_name="mistral_v03_to_llama31"
run_name="${RUN_NAME:-$config_name}"

mkdir -p ./results ./logs

control_init=""
for ((i=0; i<len; i++)); do
    control_init+="! "
done

python -u ./main.py \
    --config="./configs/transfer_${config_name}.py" \
    --config.attack=gcg \
    --config.train_data="${data_path}" \
    --config.result_prefix="./results/transfer_${run_name}_gcg_${n}_progressive_w_sys_len_${len}_suffix" \
    --config.progressive_goals=True \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data="$n" \
    --config.n_test_data="$n" \
    --config.n_steps="${steps}" \
    --config.test_steps=5 \
    --config.batch_size=128 \
    --config.topk=256 \
    --config.ce_prefix=True \
    --config.system_message="${system_message}" \
    --config.control_init="$control_init" \
    --config.token_num="${token_num}" | tee "logs/data_${n}_len_${len}_${run_name}_sys_${system_message}_${token_num}_token.txt"
