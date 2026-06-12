#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_BASE_PATH="${CONDA_BASE_PATH:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mllmseg}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
LORA_RANK=${LORA_RANK:-64}
LR=${LR:-1e-4}
RES_TOTAL_STEPS=${RES_TOTAL_STEPS:-50000}
GRES_TOTAL_STEPS=${GRES_TOTAL_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-10000}
LOGGING_STEPS=${LOGGING_STEPS:-100}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-16}
RES_REFER_DATA="refclef||refcoco||refcoco+||refcocog"
GRES_REFER_DATA="grefcoco"

new_master_port() {
    export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
}

setup_env() {
    cd "${PROJECT_ROOT}"
    source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    export PATH="${CUDA_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
    export CUDA_HOME="${CUDA_PATH}"
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export TF_CPP_MIN_LOG_LEVEL=3
    export LAUNCHER=pytorch
    new_master_port
}

build_output_dir() {
    RES_OUTPUT_DIR="work_dirs/mllmseg_${MODEL_TAG}_lora${LORA_RANK}_b${BATCH_SIZE}_pdbs${PER_DEVICE_BATCH_SIZE}_gas${GRADIENT_ACC}_gpu${GPUS}_lr${LR}"
    GRES_OUTPUT_DIR="${RES_OUTPUT_DIR}_gres"
}

run_torchrun() {
    new_master_port
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc_per_node="${GPUS}" \
        --master_port="${MASTER_PORT}" \
        "$@"
}

run_res_train() {
    run_torchrun -m mllmseg.finetune_mllmseg \
        --model_name_or_path "${MODEL_NAME}" \
        --train_dataset "refer_seg" \
        --refer_seg_data "${RES_REFER_DATA}" \
        --output_dir "${RES_OUTPUT_DIR}" \
        --overwrite_output_dir True \
        --freeze_llm True \
        --freeze_mlp True \
        --freeze_backbone True \
        --use_llm_lora "${LORA_RANK}" \
        --dataloader_num_workers "${DATALOADER_WORKERS}" \
        --bf16 True \
        --num_train_epochs 1 \
        --total_steps "${RES_TOTAL_STEPS}" \
        --ce_loss_weight 1.0 \
        --focal_loss_weight 1.0 \
        --dice_loss_weight 1.0 \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACC}" \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit 1 \
        --learning_rate "${LR}" \
        --weight_decay 0.05 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps "${LOGGING_STEPS}" \
        --do_train True \
        --ps_version 'v2' \
        --deepspeed "zero_stage1_config.json" \
        --report_to "tensorboard" \
        >> "${RES_OUTPUT_DIR}/training_log.txt" 2>&1
}

run_res_eval() {
    run_torchrun -m mllmseg.test_mllmseg_res \
        --checkpoint "${RES_OUTPUT_DIR}" \
        >> "${RES_OUTPUT_DIR}/eval_log.txt" 2>&1
}

run_gres_train() {
    run_torchrun -m mllmseg.finetune_mllmseg \
        --model_name_or_path "${MODEL_NAME}" \
        --pretrained_model "${RES_OUTPUT_DIR}" \
        --train_dataset "refer_seg" \
        --refer_seg_data "${GRES_REFER_DATA}" \
        --output_dir "${GRES_OUTPUT_DIR}" \
        --overwrite_output_dir True \
        --freeze_llm True \
        --freeze_mlp True \
        --freeze_backbone True \
        --use_llm_lora "${LORA_RANK}" \
        --dataloader_num_workers "${DATALOADER_WORKERS}" \
        --bf16 True \
        --num_train_epochs 1 \
        --total_steps "${GRES_TOTAL_STEPS}" \
        --ce_loss_weight 1.0 \
        --focal_loss_weight 1.0 \
        --dice_loss_weight 1.0 \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACC}" \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit 1 \
        --learning_rate "${LR}" \
        --weight_decay 0.05 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps "${LOGGING_STEPS}" \
        --do_train True \
        --ps_version 'v2' \
        --deepspeed "zero_stage1_config.json" \
        --report_to "tensorboard" \
        >> "${GRES_OUTPUT_DIR}/training_log.txt" 2>&1
}

run_gres_eval() {
    run_torchrun -m mllmseg.test_mllmseg_gres \
        --checkpoint "${GRES_OUTPUT_DIR}" \
        >> "${GRES_OUTPUT_DIR}/eval_log.txt" 2>&1
}

run_full_pipeline() {
    setup_env
    build_output_dir
    mkdir -p "${RES_OUTPUT_DIR}" "${GRES_OUTPUT_DIR}"
    run_res_train
    run_res_eval
    run_gres_train
    run_gres_eval
}
