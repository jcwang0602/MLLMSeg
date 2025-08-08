set -x

GPUS=4
BATCH_SIZE=32
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
LORA_RANK=64
MODEL_NAME="OpenGVLab/InternVL2_5-8B"
arr=(${MODEL_NAME//\// })

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR="work_dirs/mllmseg_lora${LORA_RANK}_${arr[1]}_b${BATCH_SIZE}_g${PER_DEVICE_BATCH_SIZE}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "Starting training..."
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    mllmseg/finetune_mllmseg.py \
    --model_name_or_path "${MODEL_NAME}" \
    --train_dataset "refer_seg" \
    --refer_seg_data "refclef||refcoco||refcoco+||refcocog" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir True \
    --freeze_llm True \
    --freeze_mlp True \
    --freeze_backbone True \
    --use_llm_lora "${LORA_RANK}" \
    --dataloader_num_workers 4 \
    --bf16 True \
    --num_train_epochs 1 \
    --total_steps 50000 \
    --ce_loss_weight 1.0 \
    --focal_loss_weight 1.0 \
    --dice_loss_weight 1.0 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACC}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1000 \
    --do_train True \
    --ps_version 'v2' \
    --deepspeed "zero_stage1_config.json" \
    --report_to "tensorboard" \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
