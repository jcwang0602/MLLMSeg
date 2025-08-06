GPUS=4
CPUS_PER_TASK=$((GPUS * 16))
srun -p mineru4s --gres=gpu:${GPUS} --job-name=eval --quotatype=reserved --cpus-per-task=${CPUS_PER_TASK} \
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=$(shuf -i 20000-65000 -n 1) \
    mllmseg_internvl/test_internvl_vlvcl_gres.py \
    --checkpoint work_dirs_internvl/01_st_lora64_internvl3_8b_b32_g4_vlvcl_gres_20000