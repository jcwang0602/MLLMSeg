export PYTHONPATH=.:$PYTHONPATH

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=$(shuf -i 20000-65000 -n 1) \
    mllmseg/test_mllmseg_res.py \
    --checkpoint checkpoints/MLLMSeg_InternVL2_RES