#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
model_name_="${model_name#*/}"
loss_function_id=2
neg_weight=64
HARD_NEGATIVE_WEIGHT=$(python3 -c "import math; print(math.log(${neg_weight}))")

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \

CUDA_VISIBLE_DEVICES=3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file simcse_config.yaml train.py \
    --model_name_or_path Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --train_file /mnt/data2/lian/projects/watermark/data/c4-train-simcse-all-filtered-formatted.csv \
    --validation_file /mnt/data2/lian/projects/watermark/data/c4-test-simcse-all-filtered-formatted.csv \
    --output_dir result/end2end-simcse-${model_name_}-c4-loss_cl${loss_function_id}_gr-wneg${neg_weight}-freeze \
    --hard_negative_weight $HARD_NEGATIVE_WEIGHT \
    --loss_function_id  ${loss_function_id} \
    --num_train_epochs 30 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 320 \
    --evaluation_strategy steps \
    --save_strategy best \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --eval_steps 5 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --report_to="wandb" \
    --run_name="wm-simcse-${model_name_}-c4-loss_cl${loss_function_id}_gr-wneg${neg_weight}-freeze" \
    --logging_steps=1 \
    --freeze_embed \
    "$@"
    # --gradient_accumulation_steps 16 \
    # --load_best_model_at_end \
