#!/bin/bash
set -e
repo="/mnt/data2/lian/projects/watermark/watermark-simcse"

model_name="cardiffnlp/twitter-roberta-base-sentiment"  # "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
gpu_id=0
dataset=c4
batch_size=8
train_epochs=1
LOSS_FUNCTION_IDS=(4)  # 2 3 4
NEG_WEIGHTS=(1)  # 1 32 64 128

model_name_="${model_name#*/}"

for loss_function_id in "${LOSS_FUNCTION_IDS[@]}"; do
  for neg_weight in "${NEG_WEIGHTS[@]}"; do
    bash SimCSE/run_sup_example_inbatch.sh \
      --gpu_id $gpu_id \
      --batch_size $batch_size \
      --train_epochs $train_epochs \
      --loss_function_id $loss_function_id \
      --neg_weight $neg_weight \
      --model_name $model_name
    
    watermark_output_dir="$repo/watermarking/outputs/end2end/$dataset/${model_name_}"
    embed_map_model="$repo/SimCSE/result/${model_name_}/end2end-c4-loss_cl${loss_function_id}_gr-wneg${neg_weight}"

    bash watermarking/run_watermark.sh \
      --gpu_id $gpu_id \
      --watermark_output_dir $watermark_output_dir \
      --embed_map_model $embed_map_model \
      --neg_weight $neg_weight \
      --loss_function_id $loss_function_id

  done
done
