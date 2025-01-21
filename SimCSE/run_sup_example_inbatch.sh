#!/bin/bash

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu_id)
      gpu_id="$2"
      shift
      shift
      ;;
    --model_name)
      model_name="$2"
      shift
      shift
      ;;
    --loss_function_id)
      loss_function_id="$2"
      shift
      shift
      ;;
    --neg_weight)
      neg_weight="$2"
      shift
      shift
      ;;
    --batch_size)
      batch_size="$2"
      shift
      shift
      ;;
    --train_epochs)
      train_epochs="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done


model_name_="${model_name#*/}"
HARD_NEGATIVE_WEIGHT=$(python3 -c "import math; print(math.log(${neg_weight}))")

echo "========Running following parameters combination...========"
echo "model: $model_name"
echo "contrastive loss index: $loss_function_id"
echo "negative weight: $neg_weight"

CUDA_VISIBLE_DEVICES=$gpu_id ACCELERATE_LOG_LEVEL=info accelerate launch --config_file SimCSE/simcse_config.yaml SimCSE/train.py \
    --model_name_or_path ${model_name} \
    --train_file SimCSE/data/c4-train-simcse-all-filtered-formatted.csv \
    --validation_file SimCSE/data/c4-test-simcse-all-filtered-formatted.csv \
    --output_dir SimCSE/result/${model_name_}/end2end-c4-loss_cl${loss_function_id}_gr-wneg${neg_weight} \
    --hard_negative_weight $HARD_NEGATIVE_WEIGHT \
    --loss_function_id  ${loss_function_id} \
    --num_train_epochs $train_epochs \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate 3e-5 \
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
    --run_name="wm-simcse-${model_name_}-c4-loss_cl${loss_function_id}_gr-wneg${neg_weight}" \
    --logging_steps=1 \
    "$@"
