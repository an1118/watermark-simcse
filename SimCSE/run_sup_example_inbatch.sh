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
    --num_paraphrased)
      num_paraphrased="$2"
      shift
      shift
      ;;
    --num_negative)
      num_negative="$2"
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
    --train_file)
      train_file="$2"
      shift
      shift
      ;;
    --data_generation_model)
      data_generation_model="$2"
      shift
      shift
      ;;
    --pooler_type)
      pooler_type="$2"
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
echo "HARD_NEGATIVE_WEIGHT: $HARD_NEGATIVE_WEIGHT"

echo "========Running following parameters combination...========"
echo "model: $model_name"
echo "contrastive loss index: $loss_function_id"
echo "negative weight: $neg_weight"

CUDA_VISIBLE_DEVICES=$gpu_id ACCELERATE_LOG_LEVEL=info accelerate launch --config_file SimCSE/simcse_config.yaml SimCSE/train.py \
    --model_name_or_path ${model_name} \
    --train_file ${train_file} \
    --output_dir SimCSE/result/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/${data_generation_model}/end2end-c4-loss_cl${loss_function_id}-wneg${neg_weight}-${num_paraphrased}paras-${num_negative}negs \
    --hard_negative_weight $HARD_NEGATIVE_WEIGHT \
    --loss_function_id  ${loss_function_id} \
    --num_paraphrased $num_paraphrased \
    --num_negative $num_negative \
    --num_train_epochs $train_epochs \
    --per_device_train_batch_size $batch_size \
    --learning_rate 3e-5 \
    --max_seq_length 320 \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 2 \
    --pooler_type ${pooler_type} \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    --gradient_checkpointing \
    --report_to="wandb" \
    --run_name="sc-${model_name_}-c4-loss_cl${loss_function_id}-wneg${neg_weight}-batch${batch_size}-${data_generation_model}-${num_paraphrased}paras-${num_negative}negs" \
    --logging_steps=1 \
    "$@"
    # --validation_file SimCSE/data/c4-test-simcse-all-filtered-formatted.csv \
    # --per_device_eval_batch_size $batch_size \
    # --do_eval \
    # --evaluation_strategy steps \
    # --metric_for_best_model loss \
    # --load_best_model_at_end \
    # --eval_steps 5 \
