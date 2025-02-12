#!/bin/bash

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu_id)
      gpu_id="$2"
      shift
      shift
      ;;
    --output_dir)
      output_dir="$2"
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
    --num_paraphrased_llama)
      num_paraphrased_llama="$2"
      shift
      shift
      ;;
    --num_paraphrased_gpt)
      num_paraphrased_gpt="$2"
      shift
      shift
      ;;
    --num_negative_llama)
      num_negative_llama="$2"
      shift
      shift
      ;;
    --num_negative_gpt)
      num_negative_gpt="$2"
      shift
      shift
      ;;
    --num_summary)
      num_summary="$2"
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
    --pooler_type)
      pooler_type="$2"
      shift
      shift
      ;;
    --freeze_base)
      freeze_base="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done


model_name_="${model_name#*/}"
if [[ "$model_name_" == "gte-Qwen2-1.5B-instruct" ]]; then
    model_name_="${model_name_}-${pooler_type}"
fi

# Append '-freeze' to model_name_ if freeze_base is True
if [ "$freeze_base" == "True" ]; then
    model_name_="${model_name_}-freeze"
fi


HARD_NEGATIVE_WEIGHT=$(python3 -c "import math; print(math.log(${neg_weight}))")
echo "HARD_NEGATIVE_WEIGHT: $HARD_NEGATIVE_WEIGHT"

echo "========Running following parameters combination...========"
echo "model: $model_name"
echo "contrastive loss index: $loss_function_id"
echo "negative weight: $neg_weight"

CUDA_VISIBLE_DEVICES=$gpu_id ACCELERATE_LOG_LEVEL=info accelerate launch --config_file SimCSE/simcse_config.yaml SimCSE/train.py \
    --model_name_or_path ${model_name} \
    --train_file ${train_file} \
    --output_dir ${output_dir} \
    --hard_negative_weight $HARD_NEGATIVE_WEIGHT \
    --loss_function_id  ${loss_function_id} \
    --num_paraphrased_llama $num_paraphrased_llama \
    --num_paraphrased_gpt $num_paraphrased_gpt \
    --num_negative_llama $num_negative_llama \
    --num_negative_gpt $num_negative_gpt \
    --num_summary $num_summary \
    --num_train_epochs $train_epochs \
    --per_device_train_batch_size $batch_size \
    --learning_rate 3e-5 \
    --max_seq_length 320 \
    --freeze_base $freeze_base \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 1 \
    --pooler_type ${pooler_type} \
    --temp 0.05 \
    --do_train \
    --overwrite_output_dir \
    --fp16 \
    --gradient_checkpointing \
    --report_to="wandb" \
    --run_name="sc-${model_name_}-c4-loss_cl${loss_function_id}-wneg${neg_weight}-batch${batch_size}-llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}" \
    --logging_steps=1 \
    "$@"
    # --validation_file SimCSE/data/c4-test-simcse-all-filtered-formatted.csv \
    # --per_device_eval_batch_size $batch_size \
    # --do_eval \
    # --evaluation_strategy steps \
    # --metric_for_best_model loss \
    # --load_best_model_at_end \
    # --eval_steps 5 \
