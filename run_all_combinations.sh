#!/bin/bash
set -e
repo="/mnt/data2/lian/projects/watermark/watermark-simcse"

model_name="cardiffnlp/twitter-roberta-base-sentiment"  # "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
gpu_id=5
dataset=c4
batch_size=64
train_epochs=2000
LOSS_FUNCTION_IDS=(2)  # 2 3 4
NEG_WEIGHTS=(1)  # 1 32 64 128
num_paraphrased=17
num_negative=1
data_generation_model="mix_wm_GPT-4o_Llama-3.1-8B-Instruct"
data_path="/mnt/data2/lian/projects/watermark/data/${data_generation_model}/onebatch-c4-train-simcse-all-filtered-formatted.csv"

model_name_="${model_name#*/}"

for loss_function_id in "${LOSS_FUNCTION_IDS[@]}"; do
  for neg_weight in "${NEG_WEIGHTS[@]}"; do
    bash SimCSE/run_sup_example_inbatch.sh \
      --gpu_id $gpu_id \
      --batch_size $batch_size \
      --train_epochs $train_epochs \
      --loss_function_id $loss_function_id \
      --num_paraphrased $num_paraphrased \
      --num_negative $num_negative \
      --neg_weight $neg_weight \
      --model_name $model_name \
      --train_file $data_path \
      --data_generation_model $data_generation_model
    
    embed_map_model="$repo/SimCSE/result/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/${data_generation_model}/end2end-c4-loss_cl${loss_function_id}-wneg${neg_weight}-${num_paraphrased}paras-${num_negative}negs"
    watermark_output_dir="$repo/watermarking/outputs/end2end/$dataset/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/${data_generation_model}/${num_paraphrased}paras-${num_negative}negs"

    bash watermarking/run_watermark.sh \
      --gpu_id $gpu_id \
      --watermark_output_dir $watermark_output_dir \
      --embed_map_model $embed_map_model \
      --neg_weight $neg_weight \
      --loss_function_id $loss_function_id \
      --data_path $data_path

  done
done
