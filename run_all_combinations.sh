#!/bin/bash
set -e
repo="/blue/buyuheng/li_an.ucsb/projects/watermark-simcse"

model_name="cardiffnlp/twitter-roberta-base-sentiment"  # Alibaba-NLP/gte-Qwen2-1.5B-instruct    cardiffnlp/twitter-roberta-base-sentiment
freeze_base=False
pooler_type="attention"
gpu_id=4
dataset=imdb
batch_size=128
train_epochs=87

cl_weight=0.1
tl_weight=0.9
neg_weight=1
margin=0.8

if (( $(echo "$cl_weight != 0.0" | bc -l) )); then
  neg_weight=$neg_weight  # 1 32 64 128
else
  neg_weight=999
fi
if (( $(echo "$tl_weight != 0.0" | bc -l) )); then
  margin=$margin  # 0.2 0.5 0.8 1.1
else
  margin=999
fi

num_paraphrased_llama=8
num_paraphrased_gpt=8
num_negative_llama=0
num_negative_gpt=1
num_summary=0
# data_path="$repo/data/sc-onebatch-$dataset-train-simcse-all-filtered-formatted.csv"
data_path_prefix="$repo/data/sc-$dataset-simcse-filtered"
watermark_data_path="https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/realnewslike/c4-train.00000-of-00512.json.gz"

model_name_=$(basename "$model_name")
if [[ "$model_name_" == "gte-Qwen2-1.5B-instruct" ]]; then
    model_name_="${model_name_}-${pooler_type}"
fi
# Append '-freeze' to model_name_ if freeze_base is True
if [ "$freeze_base" == "True" ]; then
    model_name_="${model_name_}-freeze"
fi


embed_map_model="${repo}/SimCSE/result/${dataset}/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-${num_summary}/loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}"
bash SimCSE/run_sup_example_inbatch.sh \
  --gpu_id $gpu_id \
  --train_file ${data_path_prefix}-train.csv \
  --valid_file ${data_path_prefix}-valid.csv \
  --dataset $dataset \
  --num_paraphrased_llama $num_paraphrased_llama \
  --num_paraphrased_gpt $num_paraphrased_gpt \
  --num_negative_llama $num_negative_llama \
  --num_negative_gpt $num_negative_gpt \
  --num_summary $num_summary \
  --model_name $model_name \
  --pooler_type $pooler_type \
  --freeze_base $freeze_base \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --cl_weight $cl_weight \
  --tl_weight $tl_weight \
  --neg_weight $neg_weight \
  --margin $margin \
  --output_dir $embed_map_model


num_of_sent=10
alpha=2.0
delta_0=0.2
delta=0.5

watermark_output_dir="$repo/watermarking/outputs/end2end/$dataset/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-${num_summary}"
watermark_output_file="$watermark_output_dir/watermark-loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}-${num_of_sent}sent-alpha${alpha}-delta${delta_0}|${delta}.csv"  # for c4
eda_output_file="$watermark_output_dir/watermark-loss_cl${cl_weight}-tl${tl_weight}-wneg${neg_weight}-margin${margin}-${num_of_sent}sent-alpha${alpha}-delta${delta_0}|${delta}-sim.csv"  # for c4

bash watermarking/run_watermark.sh \
  --gpu_id $gpu_id \
  --embed_map_model $embed_map_model \
  --data_path $watermark_data_path \
  --watermark_output_file $watermark_output_file \
  --eda_output_file $eda_output_file \
  --num_of_sent $num_of_sent \
  --alpha $alpha --delta_0 $delta_0 --delta $delta


# # add watermarked text
# data_path="/mnt/data2/lian/projects/watermark/watermark-simcse/watermarking/outputs/end2end/c4/twitter-roberta-base-sentiment/128batch_2000epochs/sanity-check/llama4-1gpt4-1-wm/onebatch-c4-train-simcse-all-filtered-formatted.csv"

# embed_map_model="${repo}/SimCSE/result/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-wm/end2end-c4-loss_cl${loss_function_id}-wneg${neg_weight}"


# for loss_function_id in "${LOSS_FUNCTION_IDS[@]}"; do
#   for neg_weight in "${NEG_WEIGHTS[@]}"; do
#     bash SimCSE/run_sup_example_inbatch.sh \
#       --gpu_id $gpu_id \
#       --output_dir $embed_map_model \
#       --batch_size $batch_size \
#       --train_epochs $train_epochs \
#       --loss_function_id $loss_function_id \
#       --num_paraphrased_llama $num_paraphrased_llama \
#       --num_paraphrased_gpt $num_paraphrased_gpt \
#       --num_negative_llama $num_negative_llama \
#       --num_negative_gpt $num_negative_gpt \
#       --neg_weight $neg_weight \
#       --model_name $model_name \
#       --train_file $data_path \
#       --pooler_type $pooler_type
    
#     watermark_output_dir="$repo/watermarking/outputs/end2end/$dataset/${model_name_}/${batch_size}batch_${train_epochs}epochs/sanity-check/llama${num_paraphrased_llama}-${num_negative_llama}gpt${num_paraphrased_gpt}-${num_negative_gpt}-wm"

#     bash watermarking/run_watermark.sh \
#       --gpu_id $gpu_id \
#       --watermark_output_dir $watermark_output_dir \
#       --embed_map_model $embed_map_model \
#       --neg_weight $neg_weight \
#       --loss_function_id $loss_function_id \
#       --data_path $data_path

#   done
# done
