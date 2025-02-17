while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpu_id)
      gpu_id="$2"
      shift
      shift
      ;;
    --watermark_output_dir)
      watermark_output_dir="$2"
      shift
      shift
      ;;
    --embed_map_model)
      embed_map_model="$2"
      shift
      shift
      ;;
    --neg_weight)
      neg_weight="$2"
      shift
      shift
      ;;
    --loss_function_id)
      loss_function_id="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

dataset=c4

num_of_sent=10
alpha=2.0
# beta=0.0
delta_0=0.2
delta=0.5

rm -f watermarking/models_cl.py
# ln -s SimCSE/simcse/models.py watermarking/models_cl.py
cp SimCSE/simcse/models.py watermarking/models_cl.py

watermark_result_file="$watermark_output_dir/watermark-8b-loss_cl${loss_function_id}_gr_wneg${neg_weight}-${num_of_sent}sent-alpha${alpha}-delta${delta_0}|${delta}.csv"  # for c4
output_file="$watermark_output_dir/eda/sim_watermark-8b-loss_cl${loss_function_id}_gr_wneg${neg_weight}-${num_of_sent}sent-alpha${alpha}-delta${delta_0}|${delta}.csv"  # for c4

# watermark_output_file="outputs/${method}/${dataset}/watermark-8b-mm|${measure_model:0:4}-alpha${alpha}-beta${beta}-delta${delta_0}|${delta}.csv"  # for lfqa

HARD_NEGATIVE_WEIGHT=$(python3 -c "import math; print(math.log(${neg_weight}))")

# watermarking
CUDA_VISIBLE_DEVICES=$gpu_id python watermarking/eda_1step_end2end.py \
    --embed_map_model=$embed_map_model \
    --hard_negative_weight=$HARD_NEGATIVE_WEIGHT \
    --num_of_sent=${num_of_sent} \
    --result_file=${watermark_result_file} \
    --output_file=${output_file} \
    --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
    # --correct_grammar=false
# # correct grammar
# CUDA_VISIBLE_DEVICES=$gpu_id python $repo/correct_grammar.py \
#     --alpha=${alpha} --generation_file=${watermark_file} \
#     --output_dir="/mnt/data2/lian/projects/watermark/adaptive-text-watermark-yepeng/outputs/correct_grammar/watermark-8b-1step-10sent-grammar-prompt4.csv"

