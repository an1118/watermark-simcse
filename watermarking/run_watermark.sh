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
    --data_path)
      data_path="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

num_of_sent=10
alpha=2.0
# beta=0.0
delta_0=0.2
delta=0.5

rm -f watermarking/models_cl.py
# ln -s SimCSE/simcse/models.py watermarking/models_cl.py
cp SimCSE/simcse/models.py watermarking/models_cl.py

watermark_output_file="$watermark_output_dir/watermark-8b-loss_cl${loss_function_id}_gr_wneg${neg_weight}-${num_of_sent}sent-alpha${alpha}-delta${delta_0}|${delta}.csv"  # for c4
# eda_output_file="$watermark_output_dir/watermark-8b-loss_cl${loss_function_id}_gr_wneg${neg_weight}-${num_of_sent}sent-alpha${alpha}-delta${delta_0}|${delta}-sim.csv"  # for c4
eda_output_file="$watermark_output_dir/multiple-spoofing-attack.csv"

HARD_NEGATIVE_WEIGHT=$(python3 -c "import math; print(math.log(${neg_weight}))")

# watermarking
CUDA_VISIBLE_DEVICES=$gpu_id python watermarking/generation_1step_end2end.py \
    --embed_map_model=$embed_map_model \
    --hard_negative_weight=$HARD_NEGATIVE_WEIGHT \
    --num_of_sent=${num_of_sent} \
    --output_file=${watermark_output_file} \
    --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
    --data_path ${data_path} \
    # --correct_grammar=false

# # ===== get similarity after 0-1 mapping =====
# CUDA_VISIBLE_DEVICES=$gpu_id python watermarking/eda_1step_end2end.py \
#     --embed_map_model=$embed_map_model \
#     --hard_negative_weight=$HARD_NEGATIVE_WEIGHT \
#     --num_of_sent=${num_of_sent} \
#     --output_file=${eda_output_file} \
#     --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
#     --result_file=${watermark_output_file}

# # ===== multiple time attack =====
# CUDA_VISIBLE_DEVICES=$gpu_id python watermarking/eda_1step_end2end_multi-attack.py \
#     --embed_map_model=$embed_map_model \
#     --hard_negative_weight=$HARD_NEGATIVE_WEIGHT \
#     --num_of_sent=${num_of_sent} \
#     --output_file=${eda_output_file} \
#     --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
#     --result_file=${watermark_output_file}
