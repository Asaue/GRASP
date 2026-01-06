#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"

data_paths="${REPO_HOME}/dataset/dataset_root/crack_ref_train.json"
image_folders="${REPO_HOME}/dataset/dataset_root/images"
model_path="/Data/Docker_liuwu/models/checkpoints/PaDT-REC-3B_RL_crack"
output_dir="${REPO_HOME}/eval/outputs/sft_inference_results"

mkdir -p $output_dir
cd ${REPO_HOME}/src/PaDT

# [关键修改] 添加环境变量
export PADT_INFERENCE_MODE=1 

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12399" \
  sft_train.py \
    --output_dir $output_dir \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --learning_rate 0.0 \
    --warmup_ratio 0.0 \
    --report_to "none" \
    --run_name "inference_run" \
    --do_train True \
    --deepspeed ${REPO_HOME}/src/PaDT/local_scripts/zero3.json # 删掉了 --predict_with_generate True