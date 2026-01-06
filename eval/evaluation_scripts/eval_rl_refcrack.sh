#!/bin/bash

# --- 1. 自动跳转到脚本所在目录 (防止路径错误) ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR" || exit
echo "Current working directory: $(pwd)"

# --- 2. 配置部分 ---

# 模型权重路径
export CHECKPOINT='/Data/Docker_liuwu/models/checkpoints/PaDT-REC-3B_RL_crack'

# 日志后缀 (输出文件会以此结尾，例如 ..._padt_crack.json)
export LOG_SUFFIX='padt_rl_crack'

# 你的自定义 Split 名称列表
# 注意：因为 Python 里数据路径已经写死了，这个名字仅仅用于给"输出文件"命名。
# 你可以把它改成 'crack_test' 或者 'my_dataset' 都可以。
SPLITS=("crack_val_rl") 

# --- 3. 循环执行 ---
for SPLIT in "${SPLITS[@]}"
do
    echo "========================================================"
    echo "Starting Process for Split: $SPLIT"
    echo "========================================================"

    # ---------------------------
    # 步骤 A: 推理 (Inference)
    # ---------------------------
    echo "[1/2] Running Inference..."
    # 使用 4 张卡 (0,1,2,3)
    # master_port 修改为 12375 防止和别人冲突
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --master_port="12375" \
        --nproc_per_node=4 \
        inference_rl_refcrack.py "$CHECKPOINT" "$SPLIT" "$LOG_SUFFIX"
    
    # 检查推理是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Inference failed for $SPLIT. Stopping."
        exit 1
    fi

    # ---------------------------
    # 步骤 B: 评测 (Evaluation)
    # ---------------------------
    echo "[2/2] Running Evaluation..."
    # 调用评测脚本计算指标
    # 注意参数顺序要和 python 脚本里接收的一致
    python eval_rl_refcrack.py "$LOG_SUFFIX" "$SPLIT"

    echo "========================================================"
    echo "Finished $SPLIT"
    echo -e "\n\n"
done;