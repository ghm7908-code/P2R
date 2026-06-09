#!/bin/bash
#SBATCH --job-name=P2R_Infer           # 任务名称
#SBATCH --partition=GEOG-HPC-GPU       # GPU 分区
#SBATCH --gres=gpu:1                   # 1块 GPU
#SBATCH --mem=32G                      # 推理内存需求较低
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

# ============================================================
#  推理全流程: 加载 checkpoint → 评测指标 → 导出 OBJ → 离线校验
# ============================================================

set -e  # 遇错即停

# ---------- 可调参数 ----------
EXTRA_TAG="full_run_v2"      # 与 train.sh --extra_tag 一致
SPLIT="val"                  # val / test
EDGE_THRESH=0.5
POINT_THRESH=0.1
AP_DIST_THRESH=0.1
GPU_ID=1
# -----------------------------

# 1. 环境初始化
source ~/miniconda3/etc/profile.d/conda.sh
conda activate p2f
cd /geogfs1/home/u3666068/Point2Roof-master/
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

# 清除旧缓存
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "============================================================"
echo "  推理启动: $(date)"
echo "  extra_tag: ${EXTRA_TAG}"
echo "  split:     ${SPLIT}"
echo "============================================================"

# 2. 自动查找 checkpoint
CKPT_DIR="output/${EXTRA_TAG}/ckpt"
LATEST_CKPT=$(ls -t ${CKPT_DIR}/checkpoint_epoch_*.pth 2>/dev/null | head -1 || true)

if [ -z "${LATEST_CKPT}" ]; then
    echo "ERROR: 未找到 checkpoint，检查 ${CKPT_DIR}/"
    exit 1
fi
echo "=> 使用 checkpoint: ${LATEST_CKPT}"

# 3. 主干推理 + 评测 (test.py 内部计算全部指标并写入 test_log.txt)
echo "=> 开始评测..."
export CUDA_VISIBLE_DEVICES=${GPU_ID}
python test.py \
    --cfg_file model_cfg.yaml \
    --ckpt "${LATEST_CKPT}" \
    --test_tag "${EXTRA_TAG}" \
    --split "${SPLIT}" \
    --batch_size 1 \
    --edge_thresh "${EDGE_THRESH}" \
    --point_thresh "${POINT_THRESH}" \
    --ap_distance_thresh "${AP_DIST_THRESH}" \
    --save_obj

if [ $? -ne 0 ]; then
    echo "ERROR: 推理失败，检查日志"
    exit 1
fi

# 4. 离线评估 — 用 evaluate_wireframe.py 对导出 OBJ 做二次校验 (ACO/WED/CP/CR/CF1/EP/ER/EF1)
PRED_OBJ_DIR="output/${EXTRA_TAG}/test_results"
GT_DATA_DIR=$(python -c "from utils import common_utils; cfg = common_utils.cfg_from_yaml_file('model_cfg.yaml'); print(cfg.DATA.root_dir)" 2>/dev/null || echo "")

if [ -n "${GT_DATA_DIR}" ] && [ -d "${PRED_OBJ_DIR}" ]; then
    echo "=> 离线 OBJ 评估..."
    python evaluate_wireframe.py \
        --pred_dir "${PRED_OBJ_DIR}" \
        --gt_dir "${GT_DATA_DIR}/${SPLIT}" \
        --pred_ext ".obj" \
        --gt_ext ".obj" \
        --distance_thresh "${AP_DIST_THRESH}" \
        --output_json "output/${EXTRA_TAG}/eval_summary.json" \
        --output_csv  "output/${EXTRA_TAG}/eval_per_sample.csv"
    echo "=> 离线评估完成 → output/${EXTRA_TAG}/eval_summary.json"
else
    echo "=> 跳过离线评估 (pred_dir 或 GT 路径不可用)"
fi

echo "============================================================"
echo "  推理完成: $(date)"
echo "  日志:   output/${EXTRA_TAG}/test_log.txt"
echo "  OBJ:    ${PRED_OBJ_DIR}/"
echo "============================================================"
