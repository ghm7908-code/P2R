#!/bin/bash
#SBATCH --job-name=P2R_Train           # 任务名称
#SBATCH --partition=GEOG-HPC-GPU       # 你的 GPU 分区
#SBATCH --gres=gpu:1                   # 1块 GPU
#SBATCH --mem=64G                      # 内存
#SBATCH --cpus-per-task=16              # CPU
#SBATCH --time=48:00:00                # 时长 48 小时
#SBATCH --output=logs/train_%j.out     # 标准输出日志
#SBATCH --error=logs/train_%j.err      # 错误输出日志

# 1. 初始化 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate p2f

# 2. 进入项目根目录（★ 请确认这里是你 git pull 的目录 ★）
cd /geogfs1/home/u3666068/Point2Roof-master/

# 3. 环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# 3.5 清除旧缓存
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 3.6 ★ 删除旧 checkpoint（基于坍缩 Z 维度的旧模型不可复用）★
echo "--- 清理旧 checkpoint ---"
rm -rf output/full_run_v8/ckpt/*checkpoint_epoch_*.pth 2>/dev/null
echo "旧 checkpoint 已清理（如有）"

# 4. 确保日志目录存在
mkdir -p logs

echo "--- 训练启动时间: $(date) ---"

# 5. GPU 诊断（排查 CUDA 初始化失败）
echo "--- GPU 诊断 ---"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi 2>/dev/null || echo "(nvidia-smi 不可用，可能未分配 GPU)"
echo "---"

# 6. 执行训练
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python train.py \
    --cfg_file model_cfg.yaml \
    --extra_tag 'full_run_v9'

# 6. 训练结束检查
if [ $? -eq 0 ]; then
    echo "--- 训练成功结束: $(date) ---"
else
    echo "--- 训练意外中止，检查 err 日志 ---"
    exit 1
fi