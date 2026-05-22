#!/bin/bash
#SBATCH --job-name=P2R_Train           # 任务名称
#SBATCH --partition=GEOG-HPC-GPU       # 你的 GPU 分区
#SBATCH --gres=gpu:1                   # 1块 GPU
#SBATCH --mem=64G                      # 内存
#SBATCH --cpus-per-task=16              # CPU
#SBATCH --time=48:00:00                # 时长 3 天
#SBATCH --output=logs/train_%j.out     # 标准输出日志
#SBATCH --error=logs/train_%j.err      # 错误输出日志

# 1. 初始化 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate p2f

# 2. 进入项目根目录
cd /geogfs1/home/u3666068/Point2Roof-master/

# 3. 环境变量（确保 Python 能找到所有模块）
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- 训练启动时间: $(date) ---"

# 4. 执行训练 (保持你之前运行成功的参数)
# 注意：请将 your_config.yaml 替换为你实际使用的配置文件名
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --cfg_file model_cfg.yaml \
    --extra_tag 'full_run_v2'

# 5. 训练结束检查
if [ $? -eq 0 ]; then
    echo "--- 训练成功结束 ---"
else
    echo "--- 训练意外中止，检查 err 日志 ---"
    exit 1
fi