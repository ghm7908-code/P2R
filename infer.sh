#!/bin/bash
#SBATCH --job-name=P2R_Infer           # 任务名称
#SBATCH --partition=GEOG-HPC-GPU       # 你的 GPU 分区
#SBATCH --gres=gpu:1                   # 1块 GPU
#SBATCH --mem=32G                      # 内存 (推理比训练需求低)
#SBATCH --cpus-per-task=8              # CPU
#SBATCH --time=12:00:00                # 时长 12 小时
#SBATCH --output=logs/infer_%j.out     # 标准输出日志
#SBATCH --error=logs/infer_%j.err      # 错误输出日志

# 1. 初始化 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate p2f

# 2. 进入项目根目录
cd /geogfs1/home/u3666068/Point2Roof-master/

# 3. 环境变量（确保 Python 能找到所有模块）
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- 推理启动时间: $(date) ---"

# 4. 执行推理 (保持与 train.sh 相同的配置)
# --test_tag 对应 train.sh 的 --extra_tag，用于定位 checkpoint
# --split test  在测试集上评估
# --save_obj    导出预测的 OBJ 线框文件
export CUDA_VISIBLE_DEVICES=1
python test.py \
    --cfg_file model_cfg.yaml \
    --test_tag 'full_run_v2' \
    --split test \
    --batch_size 1 \
    --edge_thresh 0.5 \
    --point_thresh 0.1 \
    --ap_distance_thresh 0.1 \
    --save_obj

# 5. 推理结束检查
if [ $? -eq 0 ]; then
    echo "--- 推理成功结束 ---"
else
    echo "--- 推理意外中止，检查 err 日志 ---"
    exit 1
fi
