#!/bin/bash
#SBATCH --job-name=P2R_Test            # 任务名称
#SBATCH --partition=GEOG-HPC-GPU       # 你的 GPU 分区
#SBATCH --gres=gpu:1                   # 1块 GPU
#SBATCH --mem=32G                      # 内存
#SBATCH --cpus-per-task=8              # CPU
#SBATCH --time=12:00:00               # 时长 12 小时
#SBATCH --output=logs/test_%j.out     # 标准输出日志
#SBATCH --error=logs/test_%j.err      # 错误输出日志

# 1. 初始化 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate p2f

# 2. 进入项目根目录
cd /geogfs1/home/u3666068/Point2Roof-master/

# 3. 环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

echo "--- 测试启动时间: $(date) ---"

# 4. 执行测试 (基础测试，不带可视化)
python test.py \
    --cfg_file model_cfg.yaml \
    --data_path /geogfs1/groups/hkurs/u3666068mgh/Tallinn \
    --test_tag full_run_v2 \
    --gpu 0

TEST_RESULT=$?

# 5. 如果基础测试成功，可选：执行可视化
if [ $TEST_RESULT -eq 0 ]; then
    echo "--- 基础测试完成，开始可视化 ---"
    
    # 可视化单个样本测试
    python test.py \
        --cfg_file model_cfg.yaml \
        --data_path /geogfs1/groups/hkurs/u3666068mgh/Tallinn \
        --test_tag full_run_v2 \
        --gpu 0 \
        --visualize \
        --vis_sample_id sample_001
    
    # 可视化所有样本 (如果需要)
    # python test.py \
    #     --cfg_file model_cfg.yaml \
    #     --data_path /geogfs1/groups/hkurs/u3666068mgh/Tallinn \
    #     --test_tag full_run_v2 \
    #     --gpu 0 \
    #     --vis_all
fi

# 6. 测试结束检查
if [ $? -eq 0 ]; then
    echo "--- 测试成功结束 ---"
else
    echo "--- 测试意外中止，检查 err 日志 ---"
    exit 1
fi
