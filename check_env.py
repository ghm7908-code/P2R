#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from pathlib import Path

def check_full_environment():
    """完整环境检查"""
    print("=" * 80)
    print("Point2Roof 训练环境检查")
    print("=" * 80)
    
    # 基本信息
    print(f"用户: {os.getenv('USER')}")
    print(f"主机名: {os.uname().nodename}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.executable}")
    print(f"Python版本: {sys.version}")
    
    # PyTorch和CUDA
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    显存: {props.total_memory / 1e9:.2f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
    
    # 关键库版本
    print(f"\n关键库版本:")
    print(f"  NumPy: {np.__version__}")
    try:
        import yaml
        print(f"  PyYAML: {yaml.__version__}")
    except:
        print("  PyYAML: 未安装")
    
    # 数据路径检查
    data_path = Path("/geogfs1/home/u3666068/Point2Roof-master/data/Entry-level")
    print(f"\n数据路径: {data_path}")
    
    if data_path.exists():
        print("✅ 数据路径存在")
        # 列出内容
        print("  目录内容:")
        for item in data_path.iterdir():
            if item.is_dir():
                print(f"    📁 {item.name}/")
            else:
                print(f"    📄 {item.name}")
    else:
        print("❌ 数据路径不存在")
    
    # 代码路径检查
    code_path = Path("/geogfs1/home/u3666068/Point2Roof-master")
    print(f"\n代码路径: {code_path}")
    
    required_files = [
        "train.py",
        "train_utils.py", 
        "model/roofnet.py",
        "dataset/data_utils.py",
        "utils/common_utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (code_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺失文件:")
        for file in missing_files:
            print(f"    {file}")
    else:
        print("✅ 所有必需文件存在")
    
    # 内存和磁盘空间
    print(f"\n系统信息:")
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"  磁盘空间: 总共{total // (2**30)}GB, 已用{used // (2**30)}GB, 剩余{free // (2**30)}GB")
    
    import psutil
    mem = psutil.virtual_memory()
    print(f"  内存: 总共{mem.total // (2**30)}GB, 可用{mem.available // (2**30)}GB")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == "__main__":
    check_full_environment()