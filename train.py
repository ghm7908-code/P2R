import os
import torch
import argparse
import datetime
import glob
from torch import optim
from dataset.data_utils import build_dataloader
from train_utils import train_model
from model.roofnet import RoofNet
from utils import common_utils
from model import model_utils

def parse_config():
    parser = argparse.ArgumentParser(description="Point2Roof Training Script")
    # 核心参数：仅保留路径、配置文件和实验标签，其余由 yaml 控制
    parser.add_argument('--cfg_file', type=str, default='./model_cfg.yaml', help='配置文件路径')
    parser.add_argument('--data_path', type=str, default=None, help='可覆盖yaml中的数据路径')
    parser.add_argument('--extra_tag', type=str, default='default', help='实验标签')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=None, help='可覆盖yaml中的batch size')
    
    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    
    # --- 参数覆盖逻辑 ---
    # 如果命令行指定了参数，则覆盖 yaml 中的配置
    if args.data_path:
        cfg.DATA.root_dir = args.data_path
    if args.batch_size:
        cfg.DATA.batch_size = args.batch_size
    cfg.DATA.NPOINT = cfg.DATA.get('NPOINT', cfg.DATA.get('num_points', 4096))
    cfg.DATA.batch_size = cfg.DATA.get('batch_size', 64)
    cfg.DATA.num_workers = cfg.DATA.get('num_workers', 16)
    cfg.MODEL.use_edge = cfg.MODEL.get('use_edge', True)
    
    # 设置默认值（以防 yaml 中缺失核心键值）
    cfg.DATA.use_building3d_format = cfg.DATA.get('use_building3d_format', True)
    cfg.DATA.use_color = cfg.DATA.get('use_color', False)
    cfg.DATA.use_intensity = cfg.DATA.get('use_intensity', False)
    
    return args, cfg

def get_input_channels(cfg):
    """根据配置动态计算输入通道数"""
    channels = 3  # 基础 xyz
    desc = "xyz"
    if cfg.DATA.use_color:
        channels += 3
        desc += "+RGB"
    if cfg.DATA.use_intensity:
        channels += 1
        desc += "+Intensity"
    return channels, desc

def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 1. 实验环境准备
    time_str = datetime.datetime.now().strftime('%m%d_%H%M')
    extra_tag = args.extra_tag
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. 日志初始化
    logger = common_utils.create_logger(output_dir / 'log.txt')
    logger.info(f"==> 实验标签: {extra_tag}")
    
    # 3. 数据加载
    logger.info("==> 正在创建数据加载器...")
    try:
        train_loader = build_dataloader(
            cfg.DATA.root_dir, 
            cfg.DATA.batch_size, 
            cfg.DATA, 
            workers=cfg.DATA.num_workers, 
            logger=logger, 
            training=True,
        )
        logger.info(f"==> 成功加载训练样本: {len(train_loader.dataset)}")
    except Exception as e:
        logger.error(f"==> 数据加载失败: {e}")
        raise

    # 4. 模型构建
    input_channels, feat_desc = get_input_channels(cfg)
    logger.info(f"==> 输入特征: {feat_desc} (通道数: {input_channels})")
    
    model = RoofNet(cfg.MODEL, input_channel=input_channels)
    model.cuda()

    # 5. 优化器与调度器 (从 cfg 读取学习率和权重衰减)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.OPTIM.lr, 
        weight_decay=cfg.OPTIM.get('weight_decay', 1e-3)
    )
    
    # 6. 断点续训检查
    start_epoch = it = 0
    last_epoch = -1
    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    if len(ckpt_list) > 0:
        latest_ckpt = max(ckpt_list, key=os.path.getmtime)
        it, start_epoch = model_utils.load_params_with_optimizer(
            model, latest_ckpt, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1
        logger.info(f"==> 已从 {latest_ckpt} 恢复训练")

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.OPTIM.get('lr_step', 20), 
        gamma=cfg.OPTIM.get('lr_gamma', 0.5), 
        last_epoch=last_epoch
    )

    # 7. 开始训练
    logger.info(f"==> 开始训练: {cfg.OPTIM.epochs} 轮")
    train_model(
        model=model,
        optim=optimizer,
        data_loader=train_loader,
        lr_sch=scheduler,
        start_it=it,
        start_epoch=start_epoch,
        total_epochs=cfg.OPTIM.epochs,
        ckpt_save_dir=ckpt_dir,
        logger=logger,
        max_ckpt_save_num=5
    )

if __name__ == '__main__':
    main()
