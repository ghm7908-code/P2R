import glob
import tqdm
import os
import torch
import numpy as np
from test_util import test_model
import datetime


def load_data_to_gpu(batch_dict):
    """
    专为Building3D格式设计的数据加载到GPU函数
    """
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            if val.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                batch_dict[key] = val.long().cuda(non_blocking=True)
            else:
                batch_dict[key] = val.float().cuda(non_blocking=True)
        elif isinstance(val, np.ndarray):
            tensor = torch.from_numpy(val)
            if np.issubdtype(val.dtype, np.integer):
                batch_dict[key] = tensor.long().cuda(non_blocking=True)
            else:
                batch_dict[key] = tensor.float().cuda(non_blocking=True)
        elif isinstance(val, list):
            # 列表类型，检查元素类型
            if len(val) > 0 and isinstance(val[0], np.ndarray):
                # 如果是NumPy数组列表，转换为Tensor列表
                batch_dict[key] = [
                    torch.from_numpy(v).long().cuda(non_blocking=True)
                    if np.issubdtype(v.dtype, np.integer)
                    else torch.from_numpy(v).float().cuda(non_blocking=True)
                    for v in val
                ]
            elif len(val) > 0 and isinstance(val[0], torch.Tensor):
                # 如果是Tensor列表，直接移动到GPU
                batch_dict[key] = [
                    v.long().cuda(non_blocking=True)
                    if v.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                    else v.float().cuda(non_blocking=True)
                    for v in val
                ]
    return batch_dict


def train_one_epoch(model, optim, data_loader, accumulated_iter,
                    tbar, leave_pbar=False, logger=None):
    """
    单epoch训练函数，适配Building3D格式
    """
    total_it_each_epoch = len(data_loader)
    dataloader_iter = iter(data_loader)
    
    pbar = tqdm.tqdm(
        total=total_it_each_epoch, 
        leave=leave_pbar, 
        desc='train', 
        dynamic_ncols=True
    )

    epoch_loss = 0
    loss_dict_accum = {}
    
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch = next(dataloader_iter)
            if logger:
                logger.info('数据加载器重新迭代')
            else:
                print('new iters')

        try:
            cur_lr = float(optim.lr)
        except:
            cur_lr = optim.param_groups[0]['lr']

        model.train()
        optim.zero_grad()
        
        # 加载数据到GPU
        batch = load_data_to_gpu(batch)
        
        # 前向传播（Building3D格式返回三个值）
        loss, loss_dict, disp_dict = model(batch)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()

        accumulated_iter += 1
        epoch_loss += loss.item()
        
        # 累加各项损失
        for key, value in loss_dict.items():
            if key not in loss_dict_accum:
                loss_dict_accum[key] = 0
            loss_dict_accum[key] += value
        
        # 更新显示字典
        disp_dict.update(loss_dict)
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # 更新进度条
        pbar.update()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{cur_lr:.6f}'
        })
        
        # 更新tbar
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        # 每50个batch记录一次
        if cur_it % 50 == 0 and logger:
            logger.info(f'Epoch iter {cur_it}/{total_it_each_epoch}: '
                       f'loss={loss.item():.4f}, lr={cur_lr:.6f}')

    pbar.close()
    
    # 计算平均损失
    avg_loss = epoch_loss / total_it_each_epoch if total_it_each_epoch > 0 else 0
    avg_loss_dict = {k: v/total_it_each_epoch for k, v in loss_dict_accum.items()}
    
    return accumulated_iter, avg_loss, avg_loss_dict


def train_model(model, optim, data_loader, lr_sch, start_it, start_epoch, total_epochs, 
                ckpt_save_dir, logger=None, sampler=None, max_ckpt_save_num=5):
    """
    主训练函数，适配Building3D格式
    """
    from pathlib import Path
    
    # 确保检查点目录存在
    ckpt_save_dir = Path(ckpt_save_dir)
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 记录训练开始信息
    if logger:
        logger.info('=' * 60)
        logger.info(f'开始训练，总轮数: {total_epochs}')
        logger.info(f'起始轮数: {start_epoch}')
        logger.info(f'检查点保存目录: {ckpt_save_dir}')
        logger.info('=' * 60)
    
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True) as tbar:
        accumulated_iter = start_it
        
        for e in tbar:
            # 设置分布式采样器epoch（如果使用分布式）
            if sampler is not None:
                sampler.set_epoch(e)
            
            edge_start_epoch = int(getattr(model, 'model_cfg', {}).get('edge_start_epoch', 6))
            if e >= edge_start_epoch:
                model.use_edge = True
                if logger:
                    logger.info(f'Epoch {e+1}: 启用边处理网络')
            
            # 训练一个epoch
            accumulated_iter, avg_loss, avg_loss_dict = train_one_epoch(
                model, optim, data_loader, accumulated_iter, tbar,
                leave_pbar=(e + 1 == total_epochs),
                logger=logger
            )
            
            # 更新学习率
            lr_sch.step()
            
            # 确保学习率不为0
            lr = max(optim.param_groups[0]['lr'], 1e-6)
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            
            # 记录epoch结果
            if logger:
                logger.info(f'Epoch {e+1} 完成')
                logger.info(f'平均损失: {avg_loss:.4f}')
                logger.info(f'学习率: {lr:.6f}')
                for key, value in avg_loss_dict.items():
                    logger.info(f'  {key}: {value:.4f}')
            
            # 管理检查点（保留最新的几个）
            ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)
            
            if ckpt_list.__len__() >= max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])
                    if logger:
                        logger.info(f'删除旧检查点: {ckpt_list[cur_file_idx]}')
            
            # 保存检查点
            ckpt_name = ckpt_save_dir / f'checkpoint_epoch_{e + 1}'
            
            # 获取模型输入通道数（用于后续恢复）
            input_channel = getattr(model, 'input_channel', 3)
            
            save_checkpoint(
                checkpoint_state(model, optim, e + 1, accumulated_iter, 
                               avg_loss=avg_loss, input_channel=input_channel), 
                filename=ckpt_name,
                logger=logger
            )
            
            if logger:
                logger.info(f'检查点保存: {ckpt_name}.pth')
            
            # 每5个epoch进行一次验证（如果有验证集）
            if (e + 1) % 5 == 0:
                if hasattr(model, 'validate') and hasattr(data_loader, 'dataset'):
                    try:
                        val_loss = model.validate()
                        if logger:
                            logger.info(f'验证损失: {val_loss:.4f}')
                    except:
                        if logger:
                            logger.warning('验证失败，跳过')
        
        if logger:
            logger.info('训练完成！')


def model_state_to_cpu(model_state):
    """将模型状态转移到CPU"""
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu() if isinstance(val, torch.Tensor) else val
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, 
                    avg_loss=None, input_channel=3):
    """
    创建检查点状态，专为Building3D格式优化
    """
    optim_state = optimizer.state_dict() if optimizer is not None else None
    
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    
    # 构建检查点字典
    checkpoint = {
        'epoch': epoch,
        'it': it,
        'model_state': model_state,
        'optimizer_state': optim_state,
        'avg_loss': avg_loss,
        'input_channel': input_channel,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存模型配置（如果存在）
    if hasattr(model, 'model_cfg'):
        checkpoint['model_cfg'] = model.model_cfg
    
    # 保存数据格式信息
    if hasattr(model, 'use_building3d_format'):
        checkpoint['use_building3d_format'] = model.use_building3d_format
    
    return checkpoint


def save_checkpoint(state, filename='checkpoint', logger=None):
    """
    保存检查点，专为Building3D格式优化
    """
    import pickle
    
    # 分离优化器状态（可选）
    if 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        # 保留在同一个文件中，不再分离
        
    # 保存主文件
    filename = f'{filename}.pth'
    
    try:
        # 尝试使用torch的保存功能
        torch.save(state, filename)
        
        if logger:
            logger.info(f'检查点保存成功: {filename}')
            logger.info(f'  轮数: {state.get("epoch", "N/A")}')
            logger.info(f'  损失: {state.get("avg_loss", "N/A")}')
            logger.info(f'  输入通道: {state.get("input_channel", 3)}')
    except Exception as e:
        if logger:
            logger.error(f'保存检查点失败: {e}')
        raise

# 在train_utils.py中添加以下函数

def validate_data_format(data_loader, logger=None):
    """
    验证数据加载器输出的格式是否符合Building3D要求
    """
    logger = logger or print
    
    # 获取一个批次的数据
    data_iter = iter(data_loader)
    batch = next(data_iter)
    
    logger('=' * 60)
    logger('数据格式验证:')
    logger('=' * 60)
    
    # 检查关键字段
    required_keys = ['point_clouds', 'wf_vertices', 'wf_edges']
    optional_keys = ['wf_centers', 'wf_edges_vertices', 'scan_idx']
    
    missing_keys = []
    for key in required_keys:
        if key not in batch:
            missing_keys.append(key)
    
    if missing_keys:
        logger(f'错误: 缺少必要字段: {missing_keys}')
        return False
    
    # 检查数据类型和形状
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger(f'{key}: Tensor, shape={value.shape}, dtype={value.dtype}')
        elif isinstance(value, np.ndarray):
            logger(f'{key}: ndarray, shape={value.shape}, dtype={value.dtype}')
        elif isinstance(value, list):
            logger(f'{key}: list, length={len(value)}')
            if len(value) > 0:
                logger(f'  第一个元素类型: {type(value[0])}')
        else:
            logger(f'{key}: {type(value)}')
    
    # 检查点云数据
    if 'point_clouds' in batch:
        points = batch['point_clouds']
        if isinstance(points, torch.Tensor):
            logger(f'点云形状: {points.shape}')
            logger(f'点云范围: x[{points[...,0].min():.3f}, {points[...,0].max():.3f}] '
                  f'y[{points[...,1].min():.3f}, {points[...,1].max():.3f}] '
                  f'z[{points[...,2].min():.3f}, {points[...,2].max():.3f}]')
    
    # 检查标签数据
    if 'wf_vertices' in batch:
        vertices = batch['wf_vertices']
        if isinstance(vertices, torch.Tensor):
            logger(f'顶点数量: {vertices.shape[1] if len(vertices.shape) > 1 else vertices.shape[0]}')
    
    logger('=' * 60)
    return True


def build_training_config(model, data_loader, logger=None):
    """
    根据数据和模型构建训练配置
    """
    logger = logger or print
    
    config = {
        'model_name': model.__class__.__name__,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'data_format': 'Building3D',
        'input_channels': getattr(model, 'input_channel', 3),
        'use_edge': getattr(model, 'use_edge', False)
    }
    
    # 获取数据信息
    if hasattr(data_loader, 'dataset'):
        config['dataset_size'] = len(data_loader.dataset)
        config['batch_size'] = data_loader.batch_size
        config['num_batches'] = len(data_loader)
    
    logger('训练配置:')
    logger(f'  模型: {config["model_name"]}')
    logger(f'  总参数: {config["total_params"]:,}')
    logger(f'  可训练参数: {config["trainable_params"]:,}')
    logger(f'  数据格式: {config["data_format"]}')
    logger(f'  输入通道: {config["input_channels"]}')
    logger(f'  使用边处理: {config["use_edge"]}')
    
    if 'dataset_size' in config:
        logger(f'  数据集大小: {config["dataset_size"]}')
        logger(f'  批大小: {config["batch_size"]}')
        logger(f'  批次数量: {config["num_batches"]}')
    
    return config
