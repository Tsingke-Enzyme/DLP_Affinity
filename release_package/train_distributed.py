#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLP-Affinity 多GPU分布式训练脚本

================================================================================
启动命令
================================================================================

1. 单节点多GPU训练：
   torchrun --nproc_per_node=<GPU数量> train_distributed.py --config <配置文件路径>
   
   示例：
   torchrun --nproc_per_node=4 train_distributed.py --config configs/7KMG_dms.json

2. 多节点训练：
   # 在节点0上运行：
   torchrun --nnodes=<节点数> --node_rank=0 --nproc_per_node=<每节点GPU数> \
       --master_addr=<主节点IP> --master_port=<端口> \
       train_distributed.py --config <配置文件路径>
   
   # 在节点1上运行：
   torchrun --nnodes=<节点数> --node_rank=1 --nproc_per_node=<每节点GPU数> \
       --master_addr=<主节点IP> --master_port=<端口> \
       train_distributed.py --config <配置文件路径>

3. 使用Mock数据快速测试：
   torchrun --nproc_per_node=2 train_distributed.py --use_mock_data --num_mock_samples 100

================================================================================
输入参数说明
================================================================================

基本参数：
  --config              配置文件路径 (JSON格式)
  --use_small           使用小模型配置（测试用）
  --use_mock_data       使用模拟数据（测试用）
  --num_mock_samples    模拟数据样本数量 (默认: 200)

数据参数：
  --train_path          训练数据路径
  --val_path            验证数据路径

输出参数：
  --output_dir          输出根目录 (默认: ./outputs)
  --exp_name            实验名称，用于创建输出子目录
                        默认: <数据名称>_<时间戳>

模型参数：
  --esm_checkpoint      ESM2微调检查点路径
  --freeze_esm          冻结ESM2骨干网络参数

训练参数：
  --seed                随机种子 (默认: 42)

分布式参数（通常由torchrun自动设置）：
  --local_rank          本地进程序号（torchrun自动设置）
  --backend             通信后端: nccl或gloo (默认: nccl)

================================================================================
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DLPAffinity, DLPAffinityLoss
from data import AffinityDataset, create_mock_data, collate_fn
from configs import DLPAffinityConfig, get_default_config, get_small_config


# ============================================================================
# 分布式训练工具函数
# ============================================================================

def setup_distributed(backend: str = 'nccl'):
    """
    初始化分布式训练环境
    
    Args:
        backend: 通信后端 ('nccl' 用于GPU, 'gloo' 用于CPU或调试)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """检查是否为主进程 (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """获取当前进程序号"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def set_seed(seed: int, rank: int = 0):
    """设置随机种子（考虑进程序号以保证各进程数据不同）"""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str, experiment_name: str, rank: int) -> logging.Logger:
    """设置日志（仅主进程输出到文件和控制台）"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    if rank == 0:
        log_file = os.path.join(
            log_dir,
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        # 非主进程使用空handler
        logger.addHandler(logging.NullHandler())
    
    return logger


import math

def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """创建带有线性预热的学习率调度器"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5
):
    """创建带有余弦预热的学习率调度器"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 分布式数据加载器
# ============================================================================

def create_distributed_dataloader(
    dataset: AffinityDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True
) -> tuple:
    """
    创建分布式数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 每个GPU的批次大小
        shuffle: 是否打乱（DistributedSampler控制）
        num_workers: 工作进程数
        drop_last: 是否丢弃最后不完整的batch
    
    Returns:
        (dataloader, sampler) 元组
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=drop_last
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, sampler


# ============================================================================
# 分布式训练器
# ============================================================================

class DistributedTrainer:
    """DLP-Affinity 分布式训练器"""
    
    def __init__(
        self,
        model: DLPAffinity,
        config: DLPAffinityConfig,
        train_dataloader,
        train_sampler,
        val_dataloader=None,
        local_rank: int = 0
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.val_dataloader = val_dataloader
        self.local_rank = local_rank
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = torch.device(f'cuda:{local_rank}')
        
        # 设置日志
        self.logger = setup_logging(
            config.log_dir,
            config.experiment_name,
            self.rank
        )
        
        # 移动模型到设备并包装DDP
        model.to(self.device)
        self.model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # ESM2懒加载可能导致未使用参数
        )
        
        # 设置损失函数
        self.loss_fn = DLPAffinityLoss(
            use_huber=config.training.use_huber_loss,
            huber_delta=config.training.huber_delta,
            use_correlation_loss=config.training.use_correlation_loss,
            correlation_weight=config.training.correlation_weight
        )
        
        # 设置优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 计算总训练步数（考虑梯度累积和分布式）
        num_training_steps = (
            len(train_dataloader) *
            config.training.num_epochs //
            config.training.gradient_accumulation_steps
        )
        
        # 设置学习率调度器
        if getattr(config.training, 'scheduler', 'linear_warmup') == 'cosine_warmup':
            if is_main_process():
                print("Using Cosine Warmup Scheduler")
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.training.warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.training.warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录（仅主进程）
        if is_main_process():
            os.makedirs(config.output_dir, exist_ok=True)
        
        # 等待所有进程
        if dist.is_initialized():
            dist.barrier()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 设置sampler的epoch（确保每个epoch数据打乱不同）
        self.train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_samples = 0
        
        # 仅主进程显示进度条
        if is_main_process():
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                leave=False
            )
        else:
            progress_bar = self.train_dataloader
        
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            seq_ab = batch['seq_ab']
            seq_ag = batch['seq_ag']
            y_true = batch['kd'].to(self.device)
            
            # 获取权重
            sample_weights = None
            if getattr(self.config.training, 'use_weighted_loss', False) and 'sample_weight' in batch:
                sample_weights = batch['sample_weight'].to(self.device)
            
            # 前向传播
            y_pred = self.model(seq_ab=seq_ab, seq_ag=seq_ag)
            
            # 计算损失
            losses = self.loss_fn(y_pred, y_true, sample_weights=sample_weights)
            loss = losses['total'] / self.config.training.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            accumulated_loss += loss.item()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                # 更新参数
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # 记录损失
                total_loss += accumulated_loss * self.config.training.gradient_accumulation_steps
                accumulated_loss = 0.0
                
                # 更新进度条（仅主进程）
                if is_main_process() and hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * self.config.training.gradient_accumulation_steps:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # 评估和保存（仅主进程）
                if self.global_step % self.config.training.eval_steps == 0:
                    if self.val_dataloader is not None:
                        val_metrics = self.evaluate()
                        self.logger.info(
                            f"Step {self.global_step}: "
                            f"Val Loss = {val_metrics['loss']:.4f}"
                        )
                        self.model.train()
                
                # 不再按步数保存检查点（仅保留best和最后一轮）
                # if self.global_step % self.config.training.save_steps == 0:
                #     self.save_checkpoint(f"checkpoint_{self.global_step}")
            
            total_samples += len(seq_ab)
        
        avg_loss = total_loss / len(self.train_dataloader)
        
        return {
            'loss': avg_loss,
            'samples': total_samples
        }
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                seq_ab = batch['seq_ab']
                seq_ag = batch['seq_ag']
                y_true = batch['kd'].to(self.device)
                
                y_pred = self.model(seq_ab=seq_ab, seq_ag=seq_ag)
                
                losses = self.loss_fn(y_pred, y_true)
                total_loss += losses['total'].item() * len(seq_ab)
                
                all_predictions.extend(y_pred.cpu().numpy())
                all_targets.extend(y_true.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(self.val_dataloader.dataset)
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Pearson 相关系数
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        return {
            'loss': avg_loss,
            'correlation': correlation,
            'rmse': rmse,
            'mae': mae
        }
    
    def train(self):
        """完整训练流程"""
        self.logger.info("Starting distributed training...")
        self.logger.info(f"Config: {self.config.experiment_name}")
        self.logger.info(f"World size: {self.world_size}")
        self.logger.info(f"Local rank: {self.local_rank}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch}/{self.config.training.num_epochs}")
            self.logger.info(f"{'='*50}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # 验证
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                self.logger.info(
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Correlation: {val_metrics['correlation']:.4f}, "
                    f"RMSE: {val_metrics['rmse']:.4f}"
                )
                
                # 保存最佳模型（仅主进程）
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model')
                    self.logger.info("Saved best model!")
            
            # 只保存最后一轮（best_model在上面验证时保存）
            if epoch == self.config.training.num_epochs:
                self.save_checkpoint('last_epoch')
            
            # 同步所有进程
            if dist.is_initialized():
                dist.barrier()
        
        self.logger.info("\nTraining completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, name: str):
        """保存检查点（仅主进程）"""
        if not is_main_process():
            return
        
        checkpoint_path = os.path.join(self.config.output_dir, f"{name}.pt")
        
        # 获取原始模型（去除DDP包装）
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载到原始模型（去除DDP包装）
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from {path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='DLP-Affinity 多GPU分布式训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单节点4GPU训练
  torchrun --nproc_per_node=4 train_distributed.py --config configs/7KMG_dms.json
  
  # 使用Mock数据测试
  torchrun --nproc_per_node=2 train_distributed.py --use_mock_data
        """
    )
    
    # 基本参数
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (JSON格式)'
    )
    parser.add_argument(
        '--use_small',
        action='store_true',
        help='使用小模型配置（测试用）'
    )
    parser.add_argument(
        '--use_mock_data',
        action='store_true',
        help='使用模拟数据（测试用）'
    )
    parser.add_argument(
        '--num_mock_samples',
        type=int,
        default=200,
        help='模拟数据样本数量 (默认: 200)'
    )
    
    # 数据参数
    parser.add_argument(
        '--train_path',
        type=str,
        default=None,
        help='训练数据路径'
    )
    parser.add_argument(
        '--val_path',
        type=str,
        default=None,
        help='验证数据路径'
    )
    
    # 输出参数
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='输出根目录 (默认: ./outputs)'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='实验名称，用于创建输出子目录。默认: <数据名称>_<时间戳>'
    )
    
    # 模型参数
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--esm_checkpoint',
        type=str,
        default=None,
        help='ESM2微调检查点路径'
    )
    parser.add_argument(
        '--freeze_esm',
        action='store_true',
        help='冻结ESM2骨干网络参数'
    )
    
    # 分布式参数
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='本地进程序号（torchrun自动设置）'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        choices=['nccl', 'gloo'],
        help='分布式通信后端 (默认: nccl)'
    )
    
    args = parser.parse_args()
    
    # 初始化分布式环境
    local_rank = setup_distributed(backend=args.backend)
    
    try:
        # 加载配置
        if args.config:
            config = DLPAffinityConfig.load(args.config)
        elif args.use_small:
            config = get_small_config()
        else:
            config = get_default_config()
        
        # 更新配置
        if args.train_path:
            config.data.train_path = args.train_path
        if args.val_path:
            config.data.val_path = args.val_path
        config.training.seed = args.seed
        
        # 生成输出目录名称
        if args.exp_name:
            exp_name = args.exp_name
        else:
            # 默认: 数据名称 + 时间戳
            if args.config:
                data_name = Path(args.config).stem
            elif config.data.train_path:
                data_name = Path(config.data.train_path).stem
            else:
                data_name = "mock_data"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            exp_name = f"{data_name}_{timestamp}"
        
        config.experiment_name = exp_name
        config.output_dir = os.path.join(args.output_dir, exp_name)
        
        # 设置随机种子（考虑rank）
        set_seed(config.training.seed, get_rank())
        
        # 准备数据
        if args.use_mock_data:
            if is_main_process():
                print("Using mock data for testing...")
            mock_data = create_mock_data(num_samples=args.num_mock_samples)
            
            # 分割训练和验证集
            split_idx = int(len(mock_data) * 0.8)
            train_data = mock_data[:split_idx]
            val_data = mock_data[split_idx:]
            
            train_dataset = AffinityDataset(data_list=train_data)
            val_dataset = AffinityDataset(data_list=val_data)
        else:
            if config.data.train_path is None:
                raise ValueError("Must provide --train_path or use --use_mock_data")
            
            train_dataset = AffinityDataset(
                data_path=config.data.train_path,
                log_transform_kd=config.data.log_transform_kd
            )
            
            val_dataset = None
            if config.data.val_path:
                val_dataset = AffinityDataset(
                    data_path=config.data.val_path,
                    log_transform_kd=config.data.log_transform_kd
                )
        
        # 创建分布式数据加载器
        train_dataloader, train_sampler = create_distributed_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        val_dataloader = None
        if val_dataset:
            # 验证集不需要分布式（仅主进程评估）
            from data import create_dataloader
            val_dataloader = create_dataloader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                use_stratified_sampler=False
            )
        
        if is_main_process():
            print(f"Train samples: {len(train_dataset)}")
            print(f"Train samples per GPU: {len(train_dataset) // get_world_size()}")
            if val_dataset:
                print(f"Val samples: {len(val_dataset)}")
            print(f"Output directory: {config.output_dir}")
        
        # 确定是否冻结ESM
        freeze_esm = config.esm.freeze_backbone or args.freeze_esm
        
        # 创建模型
        model = DLPAffinity(
            esm_model_name=config.esm.model_name,
            esm_hidden_dim=config.esm.hidden_dim,
            freeze_esm=freeze_esm,
            esm_unfreeze_last_n_layers=getattr(config.esm, 'unfreeze_last_n_layers', 0),
            use_mock_esm=config.esm.use_mock,
            esm_checkpoint_path=args.esm_checkpoint,
            r2r_compress_dim=config.r2r.compress_dim,
            r2r_out_dim=config.r2r.output_dim,
            r2r_pooling=config.r2r.pooling,
            use_simple_r2r=config.r2r.use_simple,
            kan_hidden_dims_reduce=config.r2r.kan_hidden_dims_reduce,
            kan_hidden_dims_inter=config.r2r.kan_hidden_dims_inter,
            num_knots=config.r2r.num_knots,
            gspe_num_projections=config.gspe.num_projections,
            gspe_num_groups=config.gspe.num_groups,
            use_extended_gspe=config.gspe.use_extended,
            gspe_trainable_projections=config.gspe.trainable_projections,
            gspe_aggregation=config.gspe.aggregation,
            gspe_use_statistics=config.gspe.use_statistics,
            gspe_use_attention=config.gspe.use_attention,
            regressor_hidden_dims=config.regressor.hidden_dims,
            regressor_dropout=config.regressor.dropout,
            regressor_use_batch_norm=config.regressor.use_batch_norm
        )
        
        if args.esm_checkpoint and is_main_process():
            print(f"Loaded ESM2 checkpoint from: {args.esm_checkpoint}")
        
        # ============ 应用 LoRA（如果启用）============
        if config.lora.enabled:
            from models.lora import LoRAConfig as ModelLoRAConfig
            
            lora_config = ModelLoRAConfig(
                r=config.lora.r,
                alpha=config.lora.alpha,
                dropout=config.lora.dropout
            )
            
            model.apply_lora(
                lora_config=lora_config,
                freeze_base=config.lora.freeze_base,
                apply_to_r2r=config.lora.apply_to_r2r,
                apply_to_gspe=config.lora.apply_to_gspe,
                apply_to_regressor=config.lora.apply_to_regressor
            )
            
            if is_main_process():
                stats = model.count_parameters()
                print(f"\\nLoRA Training Mode Enabled!")
                print(f"  - LoRA rank: {config.lora.r}")
                print(f"  - LoRA alpha: {config.lora.alpha}")
                print(f"  - Trainable params: {stats['trainable']:,}")
                print(f"  - Frozen params: {stats['frozen']:,}")
        
        # 创建分布式训练器
        trainer = DistributedTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            train_sampler=train_sampler,
            val_dataloader=val_dataloader,
            local_rank=local_rank
        )
        
        # 开始训练
        trainer.train()
        
    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == "__main__":
    main()
