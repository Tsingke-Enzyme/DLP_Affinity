"""
ESM2 MLM 微调脚本
在抗体数据集上进行掩码语言模型微调
支持 CDR 优先掩码
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import MLMDataset


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


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


class MLMCollator:
    """MLM 批次整理器"""
    
    def __init__(self, pad_token_id: int = 1):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict:
        # 找到最大长度
        max_len = max(len(item['input_ids']) for item in batch)
        
        # 初始化张量
        input_ids = torch.full(
            (len(batch), max_len),
            self.pad_token_id,
            dtype=torch.long
        )
        labels = torch.full(
            (len(batch), max_len),
            -100,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (len(batch), max_len),
            dtype=torch.long
        )
        
        # 填充
        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids[i, :seq_len] = item['input_ids']
            labels[i, :seq_len] = item['labels']
            attention_mask[i, :seq_len] = item['attention_mask']
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


class MLMTrainer:
    """ESM2 MLM 微调训练器"""
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-5,
        warmup_steps: int = 40,
        num_epochs: int = 3,
        device: str = 'cuda',
        output_dir: str = './outputs/mlm',
        log_dir: str = './logs'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.output_dir = output_dir
        
        # 设置日志
        self.logger = setup_logging(log_dir, 'esm2_mlm_finetune')
        
        # 移动模型到设备
        self.model.to(device)
        
        # 设置优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 计算总训练步数
        num_training_steps = len(train_dataloader) * num_epochs
        
        # 设置学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        for batch in progress_bar:
            # 获取数据
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 计算损失
            # outputs 形状: [batch, seq_len, vocab_size]
            logits = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
            
            # 如果模型有 LM head
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(logits)
            
            # 计算交叉熵损失
            # 需要将 logits 和 labels 展平
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            mask = labels != -100
            predictions = logits.argmax(dim=-1)
            correct = ((predictions == labels) & mask).sum().item()
            total_correct += correct
            total_masked += mask.sum().item()
            
            # 更新进度条
            accuracy = total_correct / max(1, total_masked)
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = total_correct / max(1, total_masked)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
                
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(logits)
                
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                total_loss += loss.item()
                
                mask = labels != -100
                predictions = logits.argmax(dim=-1)
                correct = ((predictions == labels) & mask).sum().item()
                total_correct += correct
                total_masked += mask.sum().item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / max(1, total_masked)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self):
        """完整训练流程"""
        self.logger.info("Starting MLM fine-tuning...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch}/{self.num_epochs}")
            self.logger.info(f"{'='*50}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics['accuracy']:.4f}"
            )
            
            # 验证
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                self.logger.info(
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Accuracy: {val_metrics['accuracy']:.4f}"
                )
                
                # 保存最佳模型
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model')
                    self.logger.info("Saved best model!")
            
            # 定期保存
            self.save_checkpoint(f'epoch_{epoch}')
        
        self.logger.info("\nMLM fine-tuning completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, f"{name}.pt")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_sequences_from_file(file_path: str) -> List[str]:
    """从文件加载序列"""
    sequences = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                sequences.append(line)
    
    return sequences


def create_mock_sequences(
    num_sequences: int = 1000,
    min_length: int = 80,
    max_length: int = 150
) -> List[str]:
    """创建模拟序列用于测试"""
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    sequences = []
    
    for _ in range(num_sequences):
        length = random.randint(min_length, max_length)
        seq = ''.join(random.choices(aa_list, k=length))
        sequences.append(seq)
    
    return sequences


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ESM2 MLM Fine-tuning')
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/esm2_t6_8M_UR50D',
        help='ESM2 model name'
    )
    parser.add_argument(
        '--sequences_file',
        type=str,
        default=None,
        help='Path to sequences file'
    )
    parser.add_argument(
        '--use_mock',
        action='store_true',
        help='Use mock sequences for testing'
    )
    parser.add_argument(
        '--num_mock_sequences',
        type=int,
        default=1000,
        help='Number of mock sequences'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=40,
        help='Warmup steps'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of epochs'
    )
    parser.add_argument(
        '--mask_prob',
        type=float,
        default=0.15,
        help='Mask probability'
    )
    parser.add_argument(
        '--cdr_boost',
        type=float,
        default=2.0,
        help='CDR mask probability boost'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/mlm',
        help='Output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载或创建序列
    if args.use_mock:
        print("Using mock sequences for testing...")
        sequences = create_mock_sequences(args.num_mock_sequences)
    elif args.sequences_file:
        print(f"Loading sequences from {args.sequences_file}...")
        sequences = load_sequences_from_file(args.sequences_file)
    else:
        raise ValueError("Must provide --sequences_file or use --use_mock")
    
    print(f"Total sequences: {len(sequences)}")
    
    # 分割训练和验证集
    split_idx = int(len(sequences) * 0.9)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    
    # 创建数据集
    train_dataset = MLMDataset(
        sequences=train_sequences,
        mask_prob=args.mask_prob,
        cdr_boost=args.cdr_boost
    )
    
    val_dataset = MLMDataset(
        sequences=val_sequences,
        mask_prob=args.mask_prob,
        cdr_boost=args.cdr_boost
    )
    
    # 创建数据加载器
    collator = MLMCollator()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    # 加载模型
    print(f"Loading model: {args.model_name}")
    
    try:
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    except ImportError:
        print("transformers not installed, using mock model")
        from models import MockESM2Encoder
        model = MockESM2Encoder(hidden_dim=320)
    
    # 创建训练器
    trainer = MLMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_epochs=args.num_epochs,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
