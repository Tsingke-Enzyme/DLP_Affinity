
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import DLPAffinity, DLPAffinityLoss
from data import AffinityDataset, create_dataloader, create_mock_data
from configs import DLPAffinityConfig, get_default_config, get_small_config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger(__name__)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

class Trainer:
    def __init__(self, model: DLPAffinity, config: DLPAffinityConfig, train_dataloader, val_dataloader=None, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.logger = setup_logging(config.log_dir, config.experiment_name)
        self.model.to(device)
        self.loss_fn = DLPAffinityLoss(
            use_huber=config.training.use_huber_loss,
            huber_delta=config.training.huber_delta,
            use_correlation_loss=config.training.use_correlation_loss,
            correlation_weight=config.training.correlation_weight
        )
        self.optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        num_training_steps = (len(train_dataloader) * config.training.num_epochs // config.training.gradient_accumulation_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=config.training.warmup_steps, num_training_steps=num_training_steps)
        self.global_step = 0
        self.best_val_loss = float('inf')
        os.makedirs(config.output_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", leave=False)
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(progress_bar):
            seq_ab = batch['seq_ab']
            seq_ag = batch['seq_ag']
            y_true = batch['kd'].to(self.device)
            
            y_pred = self.model(seq_ab=seq_ab, seq_ag=seq_ag)
            losses = self.loss_fn(y_pred, y_true)
            loss = losses['total'] / self.config.training.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                total_loss += accumulated_loss * self.config.training.gradient_accumulation_steps
                accumulated_loss = 0.0
                progress_bar.set_postfix({'loss': f"{loss.item() * self.config.training.gradient_accumulation_steps:.4f}", 'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"})
                
                if self.global_step % self.config.training.eval_steps == 0:
                    if self.val_dataloader is not None:
                        val_metrics = self.evaluate()
                        self.logger.info(f"Step {self.global_step}: Val Loss = {val_metrics['loss']:.4f}")
                        self.model.train()
                if self.global_step % self.config.training.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}")
            total_samples += len(seq_ab)
        return {'loss': total_loss / len(self.train_dataloader), 'samples': total_samples}
    
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                seq_ab = batch['seq_ab']
                seq_ag = batch['seq_ag']
                y_true = batch['kd'].to(self.device)
                y_pred = self.model(seq_ab=seq_ab, seq_ag=seq_ag)
                losses = self.loss_fn(y_pred, y_true)
                total_loss += losses['total'].item() * len(seq_ab)
                all_predictions.extend(y_pred.cpu().numpy())
                all_targets.extend(y_true.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_dataloader.dataset)
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        correlation = np.corrcoef(predictions, targets)[0, 1]
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mae = np.mean(np.abs(predictions - targets))
        return {'loss': avg_loss, 'correlation': correlation, 'rmse': rmse, 'mae': mae}
    
    def train(self):
        self.logger.info(f"Starting training... Config: {self.config.experiment_name}, Device: {self.device}")
        for epoch in range(1, self.config.training.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}")
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Correlation: {val_metrics['correlation']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model')
                    self.logger.info("Saved best model!")
            self.save_checkpoint(f'epoch_{epoch}')
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, name: str):
        path = os.path.join(self.config.output_dir, f"{name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)
        self.logger.info(f"Saved checkpoint: {path}")

def main():
    parser = argparse.ArgumentParser(description='Train DLP-Affinity model')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--use_small', action='store_true')
    parser.add_argument('--use_mock_data', action='store_true')
    parser.add_argument('--num_mock_samples', type=int, default=200)
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--esm_checkpoint', type=str, default=None)
    parser.add_argument('--freeze_esm', action='store_true')
    args = parser.parse_args()
    
    if args.config: config = DLPAffinityConfig.load(args.config)
    elif args.use_small: config = get_small_config()
    else: config = get_default_config()
    
    if args.train_path: config.data.train_path = args.train_path
    if args.val_path: config.data.val_path = args.val_path
    config.training.seed = args.seed
    
    if args.exp_name:
        exp_name = args.exp_name
    else:
        data_name = Path(args.config).stem if args.config else (Path(config.data.train_path).stem if config.data.train_path else "mock_data")
        exp_name = f"{data_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config.experiment_name = exp_name
    config.output_dir = os.path.join(args.output_dir, exp_name)
    set_seed(config.training.seed)
    
    if args.use_mock_data:
        print("Using mock data...")
        mock_data = create_mock_data(num_samples=args.num_mock_samples)
        split_idx = int(len(mock_data) * 0.8)
        train_dataset = AffinityDataset(data_list=mock_data[:split_idx])
        val_dataset = AffinityDataset(data_list=mock_data[split_idx:])
    else:
        if config.data.train_path is None: raise ValueError("Must provide --train_path or use --use_mock_data")
        train_dataset = AffinityDataset(data_path=config.data.train_path, log_transform_kd=config.data.log_transform_kd)
        val_dataset = AffinityDataset(data_path=config.data.val_path, log_transform_kd=config.data.log_transform_kd) if config.data.val_path else None
    
    train_dataloader = create_dataloader(train_dataset, batch_size=config.training.batch_size, shuffle=True, use_stratified_sampler=config.data.use_stratified_sampler)
    val_dataloader = create_dataloader(val_dataset, batch_size=config.training.batch_size, shuffle=False, use_stratified_sampler=False) if val_dataset else None
    
    model = DLPAffinity(
        esm_model_name=config.esm.model_name,
        esm_hidden_dim=config.esm.hidden_dim,
        freeze_esm=config.esm.freeze_backbone or args.freeze_esm,
        use_mock_esm=config.esm.use_mock,
        esm_checkpoint_path=args.esm_checkpoint,
        r2r_compress_dim=config.r2r.compress_dim,
        r2r_out_dim=config.r2r.output_dim,
        r2r_pooling=config.r2r.pooling,
        use_simple_r2r=config.r2r.use_simple,
        gspe_num_projections=config.gspe.num_projections,
        gspe_num_groups=config.gspe.num_groups,
        use_extended_gspe=config.gspe.use_extended,
        regressor_hidden_dims=config.regressor.hidden_dims,
        regressor_dropout=config.regressor.dropout
    )
    
    Trainer(model=model, config=config, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=args.device).train()

if __name__ == "__main__":
    main()
