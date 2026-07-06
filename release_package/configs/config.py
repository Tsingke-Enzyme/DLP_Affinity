
from dataclasses import dataclass, field
from typing import List, Optional
import json
from pathlib import Path

@dataclass
class ESMConfig:
    model_name: str = "facebook/esm2_t36_3B_UR50D"
    hidden_dim: int = 2560
    freeze_backbone: bool = True
    unfreeze_last_n_layers: int = 0
    use_mock: bool = False
    checkpoint_path: Optional[str] = None

@dataclass
class R2RConfig:
    compress_dim: int = 1
    output_dim: int = 32
    kan_hidden_dims_reduce: List[int] = field(default_factory=lambda: [512, 128, 32])
    kan_hidden_dims_inter: List[int] = field(default_factory=lambda: [256, 128])
    num_knots: int = 8
    pooling: str = 'mean'
    use_simple: bool = False

@dataclass
class GSPEConfig:
    num_projections: int = 64
    num_groups: int = 8
    trainable_projections: bool = False
    aggregation: str = 'mean'
    use_extended: bool = False
    use_statistics: bool = True
    use_attention: bool = False

@dataclass
class RegressorConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    use_batch_norm: bool = False

@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    apply_to_r2r: bool = True
    apply_to_gspe: bool = True
    apply_to_regressor: bool = True
    freeze_base: bool = True

@dataclass
class TrainingConfig:
    optimizer: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = 'linear_warmup'
    warmup_steps: int = 100
    max_steps: int = 10000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    use_huber_loss: bool = False
    huber_delta: float = 1.0
    use_correlation_loss: bool = False
    correlation_weight: float = 0.1
    use_weighted_loss: bool = False
    seed: int = 42
    fp16: bool = False
    max_grad_norm: float = 1.0

@dataclass
class MLMTrainingConfig:
    mask_prob: float = 0.15
    cdr_boost: float = 2.0
    max_seq_length: int = 512
    learning_rate: float = 1e-5
    warmup_steps: int = 40
    batch_size: int = 16
    num_epochs: int = 3
    num_gpus: int = 1

@dataclass
class DataConfig:
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    seq_ab_col: str = 'seq_ab'
    seq_ag_col: str = 'seq_ag'
    kd_col: str = 'kd'
    log_transform_kd: bool = True
    max_ab_len: int = 512
    max_ag_len: int = 512
    num_workers: int = 4
    use_stratified_sampler: bool = True

@dataclass
class DLPAffinityConfig:
    esm: ESMConfig = field(default_factory=ESMConfig)
    r2r: R2RConfig = field(default_factory=R2RConfig)
    gspe: GSPEConfig = field(default_factory=GSPEConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    experiment_name: str = 'dlp_affinity_default'
    output_dir: str = './outputs'
    log_dir: str = './logs'
    
    def save(self, path: str):
        config_dict = {
            'esm': self.esm.__dict__,
            'r2r': self.r2r.__dict__,
            'gspe': self.gspe.__dict__,
            'regressor': self.regressor.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'lora': self.lora.__dict__,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DLPAffinityConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        if 'esm' in config_dict: config.esm = ESMConfig(**config_dict['esm'])
        if 'r2r' in config_dict: config.r2r = R2RConfig(**config_dict['r2r'])
        if 'gspe' in config_dict: config.gspe = GSPEConfig(**config_dict['gspe'])
        if 'regressor' in config_dict: config.regressor = RegressorConfig(**config_dict['regressor'])
        if 'training' in config_dict: config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict: config.data = DataConfig(**config_dict['data'])
        if 'lora' in config_dict: config.lora = LoRAConfig(**config_dict['lora'])
        config.experiment_name = config_dict.get('experiment_name', 'dlp_affinity_default')
        config.output_dir = config_dict.get('output_dir', './outputs')
        config.log_dir = config_dict.get('log_dir', './logs')
        return config

def get_default_config() -> DLPAffinityConfig:
    return DLPAffinityConfig()

def get_small_config() -> DLPAffinityConfig:
    config = DLPAffinityConfig()
    config.esm.model_name = "facebook/esm2_t6_8M_UR50D"
    config.esm.hidden_dim = 320
    config.esm.use_mock = True
    config.r2r.kan_hidden_dims_reduce = [128, 32]
    config.r2r.kan_hidden_dims_inter = [64, 32]
    config.r2r.output_dim = 16
    config.r2r.use_simple = True
    config.gspe.num_projections = 32
    config.gspe.num_groups = 4
    config.regressor.hidden_dims = [64, 32]
    config.training.num_epochs = 10
    config.training.batch_size = 16
    return config

def get_3b_config() -> DLPAffinityConfig:
    config = DLPAffinityConfig()
    config.esm.model_name = "facebook/esm2_t36_3B_UR50D"
    config.esm.hidden_dim = 2560
    config.r2r.compress_dim = 1
    config.r2r.output_dim = 32
    config.r2r.pooling = 'mean'
    config.gspe.num_projections = 64
    config.gspe.num_groups = 8
    config.training.learning_rate = 1e-5
    config.training.warmup_steps = 40
    config.training.batch_size = 4
    config.training.gradient_accumulation_steps = 4
    return config

if __name__ == "__main__":
    print("Testing configurations...")
    default_config = get_default_config()
    print(f"Default ESM model: {default_config.esm.model_name}")
    print(f"Default R2R output dim: {default_config.r2r.output_dim}")
    print(f"Default GSPE groups: {default_config.gspe.num_groups}")
    
    config_path = '/tmp/test_config.json'
    default_config.save(config_path)
    print(f"\nSaved config to {config_path}")
    
    loaded_config = DLPAffinityConfig.load(config_path)
    print(f"Loaded ESM model: {loaded_config.esm.model_name}")
    
    small_config = get_small_config()
    print(f"\nSmall config ESM: {small_config.esm.model_name}")
    print(f"Small config hidden dim: {small_config.esm.hidden_dim}")
    
    big_config = get_3b_config()
    print(f"\n3B config ESM: {big_config.esm.model_name}")
    print(f"3B config learning rate: {big_config.training.learning_rate}")
