
import torch
import torch.nn as nn
from typing import List, Tuple, Union

from .esm_encoder import ESM2Encoder, MockESM2Encoder
from .r2r import R2RModule, R2RModuleSimple
from .gspe import GSPEModule, GSPEModuleExtended
from .lora import (
    LoRAConfig, apply_lora_to_linear, get_lora_parameters, 
    get_non_lora_parameters, freeze_non_lora_parameters,
    merge_lora_weights, unmerge_lora_weights
)

class AffinityRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int] = None, dropout: float = 0.1, use_batch_norm: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)

class DLPAffinity(nn.Module):
    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t36_3B_UR50D",
        esm_hidden_dim: int = 2560,
        freeze_esm: bool = False,
        esm_unfreeze_last_n_layers: int = 0,
        use_mock_esm: bool = False,
        esm_checkpoint_path: str = None,
        r2r_compress_dim: int = 1,
        r2r_out_dim: int = 32,
        r2r_pooling: str = 'mean',
        use_simple_r2r: bool = False,
        kan_hidden_dims_reduce: List[int] = None,
        kan_hidden_dims_inter: List[int] = None,
        num_knots: int = 8,
        gspe_num_projections: int = 64,
        gspe_num_groups: int = 8,
        use_extended_gspe: bool = False,
        gspe_trainable_projections: bool = False,
        gspe_aggregation: str = 'mean',
        gspe_use_statistics: bool = True,
        gspe_use_attention: bool = False,
        regressor_hidden_dims: List[int] = None,
        regressor_dropout: float = 0.1,
        regressor_use_batch_norm: bool = False
    ):
        super().__init__()
        self.esm_hidden_dim = esm_hidden_dim
        
        if use_mock_esm:
            self.esm_encoder = MockESM2Encoder(hidden_dim=esm_hidden_dim)
        else:
            self.esm_encoder = ESM2Encoder(
                model_name=esm_model_name,
                freeze_backbone=freeze_esm,
                unfreeze_last_n_layers=esm_unfreeze_last_n_layers,
                checkpoint_path=esm_checkpoint_path
            )
        
        if use_simple_r2r:
            self.r2r = R2RModuleSimple(
                esm_dim=esm_hidden_dim,
                compress_dim=r2r_compress_dim,
                r2r_out_dim=r2r_out_dim,
                pooling=r2r_pooling
            )
        else:
            self.r2r = R2RModule(
                esm_dim=esm_hidden_dim,
                compress_dim=r2r_compress_dim,
                r2r_out_dim=r2r_out_dim,
                kan_hidden_dims_reduce=kan_hidden_dims_reduce,
                kan_hidden_dims_inter=kan_hidden_dims_inter,
                num_knots=num_knots,
                pooling=r2r_pooling
            )
        
        if use_extended_gspe:
            self.gspe = GSPEModuleExtended(
                input_dim=esm_hidden_dim,
                num_projections=gspe_num_projections,
                num_groups=gspe_num_groups,
                use_statistics=gspe_use_statistics,
                use_attention=gspe_use_attention
            )
        else:
            self.gspe = GSPEModule(
                input_dim=esm_hidden_dim,
                num_projections=gspe_num_projections,
                num_groups=gspe_num_groups,
                trainable_projections=gspe_trainable_projections,
                aggregation=gspe_aggregation
            )
        
        regressor_in_dim = r2r_out_dim + self.gspe.output_dim
        self.regressor = AffinityRegressor(
            in_dim=regressor_in_dim,
            hidden_dims=regressor_hidden_dims,
            dropout=regressor_dropout,
            use_batch_norm=regressor_use_batch_norm
        )
        
        self.config = {
            'esm_model_name': esm_model_name,
            'esm_hidden_dim': esm_hidden_dim,
            'r2r_out_dim': r2r_out_dim,
            'gspe_output_dim': self.gspe.output_dim
        }
    
    def encode_sequences(self, seq_ab: Union[str, List[str]], seq_ag: Union[str, List[str]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        X_ab_list = self.esm_encoder.encode(seq_ab)
        X_ag_list = self.esm_encoder.encode(seq_ag)
        return X_ab_list, X_ag_list
    
    def forward_single(self, X_ab: torch.Tensor, X_ag: torch.Tensor) -> torch.Tensor:
        h_R2R = self.r2r(X_ab, X_ag)
        h_pair = self.gspe(X_ab, X_ag)
        h_all = torch.cat([h_R2R, h_pair], dim=0)
        return self.regressor(h_all)
    
    def forward(
        self,
        seq_ab: Union[str, List[str]] = None,
        seq_ag: Union[str, List[str]] = None,
        X_ab_list: List[torch.Tensor] = None,
        X_ag_list: List[torch.Tensor] = None
    ) -> torch.Tensor:
        if seq_ab is not None and seq_ag is not None:
            X_ab_list, X_ag_list = self.encode_sequences(seq_ab, seq_ag)
        
        if X_ab_list is None or X_ag_list is None:
            raise ValueError("Must provide either sequences or embeddings")
        
        y_hat_list = []
        for X_ab, X_ag in zip(X_ab_list, X_ag_list):
            y_hat_list.append(self.forward_single(X_ab, X_ag))
        
        return torch.stack(y_hat_list)
    
    def predict(self, seq_ab: Union[str, List[str]], seq_ag: Union[str, List[str]]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(seq_ab=seq_ab, seq_ag=seq_ag)
    
    def apply_lora(
        self,
        lora_config: LoRAConfig = None,
        freeze_base: bool = True,
        apply_to_r2r: bool = True,
        apply_to_gspe: bool = True,
        apply_to_regressor: bool = True
    ):
        if lora_config is None:
            lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1)
        self._lora_config = lora_config
        
        if apply_to_r2r:
            from .kan import KANLayer
            for name, module in self.r2r.named_modules():
                if isinstance(module, KANLayer):
                    module.apply_lora(r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        
        if apply_to_gspe:
            apply_lora_to_linear(self.gspe, lora_config)
        
        if apply_to_regressor:
            apply_lora_to_linear(self.regressor, lora_config)
        
        if freeze_base:
            freeze_non_lora_parameters(self)
        
        if hasattr(self.esm_encoder, '_setup_freeze_state'):
            self.esm_encoder._setup_freeze_state()
    
    def get_lora_parameters(self) -> list:
        return get_lora_parameters(self)
    
    def get_non_lora_trainable_parameters(self) -> list:
        return get_non_lora_parameters(self)
    
    def merge_lora(self):
        merge_lora_weights(self)
    
    def unmerge_lora(self):
        unmerge_lora_weights(self)
    
    def save_lora_weights(self, path: str):
        lora_state = {name: param.data.clone() for name, param in self.named_parameters() if 'lora_' in name}
        torch.save({
            'lora_state_dict': lora_state,
            'lora_config': self._lora_config.to_dict() if hasattr(self, '_lora_config') else None
        }, path)
    
    def load_lora_weights(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        lora_state = checkpoint['lora_state_dict']
        current_state = self.state_dict()
        for name, param in lora_state.items():
            if name in current_state:
                current_state[name].copy_(param)

class DLPAffinityLoss(nn.Module):
    def __init__(
        self,
        use_huber: bool = False,
        huber_delta: float = 1.0,
        use_correlation_loss: bool = False,
        correlation_weight: float = 0.1
    ):
        super().__init__()
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.use_correlation_loss = use_correlation_loss
        self.correlation_weight = correlation_weight
        
        if use_huber:
            self.mse_loss = nn.HuberLoss(delta=huber_delta)
        else:
            self.mse_loss = nn.MSELoss()
    
    def correlation_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_centered = y_pred - y_pred.mean()
        y_true_centered = y_true - y_true.mean()
        cov = (y_pred_centered * y_true_centered).mean()
        std_pred = y_pred_centered.std() + 1e-8
        std_true = y_true_centered.std() + 1e-8
        return 1 - cov / (std_pred * std_true)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor = None) -> dict:
        losses = {}
        if sample_weights is not None:
            if self.use_huber:
                diff = y_pred - y_true
                abs_diff = torch.abs(diff)
                quadratic = torch.clamp(abs_diff, max=self.huber_delta)
                linear = abs_diff - quadratic
                mse = (0.5 * quadratic ** 2 + self.huber_delta * linear * sample_weights).mean()
            else:
                mse = (((y_pred - y_true) ** 2) * sample_weights).mean()
        else:
            mse = self.mse_loss(y_pred, y_true)
        
        losses['mse'] = mse
        if self.use_correlation_loss and len(y_pred) > 1:
            corr_loss = self.correlation_loss(y_pred, y_true)
            losses['correlation'] = corr_loss
            losses['total'] = mse + self.correlation_weight * corr_loss
        else:
            losses['total'] = mse
        return losses
