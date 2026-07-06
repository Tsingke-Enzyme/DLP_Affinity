
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math

class LoRAConfig:
    def __init__(self, r: int = 8, alpha: int = 16, dropout: float = 0.1, target_modules: List[str] = None, bias: str = "none"):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["base_weight", "net"]
        self.bias = bias
        self.scaling = alpha / r
    
    def to_dict(self) -> dict:
        return {"r": self.r, "alpha": self.alpha, "dropout": self.dropout, "target_modules": self.target_modules, "bias": self.bias}
    
    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        return cls(**d)

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.1, merge_weights: bool = False):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.scaling = alpha / r
        self.merge_weights = merge_weights
        self.merged = False
        
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False) if original_linear.bias is not None else self.register_parameter('bias', None)
        
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if not self.merged:
            lora_output = self.lora_dropout(x)
            lora_output = F.linear(F.linear(lora_output, self.lora_A), self.lora_B)
            result = result + lora_output * self.scaling
        return result
    
    def merge(self):
        if not self.merged:
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_weight
            self.merged = True
    
    def unmerge(self):
        if self.merged:
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_weight
            self.merged = False
            
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, scaling={self.scaling}"

class LoRAParameterLinear(nn.Module):
    def __init__(self, weight_shape: tuple, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.out_features, self.in_features = weight_shape
        self.r = r
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
        result = torch.einsum('...i,oi->...o', x, original_weight)
        lora_x = self.lora_dropout(x)
        lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B)
        return result + lora_out * self.scaling
    
    def get_merged_weight(self, original_weight: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return original_weight + delta

def apply_lora_to_kan(module: nn.Module, config: LoRAConfig) -> nn.Module:
    from .kan import KANLayer
    for name, child in module.named_modules():
        if isinstance(child, KANLayer):
            child.apply_lora(r=config.r, alpha=config.alpha, dropout=config.dropout)
    return module

def apply_lora_to_linear(module: nn.Module, config: LoRAConfig) -> nn.Module:
    try:
        from .kan import KANLayer
        has_kan = True
    except ImportError:
        has_kan = False
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            should_apply = any(target in name for target in config.target_modules)
            if should_apply or len(config.target_modules) == 0:
                setattr(module, name, LoRALinear(child, r=config.r, alpha=config.alpha, dropout=config.dropout))
        elif has_kan and isinstance(child, KANLayer):
            child.apply_lora(r=config.r, alpha=config.alpha, dropout=config.dropout)
        elif isinstance(child, nn.Sequential):
            for i, seq_child in enumerate(child):
                if isinstance(seq_child, nn.Linear):
                    child[i] = LoRALinear(seq_child, r=config.r, alpha=config.alpha, dropout=config.dropout)
                elif has_kan and isinstance(seq_child, KANLayer):
                    seq_child.apply_lora(r=config.r, alpha=config.alpha, dropout=config.dropout)
        else:
            apply_lora_to_linear(child, config)
            
def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if 'lora_' in n]

def get_non_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]

def freeze_non_lora_parameters(model: nn.Module):
    for n, p in model.named_parameters():
        if 'lora_' not in n: p.requires_grad = False

def unfreeze_all_parameters(model: nn.Module):
    for p in model.parameters(): p.requires_grad = True

def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"lora_params": lora, "frozen_params": total - trainable, "trainable_params": trainable, "total_params": total}

def merge_lora_weights(model: nn.Module):
    try:
        from .kan import KANLayer
        has_kan = True
    except ImportError: has_kan = False
    
    for module in model.modules():
        if isinstance(module, LoRALinear): module.merge()
        elif has_kan and isinstance(module, KANLayer) and hasattr(module, 'merge_lora'): module.merge_lora()

def unmerge_lora_weights(model: nn.Module):
    try:
        from .kan import KANLayer
        has_kan = True
    except ImportError: has_kan = False
    
    for module in model.modules():
        if isinstance(module, LoRALinear): module.unmerge()
        elif has_kan and isinstance(module, KANLayer) and hasattr(module, 'unmerge_lora'): module.unmerge_lora()
