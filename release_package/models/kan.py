
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BSplineBasis(nn.Module):
    def __init__(self, num_knots: int = 8, degree: int = 3):
        super().__init__()
        self.num_knots = num_knots
        self.degree = degree
        self.register_buffer('knots', torch.linspace(-1, 1, num_knots + 2 * degree))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        knots = self.knots
        n_basis = self.num_knots + self.degree - 1
        basis = torch.zeros(*x.shape[:-1], n_basis, device=x.device, dtype=x.dtype)
        
        for i in range(n_basis):
            left = knots[i]
            right = knots[i + self.degree + 1] if i + self.degree + 1 < len(knots) else knots[-1]
            center = (left + right) / 2
            width = (right - left) / 2 + 1e-6
            dist = (x.squeeze(-1) - center) / width
            basis[..., i] = torch.exp(-dist ** 2)
        return basis

class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_knots: int = 8, degree: int = 3, base_activation: str = 'silu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        
        self.spline_basis = BSplineBasis(num_knots, degree)
        n_basis = num_knots + degree - 1
        
        self.spline_coeffs = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        
        if base_activation == 'silu': self.base_activation = nn.SiLU()
        elif base_activation == 'relu': self.base_activation = nn.ReLU()
        elif base_activation == 'gelu': self.base_activation = nn.GELU()
        else: self.base_activation = nn.Identity()
        
        self.scale = nn.Parameter(torch.ones(out_features))
        self._lora_enabled = False
        self._lora_merged = False
    
    def apply_lora(self, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        self._lora_r = r
        self._lora_alpha = alpha
        self._lora_scaling = alpha / r
        
        self.base_weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._lora_enabled = True
    
    def merge_lora(self):
        if self._lora_enabled and not self._lora_merged:
            delta = (self.lora_B @ self.lora_A) * self._lora_scaling
            self.base_weight.data += delta
            self._lora_merged = True
    
    def unmerge_lora(self):
        if self._lora_enabled and self._lora_merged:
            delta = (self.lora_B @ self.lora_A) * self._lora_scaling
            self.base_weight.data -= delta
            self._lora_merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 2: x = x.unsqueeze(1)
        
        base_out = self.base_activation(x)
        base_out = torch.einsum('bsi,oi->bso', base_out, self.base_weight)
        
        if self._lora_enabled and not self._lora_merged:
            lora_x = self.lora_dropout(self.base_activation(x))
            lora_out = torch.einsum('bsi,ri->bsr', lora_x, self.lora_A)
            lora_out = torch.einsum('bsr,or->bso', lora_out, self.lora_B)
            base_out = base_out + lora_out * self._lora_scaling
        
        basis_values = self.spline_basis(x.unsqueeze(-1))
        spline_out = torch.einsum('bsik,oik->bso', basis_values, self.spline_coeffs)
        
        out = (base_out + spline_out) * self.scale.unsqueeze(0).unsqueeze(0)
        
        if len(original_shape) == 2: out = out.squeeze(1)
        return out

class KAN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list = None, num_knots: int = 8, degree: int = 3, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None: hidden_dims = [64, 32]
        dims = [in_dim] + hidden_dims + [out_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1], num_knots=num_knots, degree=degree))
            if i < len(dims) - 2: layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class KANReduce(nn.Module):
    def __init__(self, in_dim: int = 2560, out_dim: int = 1, hidden_dims: list = None, num_knots: int = 8):
        super().__init__()
        if hidden_dims is None: hidden_dims = [512, 128, 32]
        self.kan = KAN(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims, num_knots=num_knots)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)

class KANInter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 32, hidden_dims: list = None, num_knots: int = 8):
        super().__init__()
        if hidden_dims is None: hidden_dims = [128, 64]
        self.kan = KAN(in_dim=in_dim, out_dim=out_dim, hidden_dims=hidden_dims, num_knots=num_knots)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(x)
