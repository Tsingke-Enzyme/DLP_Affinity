
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan import KANReduce, KAN

class R2RModule(nn.Module):
    def __init__(
        self,
        esm_dim: int = 2560,
        compress_dim: int = 1,
        r2r_out_dim: int = 32,
        kan_hidden_dims_reduce: list = None,
        kan_hidden_dims_inter: list = None,
        num_knots: int = 8,
        pooling: str = 'mean',
        max_ag_len: int = 512
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.compress_dim = compress_dim
        self.r2r_out_dim = r2r_out_dim
        self.pooling = pooling
        self.max_ag_len = max_ag_len
        
        if kan_hidden_dims_reduce is None:
            kan_hidden_dims_reduce = [512, 128, 32]
        if kan_hidden_dims_inter is None:
            kan_hidden_dims_inter = [256, 128]
        
        self.kan_ab = KANReduce(in_dim=esm_dim, out_dim=compress_dim, hidden_dims=kan_hidden_dims_reduce, num_knots=num_knots)
        self.kan_ag = KANReduce(in_dim=esm_dim, out_dim=compress_dim, hidden_dims=kan_hidden_dims_reduce, num_knots=num_knots)
        
        self.ag_proj = nn.Linear(1, 64)
        self.inter_kan = KAN(
            in_dim=64 + compress_dim,
            out_dim=r2r_out_dim,
            hidden_dims=kan_hidden_dims_inter,
            num_knots=num_knots
        )
        
        if pooling == 'attention':
            self.attention_weight = nn.Sequential(
                nn.Linear(r2r_out_dim, r2r_out_dim // 2), nn.Tanh(), nn.Linear(r2r_out_dim // 2, 1)
            )
    
    def forward(self, X_ab: torch.Tensor, X_ag: torch.Tensor) -> torch.Tensor:
        V_ab = self.kan_ab(X_ab)
        V_ag = self.kan_ag(X_ag)
        M_inter = torch.matmul(V_ab, V_ag.T)
        
        attn_weights = F.softmax(M_inter, dim=1)
        V_ag_proj = self.ag_proj(V_ag)
        M_proj = torch.matmul(attn_weights, V_ag_proj)
        
        M_concat = torch.cat([M_proj, V_ab], dim=1)
        H_AA = self.inter_kan(M_concat)
        
        if self.pooling == 'mean':
            return H_AA.mean(dim=0)
        elif self.pooling == 'max':
            return H_AA.max(dim=0)[0]
        elif self.pooling == 'attention':
            weights = F.softmax(self.attention_weight(H_AA), dim=0)
            return (H_AA * weights).sum(dim=0)
        return H_AA.mean(dim=0)

class R2RModuleSimple(nn.Module):
    def __init__(
        self,
        esm_dim: int = 2560,
        compress_dim: int = 1,
        r2r_out_dim: int = 32,
        num_knots: int = 8,
        pooling: str = 'mean'
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.pooling = pooling
        
        self.kan_ab = KANReduce(in_dim=esm_dim, out_dim=compress_dim, hidden_dims=[512, 128, 32], num_knots=num_knots)
        self.kan_ag = KANReduce(in_dim=esm_dim, out_dim=compress_dim, hidden_dims=[512, 128, 32], num_knots=num_knots)
        
        self.inter_mlp = nn.Sequential(
            nn.LazyLinear(256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, r2r_out_dim)
        )
        
        if pooling == 'attention':
            self.attention_weight = nn.Sequential(
                nn.Linear(r2r_out_dim, r2r_out_dim // 2), nn.Tanh(), nn.Linear(r2r_out_dim // 2, 1)
            )
    
    def forward(self, X_ab: torch.Tensor, X_ag: torch.Tensor) -> torch.Tensor:
        V_ab = self.kan_ab(X_ab)
        V_ag = self.kan_ag(X_ag)
        M_inter = torch.matmul(V_ab, V_ag.T)
        M_concat = torch.cat([M_inter, V_ab], dim=1)
        H_AA = self.inter_mlp(M_concat)
        
        if self.pooling == 'mean':
            return H_AA.mean(dim=0)
        elif self.pooling == 'max':
            return H_AA.max(dim=0)[0]
        elif self.pooling == 'attention':
            weights = F.softmax(self.attention_weight(H_AA), dim=0)
            return (H_AA * weights).sum(dim=0)
        return H_AA.mean(dim=0)
