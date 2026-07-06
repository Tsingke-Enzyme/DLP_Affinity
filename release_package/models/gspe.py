
import torch
import torch.nn as nn
import torch.nn.functional as F

class GSPEModule(nn.Module):
    def __init__(
        self,
        input_dim: int = 2560,
        num_projections: int = 64,
        num_groups: int = 8,
        trainable_projections: bool = False,
        aggregation: str = 'mean'
    ):
        super().__init__()
        self.num_groups = num_groups
        self.aggregation = aggregation
        
        projections = []
        for _ in range(num_groups):
            R = torch.randn(num_projections, input_dim)
            R = R / R.norm(dim=1, keepdim=True)
            projections.append(R)
        
        projections = torch.stack(projections, dim=0)
        if trainable_projections:
            self.R = nn.Parameter(projections)
        else:
            self.register_buffer('R', projections)
            
        self.sigma = nn.Parameter(torch.ones(num_projections))
        self.output_dim = num_groups
    
    def project_and_sort(self, X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        P = torch.matmul(R, X.T)
        P_sorted, _ = torch.sort(P, dim=1)
        
        if self.aggregation == 'mean': return P_sorted.mean(dim=1)
        elif self.aggregation == 'max': return P_sorted.max(dim=1)[0]
        elif self.aggregation == 'sum': return P_sorted.sum(dim=1)
        return P_sorted.mean(dim=1)
    
    def compute_distance(self, hat_p_A: torch.Tensor, hat_p_B: torch.Tensor) -> torch.Tensor:
        diff = (hat_p_A - hat_p_B) / (self.sigma + 1e-8)
        return torch.sqrt((diff ** 2).sum() + 1e-8)
    
    def forward(self, X_ab: torch.Tensor, X_ag: torch.Tensor) -> torch.Tensor:
        z_list = []
        for i in range(self.num_groups):
            R_i = self.R[i]
            hat_p_A = self.project_and_sort(X_ab, R_i)
            hat_p_B = self.project_and_sort(X_ag, R_i)
            z_list.append(F.softplus(self.compute_distance(hat_p_A, hat_p_B)))
        return torch.stack(z_list)
    
    def get_single_representation(self, X: torch.Tensor) -> torch.Tensor:
        h_list = []
        for i in range(self.num_groups):
            h_list.append(self.project_and_sort(X, self.R[i]))
        return torch.stack(h_list)

class GSPEModuleExtended(nn.Module):
    def __init__(
        self,
        input_dim: int = 2560,
        num_projections: int = 64,
        num_groups: int = 8,
        use_statistics: bool = True,
        use_attention: bool = False
    ):
        super().__init__()
        self.use_statistics = use_statistics
        self.use_attention = use_attention
        
        self.gspe = GSPEModule(input_dim, num_projections, num_groups)
        
        self.stats_dim = 4 * num_groups if use_statistics else 0
        if use_attention:
            self.attention = nn.MultiheadAttention(num_projections, 4, batch_first=True)
            self.attn_dim = num_groups
        else:
            self.attn_dim = 0
            
        self.sigma = nn.Parameter(torch.ones(num_projections))
        self.output_dim = num_groups + self.stats_dim + self.attn_dim
    
    def forward(self, X_ab: torch.Tensor, X_ag: torch.Tensor) -> torch.Tensor:
        features = [self.gspe(X_ab, X_ag)]
        
        if self.use_statistics:
            h_ab = self.gspe.get_single_representation(X_ab)
            h_ag = self.gspe.get_single_representation(X_ag)
            diff = h_ab - h_ag
            features.append(torch.cat([diff.mean(1), diff.std(1), diff.min(1)[0], diff.max(1)[0]]))
        
        if self.use_attention:
            h_ab = self.gspe.get_single_representation(X_ab)
            h_ag = self.gspe.get_single_representation(X_ag)
            attn_out, _ = self.attention(h_ab.unsqueeze(0), h_ag.unsqueeze(0), h_ag.unsqueeze(0))
            features.append(attn_out.squeeze(0).mean(1))
            
        return torch.cat(features)
