"""
DLP-Affinity Model Module
"""

from .kan import KAN, KANLayer, KANReduce, KANInter
from .r2r import R2RModule, R2RModuleSimple
from .gspe import GSPEModule, GSPEModuleExtended
from .esm_encoder import ESM2Encoder, MockESM2Encoder, CDRMasker
from .dlp_affinity import DLPAffinity, AffinityRegressor, DLPAffinityLoss

__all__ = [
    # KAN
    'KAN',
    'KANLayer',
    'KANReduce',
    'KANInter',
    # R2R
    'R2RModule',
    'R2RModuleSimple',
    # GSPE
    'GSPEModule',
    'GSPEModuleExtended',
    # ESM Encoder
    'ESM2Encoder',
    'MockESM2Encoder',
    'CDRMasker',
    # Main Model
    'DLPAffinity',
    'AffinityRegressor',
    'DLPAffinityLoss',
]
