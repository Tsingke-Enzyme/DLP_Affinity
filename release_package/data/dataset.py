
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import random
from collections import defaultdict

class AffinityDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path] = None,
        data_list: List[Dict] = None,
        seq_ab_col: str = 'seq_ab',
        seq_ag_col: str = 'seq_ag',
        kd_col: str = 'kd',
        log_transform_kd: bool = True,
        max_ab_len: int = 512,
        max_ag_len: int = 512,
        cache_embeddings: bool = False
    ):
        self.log_transform_kd = log_transform_kd
        self.max_ab_len = max_ab_len
        self.max_ag_len = max_ag_len
        self.cache_embeddings = cache_embeddings
        
        if data_list is not None:
            self.samples = data_list
        elif data_path is not None:
            self.samples = self._load_data(data_path, seq_ab_col, seq_ag_col, kd_col)
        else:
            raise ValueError("Must provide either data_path or data_list")
        
        self.samples = [s for s in self.samples if len(s['seq_ab']) <= max_ab_len and len(s['seq_ag']) <= max_ag_len]
        self.embedding_cache = {} if cache_embeddings else None
        self.weights = self._compute_weights()

    def _compute_weights(self) -> torch.Tensor:
        kds = [s['kd'] for s in self.samples]
        df = pd.DataFrame({'kd': kds})
        counts = df['kd'].value_counts()
        sample_counts = df['kd'].map(counts).values
        weights = 1.0 / sample_counts
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float32)
    
    @staticmethod
    def _resolve_column(df_columns, preferred: str, aliases: List[str]) -> str:
        """Resolve CSV column name; support DMS naming (antibody_seq/antigen_seq/escape_fraction)."""
        candidates = [preferred] + [a for a in aliases if a != preferred]
        for name in candidates:
            if name in df_columns:
                return name
        raise ValueError(
            f"找不到列 '{preferred}'（备选: {aliases}），实际列: {list(df_columns)}"
        )

    def _load_data(self, data_path: Union[str, Path], seq_ab_col: str, seq_ag_col: str, kd_col: str) -> List[Dict]:
        data_path = Path(data_path)
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            # 7KMG DMS 原始列：antibody_seq / antigen_seq / escape_fraction
            ab_col = self._resolve_column(df.columns, seq_ab_col, ['seq_ab', 'antibody_seq', 'ab_seq'])
            ag_col = self._resolve_column(df.columns, seq_ag_col, ['seq_ag', 'antigen_seq', 'ag_seq'])
            label_col = self._resolve_column(df.columns, kd_col, ['kd', 'escape_fraction', 'affinity', 'label'])
            samples = []
            for _, row in df.iterrows():
                sample = {
                    'seq_ab': row[ab_col],
                    'seq_ag': row[ag_col],
                    'kd': float(row[label_col])
                }
                if 'chain_type' in row: sample['chain_type'] = row['chain_type']
                samples.append(sample)
            return samples
        elif data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list): return data
            elif 'samples' in data: return data['samples']
            else: raise ValueError(f"Unknown JSON format in {data_path}")
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        seq_ab = sample['seq_ab']
        seq_ag = sample['seq_ag']
        kd = sample['kd']
        if self.log_transform_kd: kd = np.log10(kd + 1e-10)
        return {
            'seq_ab': seq_ab,
            'seq_ag': seq_ag,
            'kd': torch.tensor(kd, dtype=torch.float32),
            'idx': idx,
            'sample_weight': self.weights[idx]
        }
    
    def get_chain_type(self, idx: int) -> Optional[str]:
        sample = self.samples[idx]
        return sample.get('chain_type', None)

class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset: AffinityDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, num_length_buckets: int = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_length_buckets = num_length_buckets
        
        self.chain_type_indices = defaultdict(list)
        for idx in range(len(dataset)):
            chain_type = dataset.get_chain_type(idx) or 'unknown'
            self.chain_type_indices[chain_type].append(idx)
        
        lengths = []
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            lengths.append(len(sample['seq_ab']) + len(sample['seq_ag']))
        
        self.length_buckets = self._create_length_buckets(lengths)
    
    def _create_length_buckets(self, lengths: List[int]) -> Dict[int, List[int]]:
        sorted_indices = np.argsort(lengths)
        bucket_size = len(lengths) // self.num_length_buckets
        buckets = {}
        for i in range(self.num_length_buckets):
            start = i * bucket_size
            end = len(lengths) if i == self.num_length_buckets - 1 else (i + 1) * bucket_size
            buckets[i] = sorted_indices[start:end].tolist()
        return buckets
    
    def __iter__(self):
        all_indices = list(range(len(self.dataset)))
        if self.shuffle:
            for bucket_id in self.length_buckets:
                random.shuffle(self.length_buckets[bucket_id])
            
            indices = []
            bucket_iters = {k: iter(v) for k, v in self.length_buckets.items()}
            bucket_keys = list(bucket_iters.keys())
            
            while bucket_iters:
                for key in list(bucket_keys):
                    try:
                        indices.append(next(bucket_iters[key]))
                    except StopIteration:
                        del bucket_iters[key]
                        bucket_keys.remove(key)
        else:
            indices = all_indices
        
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        if self.shuffle: random.shuffle(batches)
        for batch in batches: yield batch
    
    def __len__(self):
        if self.drop_last: return len(self.dataset) // self.batch_size
        else: return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def collate_fn(batch: List[Dict]) -> Dict:
    return {
        'seq_ab': [item['seq_ab'] for item in batch],
        'seq_ag': [item['seq_ag'] for item in batch],
        'kd': torch.stack([item['kd'] for item in batch]),
        'idx': [item['idx'] for item in batch],
        'sample_weight': torch.stack([item['sample_weight'] for item in batch])
    }

def create_dataloader(
    dataset: AffinityDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    use_stratified_sampler: bool = True,
    drop_last: bool = False
) -> DataLoader:
    if use_stratified_sampler:
        sampler = StratifiedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, drop_last=drop_last)

class MLMDataset(Dataset):
    def __init__(
        self,
        sequences: List[str],
        chain_types: List[str] = None,
        cdr_indices_list: List[List[int]] = None,
        mask_prob: float = 0.15,
        cdr_boost: float = 2.0,
        max_length: int = 512,
        vocab_size: int = 33
    ):
        self.sequences = [seq[:max_length] for seq in sequences]
        self.chain_types = chain_types
        self.cdr_indices_list = cdr_indices_list
        self.mask_prob = mask_prob
        self.cdr_boost = cdr_boost
        self.vocab_size = vocab_size
        
        self.aa_to_idx = {
            'A': 5, 'R': 6, 'N': 7, 'D': 8, 'C': 9, 'Q': 10, 'E': 11, 'G': 12, 'H': 13, 'I': 14,
            'L': 15, 'K': 16, 'M': 17, 'F': 18, 'P': 19, 'S': 20, 'T': 21, 'W': 22, 'Y': 23, 'V': 24,
            'X': 25, '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, '<mask>': 4
        }
        self.mask_token_id = self.aa_to_idx['<mask>']
    
    def __len__(self) -> int: return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        token_ids = torch.tensor([self.aa_to_idx.get(aa, self.aa_to_idx['X']) for aa in seq], dtype=torch.long)
        seq_len = len(seq)
        
        mask_prob = np.full(seq_len, self.mask_prob)
        if self.cdr_indices_list is not None and idx < len(self.cdr_indices_list):
            cdr_indices = self.cdr_indices_list[idx]
            if cdr_indices: mask_prob[cdr_indices] *= self.cdr_boost
        
        mask_prob = np.clip(mask_prob, 0.0, 0.9)
        mask = np.random.random(seq_len) < mask_prob
        mask_indices = np.where(mask)[0]
        
        labels = token_ids.clone()
        labels[~torch.tensor(mask)] = -100
        
        masked_token_ids = token_ids.clone()
        for i in mask_indices:
            rand = random.random()
            if rand < 0.8: masked_token_ids[i] = self.mask_token_id
            elif rand < 0.9: masked_token_ids[i] = random.randint(5, 24)
            
        return {'input_ids': masked_token_ids, 'labels': labels, 'attention_mask': torch.ones(seq_len, dtype=torch.long)}

def create_mock_data(num_samples: int = 100, min_ab_len: int = 80, max_ab_len: int = 150, min_ag_len: int = 50, max_ag_len: int = 200) -> List[Dict]:
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    chain_types = ['heavy', 'light', 'sdAb']
    samples = []
    for _ in range(num_samples):
        ab_len = random.randint(min_ab_len, max_ab_len)
        ag_len = random.randint(min_ag_len, max_ag_len)
        samples.append({
            'seq_ab': ''.join(random.choices(aa_list, k=ab_len)),
            'seq_ag': ''.join(random.choices(aa_list, k=ag_len)),
            'kd': 10 ** random.uniform(-2, 4),
            'chain_type': random.choice(chain_types)
        })
    return samples

if __name__ == "__main__":
    print("Testing data loading...")
    mock_data = create_mock_data(num_samples=100)
    print(f"Created {len(mock_data)} mock samples")
    dataset = AffinityDataset(data_list=mock_data)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    dataloader = create_dataloader(dataset, batch_size=8, shuffle=True, use_stratified_sampler=False)
    batch = next(iter(dataloader))
    print(f"Batch size: {len(batch['seq_ab'])}")
