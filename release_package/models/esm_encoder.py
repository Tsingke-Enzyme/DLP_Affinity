
import torch
import torch.nn as nn
from typing import List, Tuple, Union
import numpy as np
import sys
import types

class ESM2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t36_3B_UR50D",
        use_half: bool = False,
        freeze_backbone: bool = False,
        unfreeze_last_n_layers: int = 0,
        output_hidden_states: bool = True,
        checkpoint_path: str = None
    ):
        super().__init__()
        self.model_name = model_name
        self.use_half = use_half
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.checkpoint_path = checkpoint_path
        
        self._model = None
        self._tokenizer = None
        self._hidden_dim = None
        self._loaded = False
        self._target_device = None
    
    def _apply(self, fn):
        super()._apply(fn)
        dummy = torch.zeros(1)
        try:
            dummy = fn(dummy)
            self._target_device = dummy.device
        except:
            pass
        if self._loaded and self._model is not None:
            self._model = self._model.to(self._target_device)
        return self
    
    def _lazy_load(self):
        if self._loaded:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
             raise ImportError("Please install transformers")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.checkpoint_path is not None:
            if 'masked_language_modeling' not in sys.modules:
                sys.modules['masked_language_modeling'] = types.ModuleType('masked_language_modeling')

            try:
                try:
                    self._model = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
                except TypeError:
                    self._model = torch.load(self.checkpoint_path, map_location='cpu')
            except Exception as e:
                raise e
            
            if hasattr(self._model, 'module'):
                self._model = self._model.module
        else:
            self._model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
        
        if self.use_half:
            self._model = self._model.half()
        
        if self.freeze_backbone:
            self._setup_freeze_state()
            
    def _setup_freeze_state(self):
        if self._model is None:
            self._lazy_load()
        
        for param in self._model.parameters():
            param.requires_grad = False
        
        if hasattr(self, 'unfreeze_last_n_layers') and self.unfreeze_last_n_layers > 0:
            if hasattr(self._model, 'encoder') and hasattr(self._model.encoder, 'layer'):
                layers = self._model.encoder.layer
                total_layers = len(layers)
                start_idx = total_layers - self.unfreeze_last_n_layers
                for i in range(start_idx, total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
            
            if hasattr(self._model.encoder, 'emb_layer_norm_after'):
                    for param in self._model.encoder.emb_layer_norm_after.parameters():
                        param.requires_grad = True
            
            if hasattr(self._model, 'pooler') and self._model.pooler is not None:
                for param in self._model.pooler.parameters():
                    param.requires_grad = True
        
        if hasattr(self._model, 'config'):
            self._hidden_dim = self._model.config.hidden_size
        else:
            self._hidden_dim = self.hidden_dim
        
        self._loaded = True
        
        if self._target_device is not None:
            self._model = self._model.to(self._target_device)

    @property
    def hidden_dim(self) -> int:
        if self._hidden_dim is None:
            if "3B" in self.model_name: return 2560
            elif "650M" in self.model_name: return 1280
            elif "150M" in self.model_name: return 640
            elif "35M" in self.model_name: return 480
            elif "8M" in self.model_name: return 320
            else: return 2560
        return self._hidden_dim
    
    @property
    def tokenizer(self):
        self._lazy_load()
        return self._tokenizer
    
    @property
    def model(self):
        self._lazy_load()
        return self._model
    
    def encode(
        self,
        sequences: Union[str, List[str]],
        return_attention_mask: bool = False,
        remove_special_tokens: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        self._lazy_load()
        if isinstance(sequences, str):
            sequences = [sequences]
        
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(next(self.model.parameters()).device)
        attention_mask = encoded["attention_mask"].to(input_ids.device)
        
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings = outputs.last_hidden_state
        
        if remove_special_tokens:
            result_embeddings = []
            for i in range(len(sequences)):
                seq_len = len(sequences[i])
                result_embeddings.append(embeddings[i, 1:seq_len+1, :])
            return res_embeddings, attention_mask if return_attention_mask else result_embeddings
        
        if return_attention_mask:
            return embeddings, attention_mask
        return embeddings
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        self._lazy_load()
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

class MockESM2Encoder(nn.Module):
    def __init__(self, hidden_dim: int = 2560, vocab_size: int = 33):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.aa_to_idx = {
            'A': 5, 'R': 6, 'N': 7, 'D': 8, 'C': 9, 'Q': 10, 'E': 11, 'G': 12, 'H': 13, 'I': 14,
            'L': 15, 'K': 16, 'M': 17, 'F': 18, 'P': 19, 'S': 20, 'T': 21, 'W': 22, 'Y': 23, 'V': 24,
            'X': 25, '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, '<mask>': 4
        }
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    def tokenize(self, sequence: str) -> torch.Tensor:
        tokens = [self.aa_to_idx.get(aa, self.aa_to_idx['X']) for aa in sequence]
        return torch.tensor(tokens, dtype=torch.long)
    
    def encode(self, sequences: Union[str, List[str]], return_attention_mask: bool = False, remove_special_tokens: bool = True):
        if isinstance(sequences, str): sequences = [sequences]
        embeddings_list = []
        device = self.embedding.weight.device
        for seq in sequences:
            tokens = self.tokenize(seq).to(device)
            emb = self.embedding(tokens)
            emb = self.transformer(emb.unsqueeze(0)).squeeze(0)
            embeddings_list.append(emb)
        if return_attention_mask:
            max_len = max(len(seq) for seq in sequences)
            attention_mask = torch.zeros(len(sequences), max_len, device=device)
            for i, seq in enumerate(sequences): attention_mask[i, :len(seq)] = 1
            return embeddings_list, attention_mask
        return embeddings_list
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        return self.transformer(emb)

class CDRMasker:
    def __init__(self, base_mask_prob: float = 0.15, cdr_boost: float = 2.0, max_mask_prob: float = 0.9):
        self.base_mask_prob = base_mask_prob
        self.cdr_boost = cdr_boost
        self.max_mask_prob = max_mask_prob
    
    def generate_mask(self, seq_len: int, cdr_indices: List[int] = None) -> np.ndarray:
        mask_prob = np.full(seq_len, self.base_mask_prob)
        if cdr_indices is not None:
            mask_prob[cdr_indices] *= self.cdr_boost
        mask_prob = np.clip(mask_prob, 0.0, self.max_mask_prob)
        return np.random.random(seq_len) < mask_prob