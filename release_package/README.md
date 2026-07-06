# DLP-Affinity

Deep learning-based antibody-antigen binding affinity prediction model.

## Overview

DLP-Affinity is a deep learning model for predicting antibody-antigen binding affinity ($K_D$), integrating the following components:

1.  **ESM2-FT**: ESM2-3B protein language model fine-tuned on antibody data.
2.  **R2R (Residue-to-Residue)**: Residue-level interaction modeling module.
3.  **GSPE (Global Stochastic Projection Embedding)**: Global stochastic projection embedding module.
4.  **Regression Head**: Predicts $K_D$ values.

## Project Structure

```
dlp_affinity/
├── configs/
│   ├── __init__.py
│   └── config.py          # Configuration class definition
├── data/
│   ├── __init__.py
│   └── dataset.py         # Dataset and data loading
├── models/
│   ├── __init__.py
│   ├── kan.py             # Kolmogorov-Arnold Network
│   ├── r2r.py             # R2R module
│   ├── gspe.py            # GSPE module
│   ├── esm_encoder.py     # ESM2 encoder
│   └── dlp_affinity.py    # Main model wrapper
├── train.py               # Main training script
├── train_mlm.py           # ESM2 MLM fine-tuning script
├── predict.py             # Prediction script
├── test_all.py            # Full test suite
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd dlp_affinity

# Create conda environment
conda create -n dlp_affinity python=3.10
conda activate dlp_affinity

# Install dependencies
pip install -r requirements.txt

# Install ESM2 (Required)
pip install transformers
```

## Usage

### Workflow

```
ESM2 Original Weights ──▶ ESM2-FT Weights ──▶ Full DLP-Affinity Model
   (Pre-training)         (MLM Fine-tuning)     (Affinity Training)
                            ↓                       ↓
                      Your checkpoint          Saved after training
                  (esm2_xxx.pt)             (best_model.pt)
                                                    ↓
                                                Prediction
```

### 1. Training the Full Model with Fine-tuned ESM2

If you already have a fine-tuned ESM2 checkpoint:

```bash
python train.py \
    --esm_checkpoint /path/to/esm2_t36_3B_UR50Dbest26.pt \
    --train_path /path/to/train.csv \
    --val_path /path/to/val.csv \
    --output_dir ./outputs
```

**Training Data Format (CSV)**:
```csv
seq_ab,seq_ag,kd
DIVLTQSPASLAVSLGQRATISCRASESVD,MKTIIALSYILCLVFAQVSNG,1.5e-9
EVQLVESGGGLVQPGGSLRLSCAASGFTFS,GPLDVQVTEDAVRRYLTRKPMAVVV,3.2e-8
...
```

**Optional Arguments**:
- `--freeze_esm`: Freeze ESM2 parameters (train R2R, GSPE, and Regression Head only).
- `--use_small`: Use a small model configuration (for testing purposes).

### 2. Running Predictions

After training, use the saved model for prediction:

```bash
python predict.py \
    --checkpoint ./outputs/best_model.pt \
    --input /path/to/test_data.csv \
    --output /path/to/predictions.csv
```

**Prediction Input Data Format (CSV)**:
```csv
seq_ab,seq_ag
DIVLTQSPASLAVSLGQRATISCRASESVD,MKTIIALSYILCLVFAQVSNG
EVQLVESGGGLVQPGGSLRLSCAASGFTFS,GPLDVQVTEDAVRRYLTRKPMAVVV
...
```

**Prediction Output**:
- Predictions are printed to the terminal.
- CSV file contains: `seq_ab, seq_ag, predicted_kd_log10, predicted_kd`

### 3. Quick Test (Using Mock Data)

```bash
# Test with small model and mock data
python train.py --use_small --use_mock_data --num_mock_samples 200
```

### 4. ESM2 MLM Fine-tuning (Optional)

```bash
# Fine-tune using real sequences
python train_mlm.py \
    --model_name facebook/esm2_t36_3B_UR50D \
    --sequences_file /path/to/antibody_sequences.txt \
    --output_dir ./outputs/mlm
```

## Module Descriptions

### KAN (Kolmogorov-Arnold Network)

Uses learnable B-spline activation functions, fitting highly non-linear residue interaction mappings better than traditional MLPs.

```python
from models import KANReduce

# Compress 2560 dims to 1 dim
kan = KANReduce(in_dim=2560, out_dim=1)
x = torch.randn(100, 2560)  # 100 residues
v = kan(x)  # [100, 1]
```

### R2R Module

Models interactions between antibody residues and antigen residues:

1. KAN Compression: 2560 → 1
2. Outer Product to construct interaction matrix
3. Concatenate antibody self-features
4. KAN to extract interaction-aware representations
5. Pooling to get fixed-length representation

```python
from models import R2RModule

r2r = R2RModule(esm_dim=2560, r2r_out_dim=32)
X_ab = torch.randn(120, 2560)  # Antibody embeddings
X_ag = torch.randn(200, 2560)  # Antigen embeddings
h_r2r = r2r(X_ab, X_ag)  # [32]
```

### GSPE Module

Maps variable-length sequences to fixed-length, order-invariant global representations:

1. Random Projection + Sorting (ensures permutation invariance)
2. Compute global antibody-antigen distance
3. Softplus mapping

```python
from models import GSPEModule

gspe = GSPEModule(input_dim=2560, num_projections=64, num_groups=8)
h_pair = gspe(X_ab, X_ag)  # [8]
```

## Data Format

Training data should be in CSV or JSON format:

### CSV Format
```csv
seq_ab,seq_ag,kd,chain_type
DIVLTQSPASLAVSLGQ...,MKTIIALSYILCLVFAQ...,1.5e-9,heavy
...
```

### JSON Format
```json
[
  {
    "seq_ab": "DIVLTQSPASLAVSLGQ...",
    "seq_ag": "MKTIIALSYILCLVFAQ...",
    "kd": 1.5e-9,
    "chain_type": "heavy"
  },
  ...
]
```

## Configuration

Main configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `esm_model_name` | `facebook/esm2_t36_3B_UR50D` | ESM2 model name |
| `esm_hidden_dim` | 2560 | ESM2 hidden dimension |
| `r2r_compress_dim` | 1 | R2R compression dimension |
| `r2r_out_dim` | 32 | R2R output dimension |
| `gspe_num_projections` | 64 | GSPE number of projections (m) |
| `gspe_num_groups` | 8 | GSPE number of groups (n) |
| `learning_rate` | 1e-4 | Learning rate |
| `warmup_steps` | 100 | Warmup steps |
| `batch_size` | 8 | Batch size |

## Training Details

### Main Task Training

- Optimizer: AdamW
- LR Schedule: Linear Warmup
- Loss Function: MSE (Optional Huber Loss)
- Gradient Clipping: max_norm=1.0

### MLM Fine-tuning (ESM2-FT)

- CDR Priority Masking: Higher masking probability for CDR regions
- Base Masking Probability: 15%
- CDR Masking Multiplier: 2.0x
- Warmup Steps: 40
- Learning Rate: 1e-5

## Metrics

- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- Pearson Correlation: Pearson Correlation Coefficient

## References

- ESM2: https://github.com/facebookresearch/esm
- KAN: Kolmogorov-Arnold Networks

## License

MIT License
