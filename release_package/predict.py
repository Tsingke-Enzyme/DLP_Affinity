
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models import DLPAffinity

def load_input_data(input_path: str) -> pd.DataFrame:
    input_path = Path(input_path)
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data.get('samples', data))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    if 'seq_ab' not in df.columns or 'seq_ag' not in df.columns:
        raise ValueError("Input data must contain 'seq_ab' and 'seq_ag' columns")
    return df

def load_model(checkpoint_path: str, device: str = 'cpu') -> DLPAffinity:
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')
    
    if config:
        if hasattr(config, 'esm'):
            model = DLPAffinity(
                esm_model_name=config.esm.model_name,
                esm_hidden_dim=config.esm.hidden_dim,
                use_mock_esm=config.esm.use_mock,
                esm_checkpoint_path=getattr(config.esm, 'checkpoint_path', None),
                r2r_compress_dim=config.r2r.compress_dim,
                r2r_out_dim=config.r2r.output_dim,
                r2r_pooling=config.r2r.pooling,
                use_simple_r2r=config.r2r.use_simple,
                gspe_num_projections=config.gspe.num_projections,
                gspe_num_groups=config.gspe.num_groups,
                use_extended_gspe=config.gspe.use_extended,
                regressor_hidden_dims=config.regressor.hidden_dims,
                regressor_dropout=config.regressor.dropout
            )
        else:
            model = DLPAffinity(
                esm_model_name=config.get('esm_model_name', 'facebook/esm2_t36_3B_UR50D'),
                esm_hidden_dim=config.get('esm_hidden_dim', 2560),
                esm_checkpoint_path=config.get('esm_checkpoint_path', None),
            )
    else:
        model = DLPAffinity()
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def predict(model: DLPAffinity, df: pd.DataFrame, batch_size: int = 8, device: str = 'cpu') -> np.ndarray:
    all_predictions = []
    iterator = range(0, len(df), batch_size)
    iterator = tqdm(iterator, total=(len(df)+batch_size-1)//batch_size, desc="Predicting")
    
    for i in iterator:
        batch_df = df.iloc[i:i+batch_size]
        with torch.no_grad():
            preds = model(seq_ab=batch_df['seq_ab'].tolist(), seq_ag=batch_df['seq_ag'].tolist())
        all_predictions.extend(preds.cpu().numpy())
    return np.array(all_predictions)

def main():
    parser = argparse.ArgumentParser(description='DLP-Affinity Prediction')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_log_transform', action='store_true')
    args = parser.parse_args()
    
    if args.output is None:
        args.output = str(Path(args.input).parent / f"{Path(args.input).stem}_predictions.csv")
    
    df = load_input_data(args.input)
    model = load_model(args.checkpoint, args.device)
    predictions_log10 = predict(model, df, batch_size=args.batch_size, device=args.device)
    
    df['predicted_kd_log10'] = predictions_log10
    df['predicted_kd'] = 10 ** predictions_log10
    
    print(f"Mean K_D (log10): {predictions_log10.mean():.4f}")
    df.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
