#!/usr/bin/env python3
"""Evaluate a trained model on a dataset CSV.

Usage:
    python src/evaluate.py --data-path data/dataset.csv --model-path models/exp1/model.pt --scaler-path models/exp1/scaler.joblib
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--scaler-path", required=True)
    p.add_argument("--target", default="satisfaction_score")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target} not found in data")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if args.target in numeric_cols:
        numeric_cols.remove(args.target)
    X = df[numeric_cols].values.astype(np.float32)
    y = df[args.target].values.astype(np.float32)

    scaler = joblib.load(args.scaler_path)
    X = scaler.transform(X)

    model = MLP(input_dim=X.shape[1])
    state = torch.load(args.model_path, map_location='cpu')
    # Support both saved state_dict and wrapped dict
    if 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy()

    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"MSE: {mse:.4f} MAE: {mae:.4f} R2: {r2:.4f}")


if __name__ == '__main__':
    main()
