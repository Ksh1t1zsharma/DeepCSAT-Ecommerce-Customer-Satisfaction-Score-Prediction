#!/usr/bin/env python3
"""Simple training script for a PyTorch MLP baseline.

Usage:
    python src/train.py --data-path data/dataset.csv --target satisfaction_score --output-dir models/exp1
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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
    p.add_argument("--target", default="satisfaction_score")
    p.add_argument("--output-dir", default="models/exp1")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target} not found in data")

    # Use all numeric columns except the target
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if args.target in numeric_cols:
        numeric_cols.remove(args.target)
    if len(numeric_cols) == 0:
        raise ValueError("No numeric feature columns found. Please preprocess your data into numeric features.")

    X = df[numeric_cols].values.astype(np.float32)
    y = df[args.target].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_losses.append(loss_fn(preds, yb).item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    # Save final model state dict and metadata
    torch.save({'state_dict': model.state_dict()}, os.path.join(args.output_dir, 'model_final.pt'))
    print(f"Training complete. Best val MSE: {best_val:.4f}. Artifacts saved to {args.output_dir}")


if __name__ == '__main__':
    main()
