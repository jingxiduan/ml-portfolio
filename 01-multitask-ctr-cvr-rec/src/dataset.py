import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class CtrCvrDataset(Dataset):
    """Loads preprocessed train/val/test csv, maps sparse feature ids and standardizes dense features."""

    def __init__(self, split: str, stats: Optional[dict] = None):
        with open(os.path.join(PROC_DIR, "meta.json")) as f:
            meta = json.load(f)
        self.sparse_cols = meta["sparse_cols"]
        self.dense_cols = meta["dense_cols"]

        df = pd.read_csv(os.path.join(PROC_DIR, f"{split}.csv"))
        self.df = df

        if stats is None:
            self.stats = {
                "dense_mean": df[self.dense_cols].mean().values.astype(np.float32),
                "dense_std": (df[self.dense_cols].std().values.astype(np.float32) + 1e-6),
                "vocab_size": {c: int(df[c].max()) + 2 for c in self.sparse_cols},
            }
        else:
            self.stats = stats

        self.sparse = df[self.sparse_cols].clip(lower=0).values.astype(np.int64)
        dense_raw = df[self.dense_cols].values.astype(np.float32)
        self.dense = (dense_raw - self.stats["dense_mean"]) / self.stats["dense_std"]
        self.label_click = df["label_click"].values.astype(np.float32)
        self.label_convert = df["label_convert"].values.astype(np.float32)
        self.user_id = df["user_id"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sparse[idx]),
            torch.from_numpy(self.dense[idx]),
            torch.tensor(self.label_click[idx]),
            torch.tensor(self.label_convert[idx]),
        )
