import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PerturbSeqDataset(Dataset):
    """Dataset for Perturb-Seq data stored in a Parquet file.
    Args:
        parquet_path: path to the parquet file (e.g. 'perturbseq_dataset.parquet')
        type: 'train' or 'test' split
        train_fraction: fraction of data to use for training (rest for test)
        seed: random seed for sampling
        encode_sequence: whether to encode sequence characters to integer
            tokens. If True, a vocabulary is built from characters present in
            the sequences and characters are mapped to 2..V; 0 is PAD.
        max_len: if provided, sequences will be truncated/padded to this
            maximum length when encoding. Padding happens in the collate fn.
    """

    def __init__(
        self,
        parquet_path: str = "perturbseq_dataset.parquet",
        type: str = "train",
        train_fraction: float = 0.8,
        seed: int = 10701,
    ):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self.parquet_path = parquet_path
        self.type = type
        self.train_fraction = train_fraction
        self.seed = seed
    # this dataset returns raw string sequences (no encoding)

        df = pd.read_parquet(parquet_path)

        # shuffle all rows
        df = df.sample(frac=1.0, random_state=seed, ignore_index=True)

        # keep only first 80 percent for train, last 20 percent for test
        if self.type == "train":
            df = df.iloc[: int(self.train_fraction * len(df))]
        else:
            df = df.iloc[int(self.train_fraction * len(df)):]


        sequence_cols = [c for c in df.columns if 'sequence' in c]
        expression_cols = [c for c in df.columns if 'expression' in c]


        self.df = df.reset_index(drop=True)
        self.expression_cols = expression_cols

        # store sequences and labels
        self.raw_sequences = []
        self.labels = []

        for idx, row in self.df.iterrows():
            seqs = row[sequence_cols].tolist()
            self.raw_sequences.append(seqs)
            self.labels.append(row[expression_cols].to_numpy(dtype=float))

        self.length = self.df.shape[0]

        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq = self.raw_sequences[idx]
        # return raw string sequence (caller can tokenize later)
        x = seq

        y = self.labels[idx]
        # convert y to tensor
        if np.isscalar(y):
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)

        return x, y_tensor


def perturbseq_collate(batch):
    """Collate function for PerturbSeqDataset.
    Returns a tuple (list_of_string_sequences, labels_tensor).
    """
    inputs, labels = zip(*batch)
    labels = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.float32) for l in labels])
    return list(inputs), labels


def get_dataloader(
    parquet_path: str = "perturbseq_dataset_50.parquet",
    batch_size: int = 32,
    shuffle: bool = True,
    type: str = "train",
):
    """Utility that returns a DataLoader for the whole parquet file.
    """
    ds = PerturbSeqDataset(
        parquet_path=parquet_path,
        type=type,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=perturbseq_collate,
        drop_last=False,
    )

    return loader
