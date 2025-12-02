import os
from typing import List, Optional, Tuple
import json
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


class PairPerturbSeqDataset(Dataset):
    """Dataset that yields one (tf_seq, gene_seq) pair and a scalar expression per item.
    This avoids loading all pairs into memory as large tensors at once; sequences
    are kept as Python strings and tokenized on the fly in the collate function.
    """

    def __init__(
        self,
        parquet_path: str = "perturbseq_dataset_50.parquet",
        type: str = "train",
        train_fraction: float = 0.8,
        seed: int = 10701
    ):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df = df.sample(frac=1.0, random_state=seed, ignore_index=True)

        if type == "train":
            df = df.iloc[: int(train_fraction * len(df))]
        else:
            df = df.iloc[int(train_fraction * len(df)):]

        sequence_cols = [c for c in df.columns if "sequence" in c]
        expression_cols = [c for c in df.columns if "expression" in c]

        self.df = df.reset_index(drop=True)
        self.sequence_cols = sequence_cols
        self.expression_cols = expression_cols

        self.raw_sequences = [row[sequence_cols].tolist() for _, row in self.df.iterrows()]
        self.labels = [row[expression_cols].to_numpy(dtype=float) for _, row in self.df.iterrows()]

        self.index: List[Tuple[int, int]] = []
        for row_idx, seqs in enumerate(self.raw_sequences):
            n_genes = max(0, len(seqs) - 1)
            for gene_pos in range(n_genes):
                self.index.append((row_idx, gene_pos))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row_idx, gene_pos = self.index[idx]
        seqs = self.raw_sequences[row_idx]
        tf_seq = seqs[0]
        gene_seq = seqs[1 + gene_pos]
        y = float(self.labels[row_idx][gene_pos])
        return (tf_seq, gene_seq), torch.tensor(y, dtype=torch.float32)


def pair_collate(batch):
    """Collate that returns ((list_tf_seq, list_gene_seq), labels_tensor)."""
    inputs, labels = zip(*batch)
    tf_seqs = [inp[0] for inp in inputs]
    gene_seqs = [inp[1] for inp in inputs]
    labels = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.float32) for l in labels])
    return (tf_seqs, gene_seqs), labels


def get_pair_dataloader(
    parquet_path: str = "perturbseq_dataset_50.parquet",
    batch_size: int = 32,
    shuffle: bool = False,
    type: str = "train",
):
    ds = PairPerturbSeqDataset(parquet_path=parquet_path, type=type)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=pair_collate, drop_last=False)
    return loader


class EmbeddingPairDataset(Dataset):
    """Dataset that serves precomputed embeddings for tf/gene pairs.
    Expects parquet with columns: 'tf_sequence','gene_sequence','expression' OR will try to infer similarly.
    """
    def __init__(self, parquet_path, embeddings_path="precomputed/embeddings.npy", seq2idx_path="precomputed/seq2idx.json", train_fraction=0.8, type="train"):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(parquet_path)
        if not os.path.exists(embeddings_path) or not os.path.exists(seq2idx_path):
            raise FileNotFoundError("Embeddings or seq2idx not found; run precompute_embeddings first")

        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        with open(seq2idx_path, "r") as f:
            self.seq2idx = json.load(f)

        if type == "train":
            self.df = self.df.iloc[: int(train_fraction * len(self.df))]
        else:
            self.df = self.df.iloc[int(train_fraction * len(self.df)):]
        
        seq_cols = [c for c in self.df.columns if "sequence" in c.lower()]
        expr_cols = [c for c in self.df.columns if "expression" in c.lower()]
        self.tf_col = seq_cols[0]
        # handle multiple gene sequence columns and multiple expression columns
        self.gene_cols = seq_cols[1:]
        self.expr_cols = expr_cols

        # build flattened lists of (tf_idx, gene_idx, label) for all gene columns per row
        tf_idx_list = []
        gene_idx_list = []
        labels_list = []
        self._pair_records = []
        
        for _, row in self.df.iterrows():
            tf_seq = str(row[self.tf_col])
            tf_i = self.seq2idx.get(tf_seq, -1)
            if tf_i < 0:
                continue
            
            for j in range(len(self.gene_cols)):
                gene_seq = str(row[self.gene_cols[j]])
                gene_i = self.seq2idx.get(gene_seq, -1)
                label_val = row[self.expr_cols[j]]

                tf_idx_list.append(tf_i)
                gene_idx_list.append(gene_i)
                labels_list.append(float(label_val))
                self._pair_records.append({"tf_idx": int(tf_i), "gene_idx": int(gene_i), "label": float(label_val)})
        
        self.pairs_df = pd.DataFrame(self._pair_records)
        self.tf_idx = np.array(tf_idx_list, dtype=np.int64)
        self.gene_idx = np.array(gene_idx_list, dtype=np.int64)
        self.labels = np.array(labels_list, dtype=float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tf_embedding = torch.from_numpy(self.embeddings[self.tf_idx[idx]]).float()
        gene_embedding = torch.from_numpy(self.embeddings[self.gene_idx[idx]]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return (tf_embedding, gene_embedding), y


def emb_collate(batch):
    inputs, labels = zip(*batch)
    tf_embeddings = torch.stack([inp[0] for inp in inputs])
    gene_embeddings = torch.stack([inp[1] for inp in inputs])
    labels = torch.stack(labels)
    return (tf_embeddings, gene_embeddings), labels


def get_embedding_dataloader(parquet_path, batch_size=1024, shuffle=False, type: str = "train"):
    ds = EmbeddingPairDataset(parquet_path, type=type)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=emb_collate)