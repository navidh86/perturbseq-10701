import os
from typing import List, Optional, Tuple
import json
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PairPerturbSeqDataset(Dataset):
    """Dataset for Perturb-Seq data stored in a Parquet file.
    Args:
        parquet_path: path to the parquet file (e.g. 'perturbseq_dataset.parquet')
        tf_sequences_path: path to dict of tf sequences
        gene_sequences_path: path to dict of gene sequences
        type: 'train' or 'test' split
        shuffle: whether to shuffle before split
        train_fraction: fraction of data to use for training (rest for test)
        majority_fraction: fraction of majority class to keep
        seed: random seed for sampling
    """

    def __init__(
        self,
        parquet_path: str = "tf_gene_expression_labeled.parquet",
        tf_sequences_path: str = "tf_sequences.pkl",
        gene_sequences_path: str = "gene_sequences_4000bp.pkl",
        type: str = "train",
        shuffle: bool = True,
        train_fraction: float = 0.8,
        majority_fraction: float = 1.0,
        seed: int = 10701,
    ):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self.parquet_path = parquet_path
        self.tf_sequences_path = tf_sequences_path
        self.gene_sequences_path = gene_sequences_path
        self.type = type
        self.train_fraction = train_fraction
        self.majority_fraction = majority_fraction
        self.seed = seed
    
        # this dataset returns raw string sequences (no encoding)
        df = pd.read_parquet(parquet_path)
        self.tf_seqs = pickle.load(open(self.tf_sequences_path, "rb"))
        self.gene_seqs = pickle.load(open(self.gene_sequences_path, "rb"))

        # drop tf rows not in tf_seqs
        df = df[df["tf_name"].isin(self.tf_seqs.keys())]

        # drop gene rows not in gene_seqs
        df = df[df["gene_name"].isin(self.gene_seqs.keys())]

        # downsample majority class if specified
        if majority_fraction < 1.0:
            counts = df['expression_label'].value_counts()
            majority_class = counts.idxmax()
            minority_classes = counts.index[counts.index != majority_class]

            # keep all minority class rows
            minority_df = df[df['expression_label'].isin(minority_classes)]

            # downsample majority class
            majority_df = df[df['expression_label'] == majority_class]
            majority_df = majority_df.sample(frac=majority_fraction, random_state=10701, ignore_index=True)

            # combine back
            df = pd.concat([minority_df, majority_df], ignore_index=True)

        # shuffle all rows
        if shuffle:
            df = df.sample(frac=1.0, random_state=seed, ignore_index=True)

        # keep only first 'train_fraction*100' percent for train, remaining for test (balance classes)
        labels = df['expression_label'].unique().tolist()
        temp_df = pd.DataFrame(columns=df.columns)
        for idx, label in enumerate(labels):
            label_df = df[df['expression_label'] == label]
            if self.type == "train":
                if idx == 0:
                    temp_df = label_df.iloc[:int(self.train_fraction * len(label_df))]
                else:
                    temp_df = pd.concat([temp_df, label_df.iloc[:int(self.train_fraction * len(label_df))]], ignore_index=True)
            else:
                if idx == 0:
                    temp_df = label_df.iloc[int(self.train_fraction * len(label_df)):]
                else:
                    temp_df = pd.concat([temp_df, label_df.iloc[int(self.train_fraction * len(label_df)):]], ignore_index=True)

        self.df = temp_df.reset_index(drop=True)

        self.length = self.df.shape[0]

        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tf_name = row["tf_name"]
        global_gene_name = row["gene_name"]
        y = row["expression_label"]

        tf_seq = self.tf_seqs[tf_name]
        gene_seq = self.gene_seqs[global_gene_name]

        # x = [tf_seq, gene_seq]

        x = {
            "tf_name": tf_name,
            "tf_seq": tf_seq,
            "gene_name": global_gene_name,
            "gene_seq": gene_seq
        }

        # convert y to tensor
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x, y_tensor


def perturbseq_collate(batch):
    """Collate function for PerturbSeqDataset.
    Returns a tuple (list_of_string_sequences, labels_tensor).
    """
    inputs, labels = zip(*batch)
    labels = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.long) for l in labels])

    return list(inputs), labels

def perturbseq_collate_2(batch):
    xs, ys = zip(*batch)
    return xs, torch.stack(ys)



def get_dataloader(
    parquet_path: str = "tf_gene_expression_labeled.parquet",
    tf_sequences_path: str = "tf_sequences.pkl",
    gene_sequences_path: str = "gene_sequences_4000bp.pkl",
    batch_size: int = 32,
    shuffle: bool = True,
    type: str = "train",
    train_fraction: float = 0.8,
    majority_fraction: float = 1.0,
    seed: int = 10701,
):
    """Utility that returns a DataLoader for the whole parquet file.
    """
    ds = PairPerturbSeqDataset(
        parquet_path=parquet_path,
        tf_sequences_path=tf_sequences_path,
        gene_sequences_path=gene_sequences_path,
        type=type,
        train_fraction=train_fraction,
        majority_fraction=majority_fraction,
        shuffle=True,
        seed=seed
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=perturbseq_collate_2,
        drop_last=False,
    )

    return loader
