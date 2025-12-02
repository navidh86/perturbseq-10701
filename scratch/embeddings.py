
import torch
import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

class NucleotideTransformer():
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)

    def tokenize(self, sequences):
        max_length = 50
        tokenized_sequences = self.tokenizer.batch_encode_plus(
            sequences,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )["input_ids"]
        tokenized_sequences = tokenized_sequences.to(self.model.device)
        attention_mask = tokenized_sequences != self.tokenizer.pad_token_id
        return tokenized_sequences, attention_mask

    def embed(self, sequences):
        tokenized_sequences, attention_mask = self.tokenize(sequences)
        tf_embeddings = self.model(
            tokenized_sequences,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        mean_tf_embeddings = torch.sum(
            attention_mask.unsqueeze(-1)*tf_embeddings.last_hidden_state,
            axis=1)/torch.sum(attention_mask, axis=1).unsqueeze(-1)
        return mean_tf_embeddings

def precompute_embeddings():
    parquet_path = "data/perturbseq_dataset_50.parquet"
    out_dir = "embeds"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    seq_cols = [c for c in df.columns if c.endswith("sequence")]
    unique_seqs = {}
    for col in seq_cols:
        for s in df[col].dropna().unique():
            s = str(s).upper()
            if s not in unique_seqs:
                unique_seqs[s] = len(unique_seqs)
    print(f"Found {len(unique_seqs)} unique sequences")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = NucleotideTransformer(device=device)
    sequences = list(unique_seqs.keys())
    batch_size = 512
    embeddings = np.zeros((len(sequences), transformer.model.config.hidden_size), dtype=np.float32)

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        with torch.no_grad():
            be = transformer.embed(batch)
            be = be.cpu().numpy()
        embeddings[i:i+len(batch)] = be
        print(f"Embedded {i+len(batch)}/{len(sequences)}")

    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, "seq2idx.json"), "w") as f:
        json.dump(unique_seqs, f)

if __name__ == "__main__":
    precompute_embeddings()