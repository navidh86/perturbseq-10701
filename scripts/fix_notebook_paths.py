"""Simple script to normalize data/embed paths across notebooks and python files.

Replacements performed:
- tf_sequences.pkl -> data/tf_sequences.pkl
- gene_sequences_4000bp.pkl -> data/gene_sequences_4000bp.pkl
- tf_gene_expression.parquet -> data/tf_gene_expression.parquet
- perturbseq_dataset_50.parquet -> data/perturbseq_dataset_50.parquet
- ./embeds/ -> embeds/

This edits files in-place. Run from repository root.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERNS = [
    ("tf_sequences.pkl", "data/tf_sequences.pkl"),
    ("gene_sequences_4000bp.pkl", "data/gene_sequences_4000bp.pkl"),
    ("tf_gene_expression.parquet", "data/tf_gene_expression.parquet"),
    ("perturbseq_dataset_50.parquet", "data/perturbseq_dataset_50.parquet"),
    ("./embeds/", "embeds/"),
]

EXTS = [".ipynb", ".py"]

modified = []
for p in REPO.rglob("*"):
    if p.suffix.lower() in EXTS:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        orig = text
        for a, b in PATTERNS:
            text = text.replace(a, b)
        if text != orig:
            p.write_text(text, encoding="utf-8")
            modified.append(str(p.relative_to(REPO)))

print("Modified files:")
for m in modified:
    print(m)

if not modified:
    print("No files changed.")

# Optionally show a few examples of replacements
print("\nSample replacements done:")
for a, b in PATTERNS:
    print(f"{a} -> {b}")
