#!/usr/bin/env python3
"""
Compute SimSon embeddings for an arbitrary CSV file containing SMILES.

Usage:
    python batch_embed.py my_dataset.csv
Output:
    simson_embeddings.npy (NumPy array, same order as input)
"""

import sys
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import classes directly from featurize.py
from featurize import SimpleSmilesTokenizer, SmilesDataset, collate_fn, Xtransformer_Encoder

def embed_smiles(smiles_list, model, tokenizer, device):
    dataset = SmilesDataset(smiles_list, tokenizer)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    all_emb = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch)
            all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_embed.py <input_csv_with_smiles_column>")
        sys.exit(1)
    input_csv = sys.argv[1]
    df = pd.read_csv(input_csv)
    if 'smiles' not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    base_dir = os.getcwd()
    simson_dir = os.path.join(base_dir, "SimSon")
    vocab_path = os.path.join(simson_dir, "data/tokenizer/pubchem_part_tokenizer.json")
    checkpoint_path = os.path.join(simson_dir, "models/pretrained/pretrained_best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = SimpleSmilesTokenizer(vocab_path, max_length=512)

    class Args:
        dic_size = len(tokenizer.tokenizer.get_vocab())
        max_len = 512
        d_model = 768
        nlayers = 2
        nhead = 8
    args = Args()

    model = Xtransformer_Encoder(args)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    print(f"Computing embeddings for {len(df)} molecules...")
    embeddings = embed_smiles(df['smiles'].tolist(), model, tokenizer, device)

    np.save("simson_embeddings.npy", embeddings)
    print(f"✅ Saved embeddings: simson_embeddings.npy (shape={embeddings.shape})")

if __name__ == "__main__":
    main()
