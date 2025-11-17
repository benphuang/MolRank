#!/usr/bin/env python3
"""
SETUP REQUIRED BEFORE RUNNING:
1. Download SimSon repository files from:
   https://drive.google.com/drive/folders/11vVMQrNog23Xjtb53CTAJ0loQXdmPwEv?usp=drive_link
   
2. Ensure the following structure exists:
   SimSon/
   ├── data/
   │   └── tokenizer/
   │       └── pubchem_part_tokenizer.json  <- Download this
   └── models/
       └── pretrained/
           └── pretrained_best_model.pth    <- Download this

3. Clone the SimSon repository:
   git clone https://github.com/lee00206/SimSon.git
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Make SimSon source importable
sys.path.append(os.path.join(os.getcwd(), "SimSon"))

from model import Xtransformer_Encoder


# ---------------- Tokenizer wrapper ---------------- #
from tokenizers import Tokenizer

class SimpleSmilesTokenizer:
    """Wrapper for HuggingFace Tokenizer JSON used by SimSon."""
    def __init__(self, vocab_file, max_length=512):  # Changed from 256 to 512
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"Tokenizer file not found: {vocab_file}\n"
                f"Please download it from:\n"
                f"https://drive.google.com/drive/folders/11vVMQrNog23Xjtb53CTAJ0loQXdmPwEv?usp=drive_link\n"
                f"and place it in: SimSon/data/tokenizer/"
            )
        
        self.tokenizer = Tokenizer.from_file(vocab_file)
        self.max_length = max_length
        # the BPE tokenizer used [PAD] = id 0
        self.pad_id = 0

    def encode(self, smiles):
        enc = self.tokenizer.encode(smiles)
        ids = enc.ids[:self.max_length]
        # pad
        if len(ids) < self.max_length:
            ids += [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# ---------------- Dataset ---------------- #
class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer):
        self.smiles = smiles_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.smiles[idx])


def collate_fn(batch):
    return torch.stack(batch, dim=0)


# ---------------- Main script ---------------- #
def main():
    base_dir = os.getcwd()
    simson_dir = os.path.join(base_dir, "SimSon")

    # Check if SimSon directory exists
    if not os.path.exists(simson_dir):
        print("❌ ERROR: SimSon directory not found!")
        print("\nPlease run:")
        print("  git clone https://github.com/lee00206/SimSon.git")
        sys.exit(1)

    # paths
    vocab_path = os.path.join(simson_dir, "data/tokenizer/pubchem_part_tokenizer.json")
    checkpoint_path = os.path.join(simson_dir, "models/pretrained/pretrained_best_model.pth")

    # Check if required files exist
    if not os.path.exists(vocab_path):
        print(f"❌ ERROR: Tokenizer file not found at: {vocab_path}")
        print("\nPlease download the required files from:")
        print("https://drive.google.com/drive/folders/11vVMQrNog23Xjtb53CTAJ0loQXdmPwEv?usp=drive_link")
        print("\nRequired files:")
        print("  - SimSon/data/tokenizer/pubchem_part_tokenizer.json")
        print("  - SimSon/models/pretrained/pretrained_best_model.pth")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Model checkpoint not found at: {checkpoint_path}")
        print("\nPlease download from the Google Drive link above.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # instantiate tokenizer
    print(f"Loading tokenizer from: {vocab_path}")
    tokenizer = SimpleSmilesTokenizer(vocab_path, max_length=512)  # Changed from 256

    # Build args object matching the actual checkpoint
    # The checkpoint inspector shows the REAL configuration:
    # - Layers 0-3 only = 4 sub-layers = 2 transformer blocks
    # - d_model = 768
    # - max_len = 512
    class Args:
        dic_size = len(tokenizer.tokenizer.get_vocab())  # 300
        max_len = 512   # Maximum sequence length
        d_model = 768   # Model dimension
        nlayers = 2     # Only 4 sub-layers in the checkpoint! (2 blocks)
        nhead = 8       # Number of attention heads
    args = Args()

    # load model
    print(f"Loading model from: {checkpoint_path}")
    model = Xtransformer_Encoder(args)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        print(dict)
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
    
    print(f"Model architecture:")
    print(f"  Vocabulary size: {args.dic_size}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Number of layers: {args.nlayers}")
    print(f"  Number of heads: {args.nhead}")

    # your dataset
    smiles_list = [
        "CCO",                       # ethanol
        "C1=CC=CC=C1",               # benzene
        "CC(=O)OC1=CC=CC=C1C(=O)O"   # aspirin
    ]
    dataset = SmilesDataset(smiles_list, tokenizer)
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    print("Generating embeddings...")
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # The model outputs (batch, 512) after the final linear layer
            # This is the contrastive learning projection dimension
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)
    output_path = os.path.join(base_dir, "simson_embeddings.npy")
    np.save(output_path, embeddings)
    print(f"✅ Saved embeddings to {output_path}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Expected shape: ({len(smiles_list)}, 512)")
    
    # Show sample embeddings
    print(f"\nSample embeddings for first molecule:")
    print(f"   SMILES: {smiles_list[0]}")
    print(f"   Embedding (first 10 dims): {embeddings[0][:10]}")


if __name__ == "__main__":
    main()