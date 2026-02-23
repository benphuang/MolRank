#!/usr/bin/env python3
"""
Inspect the checkpoint to determine the exact model configuration.
"""
import os
import sys
import torch

def inspect_checkpoint():
    """Load and inspect the checkpoint structure."""
    
    base_dir = os.getcwd()
    simson_dir = os.path.join(base_dir, "SimSon")
    checkpoint_path = os.path.join(simson_dir, "models/pretrained/pretrained_best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("SimSon Checkpoint Inspector")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if it's a dict or direct state_dict
    if isinstance(ckpt, dict):
        print(f"\nCheckpoint type: dict")
        print(f"Keys: {list(ckpt.keys())}")
        
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print("\nUsing 'model_state_dict' key")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("\nUsing 'state_dict' key")
        else:
            state_dict = ckpt
            print("\nUsing checkpoint as state_dict directly")
    else:
        state_dict = ckpt
        print(f"\nCheckpoint type: {type(ckpt)}")
    
    print("\n" + "=" * 70)
    print("Analyzing model architecture from checkpoint...")
    print("=" * 70)
    
    # Analyze key patterns
    layer_indices = set()
    embedding_dims = set()
    max_positions = set()
    vocab_sizes = set()
    
    for key, value in state_dict.items():
        # Find layer indices
        if "encoder.attn_layers.layers." in key:
            parts = key.split(".")
            layer_idx = int(parts[3])
            layer_indices.add(layer_idx)
        
        # Find embedding dimensions
        if "token_emb.emb.weight" in key:
            vocab_sizes.add(value.shape[0])
            embedding_dims.add(value.shape[1])
            print(f"\n📊 Token embedding: {key}")
            print(f"   Shape: {value.shape}")
            print(f"   → Vocab size: {value.shape[0]}")
            print(f"   → d_model: {value.shape[1]}")
        
        if "pos_emb.emb.weight" in key:
            max_positions.add(value.shape[0])
            print(f"\n📊 Positional embedding: {key}")
            print(f"   Shape: {value.shape}")
            print(f"   → Max length: {value.shape[0]}")
            print(f"   → d_model: {value.shape[1]}")
        
        if "linear.weight" in key and "encoder" not in key:
            print(f"\n📊 Final linear projection: {key}")
            print(f"   Shape: {value.shape}")
            print(f"   → Output dim: {value.shape[0]}")
            print(f"   → Input dim: {value.shape[1]}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    
    # Determine number of layers
    max_layer = max(layer_indices)
    # In x-transformers, each "block" = 2 layers (attention + FF)
    # So total sub-layers = max_layer + 1
    # Total transformer blocks = (max_layer + 1) / 2
    num_sublayers = max_layer + 1
    
    print(f"\n✓ Vocabulary size: {list(vocab_sizes)[0] if vocab_sizes else 'Unknown'}")
    print(f"✓ d_model: {list(embedding_dims)[0] if embedding_dims else 'Unknown'}")
    print(f"✓ Max sequence length: {list(max_positions)[0] if max_positions else 'Unknown'}")
    print(f"✓ Layer indices found: 0 to {max_layer}")
    print(f"✓ Total sub-layers: {num_sublayers}")
    print(f"✓ Transformer blocks: {num_sublayers // 2}")
    
    print("\n" + "=" * 70)
    print("CORRECT CONFIGURATION FOR YOUR CODE")
    print("=" * 70)
    
    d_model = list(embedding_dims)[0] if embedding_dims else None
    max_len = list(max_positions)[0] if max_positions else None
    vocab = list(vocab_sizes)[0] if vocab_sizes else None
    
    print("\nclass Args:")
    print(f"    dic_size = {vocab}")
    print(f"    max_len = {max_len}")
    print(f"    d_model = {d_model}")
    print(f"    nlayers = {num_sublayers}  # This is the number of sub-layers, not blocks!")
    print(f"    nhead = 8  # Assuming 8 heads (check if d_model % nhead == 0)")
    
    if d_model and d_model % 8 != 0:
        print(f"\n⚠️  WARNING: d_model={d_model} is not divisible by 8!")
        print(f"   You may need to adjust nhead to: {d_model // 64} or another divisor")
    
    print("\n" + "=" * 70)
    print("Sample checkpoint keys:")
    print("=" * 70)
    for i, key in enumerate(sorted(state_dict.keys())):
        print(f"  {key}")
    print(f"({len(state_dict)} total keys)")

if __name__ == "__main__":
    try:
        inspect_checkpoint()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
