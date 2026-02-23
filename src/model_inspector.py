#!/usr/bin/env python3
"""
Diagnostic script to inspect the SimSon model architecture
and understand what inputs/outputs it expects.
"""
import os
import sys
import torch

sys.path.append(os.path.join(os.getcwd(), "SimSon"))

from model import Xtransformer_Encoder

def inspect_model():
    """Inspect the model architecture to understand input/output shapes."""
    
    # Build args matching the ACTUAL checkpoint (not the paper!)
    # The paper mentions d_model=512, nlayers=6
    # But the checkpoint uses d_model=768, nlayers=12
    class Args:
        dic_size = 300  # BPE tokenizer with 300 tokens
        max_len = 512   # Maximum sequence length
        d_model = 768   # Model dimension (checkpoint uses 768, not 512!)
        nlayers = 12    # Number of layers (checkpoint uses 12, not 6!)
        nhead = 8       # Number of attention heads
    
    args = Args()
    
    print("=" * 70)
    print("SimSon Model Architecture Inspector")
    print("=" * 70)
    print("\nModel Configuration:")
    print(f"  Vocabulary size: {args.dic_size}")
    print(f"  Max sequence length: {args.max_len}")
    print(f"  Model dimension (d_model): {args.d_model}")
    print(f"  Number of layers: {args.nlayers}")
    print(f"  Number of attention heads: {args.nhead}")
    
    # Create model
    model = Xtransformer_Encoder(args)
    model.eval()
    
    print("\n" + "=" * 70)
    print("Model Structure:")
    print("=" * 70)
    print(model)
    
    # Test with dummy input
    print("\n" + "=" * 70)
    print("Testing with dummy input:")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 256
    
    # Create dummy token IDs (integers from 0 to vocab_size-1)
    dummy_input = torch.randint(0, args.dic_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input dtype: {dummy_input.dtype}")
    print(f"Input range: [{dummy_input.min()}, {dummy_input.max()}]")
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\n✅ Model forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        if len(output.shape) == 3:
            print(f"\nOutput is 3D: (batch={output.shape[0]}, seq_len={output.shape[1]}, d_model={output.shape[2]})")
            print("⚠️  You need to pool this to get sentence embeddings!")
            print("   Use: output.mean(dim=1) for global average pooling")
            
            # Test pooling
            pooled = output.mean(dim=1)
            print(f"\nAfter pooling: {pooled.shape}")
            
        elif len(output.shape) == 2:
            print(f"\nOutput is 2D: (batch={output.shape[0]}, d_model={output.shape[1]})")
            print("✅ Already pooled - ready to use!")
        
    except Exception as e:
        print(f"\n❌ Model forward pass failed!")
        print(f"Error: {e}")
        print("\nPossible issues:")
        print("  1. Model expects embeddings, not token IDs")
        print("  2. Wrong input shape")
        print("  3. Model architecture mismatch with checkpoint")
    
    print("\n" + "=" * 70)
    print("Model Parameters:")
    print("=" * 70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("Layer-by-layer breakdown:")
    print("=" * 70)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:30s} {params:>15,} params")
    
    print("\n" + "=" * 70)
    print("According to the SimSon paper:")
    print("=" * 70)
    print("Expected flow:")
    print("  1. Tokenize SMILES → Token IDs")
    print("  2. Embed Token IDs → Embeddings (handled by model)")
    print("  3. Add positional encoding (handled by model)")
    print("  4. Pass through Transformer encoder")
    print("  5. Global average pooling")
    print("  6. Linear projection for contrastive learning")
    print("\nFor inference, you only need steps 1-5")
    print("The model should internally handle embedding and positional encoding.")

if __name__ == "__main__":
    try:
        inspect_model()
    except Exception as e:
        print(f"\n❌ Failed to inspect model: {e}")
        print("\nMake sure:")
        print("  1. SimSon directory exists")
        print("  2. model.py contains Xtransformer_Encoder class")
        sys.exit(1)
