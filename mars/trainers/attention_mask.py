"""
MARS training attention mask.

During MARS training, the input is constructed as [noisy | clean] where:
  - noisy (positions 0..n-1): fully masked tokens to be predicted
  - clean (positions n..2n-1): original unmasked tokens providing context

The attention mask enforces three visibility rules:
  1. Clean context (clean-to-clean): strict causal -- each clean token sees
     only previous clean tokens.
  2. Noisy block (noisy-to-noisy): strict causal within the same block --
     each noisy token sees only previous noisy tokens that belong to the
     same block.
  3. Noisy sees clean (noisy-to-clean): inter-block causal -- each noisy
     token in block k can attend to all clean tokens in blocks < k.

These rules prevent information leakage while allowing the model to learn
block-parallel generation from clean context.
"""

import torch


def mars_attention_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the MARS training attention mask.

    Args:
        b: Batch dimension (unused, for flex_attention compatibility).
        h: Head dimension (unused, for flex_attention compatibility).
        q_idx: Query position indices.
        kv_idx: Key/value position indices.
        block_size: Number of tokens per block.
        n: Length of each stream (noisy or clean). The full sequence
           has length 2*n with layout [noisy | clean].

    Returns:
        Boolean mask of shape (Q_LEN, KV_LEN) where True means "allowed
        to attend".

    Structure:
        - Clean Context: Strict Causal
        - Noisy Block: Strict Causal (within-block only)
        - Noisy sees Clean: Inter-Block Causal
    """
    # Indicate whether token belongs to Noisy (0) or Clean (1)
    x0_flag_q = (q_idx >= n).int()
    x0_flag_kv = (kv_idx >= n).int()

    # Logical positions (aligned to 0..n-1)
    pos_q = torch.where(x0_flag_q == 1, q_idx - n, q_idx)
    pos_kv = torch.where(x0_flag_kv == 1, kv_idx - n, kv_idx)

    block_q = pos_q // block_size
    block_kv = pos_kv // block_size

    # 1. Clean Context Self-Attention -- Strict Causal
    clean_causal = (x0_flag_q == 1) & (x0_flag_kv == 1) & (pos_q >= pos_kv)

    # 2. Noisy Block Self-Attention -- Causal AND same block
    noisy_block_causal = (x0_flag_q == 0) & (x0_flag_kv == 0) & (block_q == block_kv) & (pos_q >= pos_kv)

    # 3. Noisy sees Clean Context -- Inter-Block Causal (no leakage)
    noisy_sees_clean = (x0_flag_q == 0) & (x0_flag_kv == 1) & (block_q > block_kv)

    # 4. Combine Masks
    return clean_causal | noisy_block_causal | noisy_sees_clean
