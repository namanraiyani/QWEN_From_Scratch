"""Rotary Position Encoding (RoPE)."""

import torch


def compute_rope_freqs(head_dim, base=10_000, max_len=4096, dtype=torch.float32):
    """Compute rotary position encoding frequencies."""
    assert head_dim % 2 == 0, "head dim must be even for rope"

    # compute inverse frequencies for each dimension pair
    inv_freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype)[:(head_dim // 2)].float() / head_dim))
    
    # position indices from 0 to max_len-1
    pos = torch.arange(max_len, dtype=dtype)
    
    # compute angles: pos * inv_freq for each position-frequency pair
    angles = pos[:, None] * inv_freqs[None, :]  # shape: (max_len, head_dim//2)
    
    # duplicate angles to match full head dimension
    angles = torch.cat([angles, angles], dim=1)  # shape: (max_len, head_dim)
    
    # precompute cos and sin for efficiency
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    return cos_vals, sin_vals


def apply_rope(x, cos_vals, sin_vals):
    """Apply rotary encoding to input tensor."""
    # x shape: (batch, heads, seq_len, head_dim)
    b, h, seq_len, d = x.shape
    assert d % 2 == 0, "head dim must be even"
    
    # split into two halves for rotation
    x1 = x[..., :d//2]   # first half
    x2 = x[..., d//2:]   # second half
    
    # adjust cos/sin shapes to match input
    cos_vals = cos_vals[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,head_dim)
    sin_vals = sin_vals[:seq_len, :].unsqueeze(0).unsqueeze(0)
    
    # rotation: combine original and rotated components
    rotated = torch.cat((-x2, x1), dim=-1)  # rotate by 90 degrees
    result = (x * cos_vals) + (rotated * sin_vals)
    
    return result.to(dtype=x.dtype)
