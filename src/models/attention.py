"""Grouped Query Attention implementation."""

import torch
import torch.nn as nn
from src.utils.rope import apply_rope
from .layers import RMSNorm


class GroupedQueryAttention(nn.Module):
    """Grouped query attention - saves memory by sharing k,v across heads."""
    
    def __init__(self, d_in, n_heads, n_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert n_heads % n_kv_groups == 0, "heads must be divisible by kv groups"
        
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups  
        self.group_size = n_heads // n_kv_groups  # how many q heads per kv head
        
        # calculate head dimension if not provided
        if head_dim is None:
            assert d_in % n_heads == 0, "d_in must divide evenly by n_heads"
            head_dim = d_in // n_heads
            
        self.head_dim = head_dim
        self.d_out = n_heads * head_dim
        
        # projection layers
        self.q_proj = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_in, n_kv_groups * head_dim, bias=False, dtype=dtype) 
        self.v_proj = nn.Linear(d_in, n_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        
        # optional query/key normalization
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos_vals, sin_vals):
        b, seq_len, _ = x.shape
        
        # project to q, k, v
        q = self.q_proj(x)  # (b, seq_len, n_heads * head_dim)
        k = self.k_proj(x)  # (b, seq_len, n_kv_groups * head_dim)  
        v = self.v_proj(x)  # (b, seq_len, n_kv_groups * head_dim)
        
        # reshape to separate heads
        q = q.view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)
        
        # apply normalization if enabled
        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)
            
        # apply rotary position encoding
        q = apply_rope(q, cos_vals, sin_vals)
        k = apply_rope(k, cos_vals, sin_vals)
        
        # expand k,v to match number of query heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        
        # compute attention scores and apply mask
        scores = q @ k.transpose(2, 3)  # (b, heads, seq_len, seq_len)
        scores = scores.masked_fill(mask, -torch.inf)  # mask future tokens
        weights = torch.softmax(scores / self.head_dim**0.5, dim=-1)  # scale and softmax
        
        # apply attention weights to values
        out = (weights @ v).transpose(1, 2).reshape(b, seq_len, self.d_out)
        return self.out_proj(out)
