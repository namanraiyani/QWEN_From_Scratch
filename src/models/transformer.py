""" Transformer model ."""

import torch
import torch.nn as nn
from .layers import RMSNorm, FeedForward
from .attention import GroupedQueryAttention
from src.utils.rope import compute_rope_freqs


class TransformerBlock(nn.Module):
    """Single transformer layer with attention + feedforward."""
    
    def __init__(self, cfg):
        super().__init__()
        self.attn = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            n_heads=cfg["n_heads"], 
            head_dim=cfg["head_dim"],
            n_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)  # pre-attention norm
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)  # pre-ff norm

    def forward(self, x, mask, cos_vals, sin_vals):
        # attention block with residual connection
        residual = x
        x = self.norm1(x)  # pre-norm
        x = self.attn(x, mask, cos_vals, sin_vals) 
        x = x + residual   # residual connection
        
        # feedforward block with residual connection  
        residual = x
        x = self.norm2(x)  # pre-norm
        x = self.ff(x)
        x = x + residual   # residual connection
        
        return x


class Qwen3Model(nn.Module):
    """Main transformer model."""
    
    def __init__(self, cfg):
        super().__init__()
        
        # embedding layer converts tokens to vectors
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        
        # stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])
        
        # final norm and output projection
        self.final_norm = RMSNorm(cfg["emb_dim"]) 
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        
        # precompute rope frequencies
        if cfg["head_dim"] is None:
            hd = cfg["emb_dim"] // cfg["n_heads"]
        else:
            hd = cfg["head_dim"]
            
        cos_vals, sin_vals = compute_rope_freqs(
            head_dim=hd,
            base=cfg["rope_base"], 
            max_len=cfg["context_length"]
        )
        # register as buffers so they move with model to gpu/cpu
        self.register_buffer("cos_vals", cos_vals, persistent=False)
        self.register_buffer("sin_vals", sin_vals, persistent=False)
        self.cfg = cfg

    def forward(self, token_ids):
        # convert tokens to embeddings
        x = self.tok_emb(token_ids)
        
        # create causal mask to prevent looking at future tokens
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        
        # pass through all transformer layers
        for layer in self.layers:
            x = layer(x, mask, self.cos_vals, self.sin_vals)
            
        # final normalization and projection to vocab
        x = self.final_norm(x) 
        logits = self.lm_head(x.to(self.cfg["dtype"]))
        return logits
