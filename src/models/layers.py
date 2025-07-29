import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """ feedforward network with gating mechanism."""
    
    def __init__(self, cfg):
        super().__init__()
        # 3 linear layers: 2 for gating, 1 for output
        self.gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)  
        self.up = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.down = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        # SwiGLU activation: silu(gate) * up_proj
        gate_out = self.gate(x)
        up_out = self.up(x) 
        activated = nn.functional.silu(gate_out) * up_out  # element-wise multiply
        return self.down(activated)


class RMSNorm(nn.Module):
    """Root Mean Square normalization - alternative to LayerNorm."""
    
    def __init__(self, dim, eps=1e-6, bias=False, qwen3_compat=True):
        super().__init__()
        self.eps = eps  # small value to avoid division by zero
        self.qwen3_compat = qwen3_compat  # compatibility flag
        self.weight = nn.Parameter(torch.ones(dim))  # learnable scale
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None  # optional bias

    def forward(self, x):
        orig_dtype = x.dtype
        
        # convert to float32 for numerical stability
        if self.qwen3_compat:
            x = x.to(torch.float32)

        # compute rms and normalize
        var = x.pow(2).mean(dim=-1, keepdim=True)  # mean of squares
        normed = x * torch.rsqrt(var + self.eps)   # x / sqrt(variance)
        normed = normed * self.weight              # scale

        if self.bias is not None:
            normed = normed + self.bias  # shift if bias exists

        return normed.to(orig_dtype)  # back to original dtype
