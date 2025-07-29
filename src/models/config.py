"""Model configuration definitions for different Qwen3 model sizes."""

import torch

MODEL_CONFIGS = {
    "0.6B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 28,
        "hidden_dim": 3072,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
    },
    "1.7B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2048,
        "n_heads": 16,
        "n_layers": 28,
        "hidden_dim": 6144,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
    },
    "4B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 2560,
        "n_heads": 32,
        "n_layers": 36,
        "hidden_dim": 9728,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
    },
    "8B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 4096,
        "n_heads": 32,
        "n_layers": 36,
        "hidden_dim": 12288,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
    },
    "14B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 40,
        "n_layers": 40,
        "hidden_dim": 17408,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
    },
    "32B": {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 64,
        "n_layers": 64,
        "hidden_dim": 25600,
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
    }
}

def get_config(model_size: str) -> dict:
    """Get configuration for specified model size."""
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Model size {model_size} not supported. "
                        f"Available sizes: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_size].copy()
