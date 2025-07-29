"""Utilities for loading pretrained weights."""

import json
import os
import torch
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download


def load_pretrained_weights(model, config, weights_dict):
    """Load weights from HuggingFace checkpoint into our model."""
    def assign_weight(left, right, name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch for {name}: {left.shape} vs {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

    # load embedding weights
    model.tok_emb.weight = assign_weight(
        model.tok_emb.weight, 
        weights_dict["model.embed_tokens.weight"], 
        "embedding"
    )

    # load transformer layer weights
    for layer_idx in range(config["n_layers"]):
        block = model.layers[layer_idx] 
        attn = block.attn
        
        # attention projection weights
        attn.q_proj.weight = assign_weight(
            attn.q_proj.weight,
            weights_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
            f"layer_{layer_idx}_q_proj"
        )
        attn.k_proj.weight = assign_weight(
            attn.k_proj.weight,
            weights_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
            f"layer_{layer_idx}_k_proj"
        )
        attn.v_proj.weight = assign_weight(
            attn.v_proj.weight,
            weights_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            f"layer_{layer_idx}_v_proj"
        )
        attn.out_proj.weight = assign_weight(
            attn.out_proj.weight,
            weights_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            f"layer_{layer_idx}_out_proj"
        )
        
        # qk normalization weights if they exist
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            attn.q_norm.weight = assign_weight(
                attn.q_norm.weight,
                weights_dict[f"model.layers.{layer_idx}.self_attn.q_norm.weight"],
                f"layer_{layer_idx}_q_norm"
            )
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            attn.k_norm.weight = assign_weight(
                attn.k_norm.weight,
                weights_dict[f"model.layers.{layer_idx}.self_attn.k_norm.weight"],
                f"layer_{layer_idx}_k_norm"
            )
        
        # layer normalization weights
        block.norm1.weight = assign_weight(
            block.norm1.weight,
            weights_dict[f"model.layers.{layer_idx}.input_layernorm.weight"],
            f"layer_{layer_idx}_norm1"
        )
        block.norm2.weight = assign_weight(
            block.norm2.weight,
            weights_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
            f"layer_{layer_idx}_norm2"
        )
        
        # feedforward weights
        block.ff.gate.weight = assign_weight(
            block.ff.gate.weight,
            weights_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"],
            f"layer_{layer_idx}_ff_gate"
        )
        block.ff.up.weight = assign_weight(
            block.ff.up.weight,
            weights_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"],
            f"layer_{layer_idx}_ff_up"
        )
        block.ff.down.weight = assign_weight(
            block.ff.down.weight,
            weights_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"],
            f"layer_{layer_idx}_ff_down"
        )
    
    # final layer norm and output head
    model.final_norm.weight = assign_weight(
        model.final_norm.weight, 
        weights_dict["model.norm.weight"], 
        "final_norm"
    )
    
    if "lm_head.weight" in weights_dict:
        model.lm_head.weight = assign_weight(
            model.lm_head.weight, 
            weights_dict["lm_head.weight"], 
            "lm_head"
        )
    else:
        # weight tying: reuse embedding weights for output
        print("Using weight tying for output head")
        model.lm_head.weight = assign_weight(
            model.lm_head.weight, 
            weights_dict["model.embed_tokens.weight"], 
            "lm_head_tied"
        )


def download_model_weights(model_size, use_reasoning=True):
    """Download model weights from HuggingFace."""
    if use_reasoning:
        repo_id = f"Qwen/Qwen3-{model_size}"
    else:
        repo_id = f"Qwen/Qwen3-{model_size}-Base"

    local_dir = Path(repo_id).parts[-1]  # extract folder name

    print(f"Downloading weights from {repo_id}...")

    if model_size == "0.6B":
        # small model has single safetensors file
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights = load_file(weights_file)
    else:
        # larger models are sharded across multiple files
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        index_file = os.path.join(repo_dir, "model.safetensors.index.json")
        
        with open(index_file, "r") as f:
            index = json.load(f)
        
        weights = {}
        # load all shard files and combine
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights.update(shard)

    return weights, repo_id
