"""Text generation utilities."""

import torch


def generate_text(model, token_ids, max_tokens=150, context_size=None, temp=0.0, top_k=None, eos_id=None):
    """Generate text by predicting next tokens one by one."""
    if context_size is None:
        context_size = model.cfg["context_length"]
    
    for _ in range(max_tokens):
        # only use last context_size tokens to avoid memory issues
        context_ids = token_ids[:, -context_size:]
        
        with torch.no_grad():
            logits = model(context_ids)
            next_logits = logits[:, -1, :]  # only care about last position
        
        # apply top-k filtering if specified
        if top_k is not None:
            top_vals, _ = torch.topk(next_logits, top_k)
            min_val = top_vals[:, -1]
            next_logits = torch.where(
                next_logits < min_val, 
                torch.tensor(-torch.inf).to(next_logits.device), 
                next_logits
            )
        
        # apply temperature and sample
        if temp > 0.0:
            next_logits = next_logits / temp
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            # greedy decoding
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        
        # stop if eos token encountered
        if eos_id is not None and next_id.item() == eos_id:
            break
            
        # append new token
        token_ids = torch.cat((token_ids, next_id), dim=1)
    
    return token_ids


def calc_memory_usage(model, dtype=torch.float32):
    """Calculate model memory requirements."""
    total_params = 0
    total_grads = 0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        
        # add gradient memory if param requires grad
        if param.requires_grad:
            total_grads += param_count
    
    # add buffer memory (non-trainable tensors)
    total_buffers = sum(buf.numel() for buf in model.buffers())
    
    # bytes per element for given dtype
    bytes_per_elem = torch.tensor(0, dtype=dtype).element_size()
    total_bytes = (total_params + total_grads + total_buffers) * bytes_per_elem
    
    # convert to GB
    total_gb = total_bytes / (1024**3)
    return total_gb
