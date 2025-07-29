"""Performance benchmarking utilities."""

import time
import torch


def speed_benchmark(model, tokenizer, num_runs=4):
    """Benchmark model inference speed."""
    print(f"\nSpeed Benchmarks:")
    
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "List the benefits of renewable energy."
    ]
    
    total_time = 0
    total_tokens = 0
    
    for i, prompt in enumerate(test_prompts):
        input_ids = tokenizer.encode(prompt)
        start = time.time()
        
        from ..utils.generation import generate_text
        output = generate_text(
            model=model,
            token_ids=torch.tensor(input_ids, device=model.tok_emb.weight.device).unsqueeze(0),
            max_tokens=50,
            temp=0.0
        )
        
        elapsed = time.time() - start
        tokens_generated = output.shape[1] - len(input_ids)
        
        total_time += elapsed
        total_tokens += tokens_generated
        
        print(f"  Prompt {i+1}: {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tok/s)")
    
    print(f"  Average: {total_tokens/total_time:.1f} tokens/second")
    return total_tokens/total_time


def memory_analysis(model, seq_len=1024):
    """Analyze memory usage for different batch sizes."""
    print(f"\nMemory Analysis (sequence length: {seq_len}):")
    
    device = next(model.parameters()).device
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            _ = model(dummy_input)
            
        if torch.cuda.is_available() and device.type == 'cuda':
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Batch size {batch_size:2d}: {memory_gb:.2f} GB")
        else:
            print(f"  Batch size {batch_size:2d}: Memory tracking not available on {device}")


def parameter_analysis(model):
    """Analyze model parameter distribution."""
    component_params = {}
    
    # embedding layer
    emb_params = model.tok_emb.weight.numel()
    component_params['Token Embedding'] = emb_params
    
    # transformer layers
    layer_params = 0
    for layer in model.layers:
        layer_params += sum(p.numel() for p in layer.parameters())
    component_params['Transformer Layers'] = layer_params
    
    # final components
    final_norm_params = sum(p.numel() for p in model.final_norm.parameters())
    lm_head_params = model.lm_head.weight.numel()
    
    component_params['Final Norm'] = final_norm_params
    component_params['LM Head'] = lm_head_params
    
    total = sum(component_params.values())

    print("\nParameter Breakdown:")
    for component, count in component_params.items():
        percentage = (count / total) * 100
        print(f"  {component}: {count:,} ({percentage:.1f}%)")
    
    return component_params
