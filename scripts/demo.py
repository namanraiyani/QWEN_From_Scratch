#!/usr/bin/env python3
"""Main demo script for Qwen3 model."""

import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.config import get_config
from models.transformer import Qwen3Model
from utils.tokenizer import Qwen3Tokenizer
from utils.weight_loader import download_model_weights, load_pretrained_weights
from utils.generation import generate_text, calc_memory_usage
from evaluation.benchmarks import speed_benchmark, memory_analysis, parameter_analysis


def main():
    # Configuration
    MODEL_SIZE = "0.6B"  # Change as needed
    USE_REASONING = True
    
    print("="*50)
    print(f"QWEN3 {MODEL_SIZE} MODEL DEMO")
    print("="*50)
    
    # Load configuration
    config = get_config(MODEL_SIZE)
    print(f"Model configuration loaded for {MODEL_SIZE}")
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model
    torch.manual_seed(123)
    model = Qwen3Model(config)
    model.to(device)
    
    # Calculate and display memory usage
    print(f"\nMemory usage estimates:")
    print(f"  float32: {calc_memory_usage(model, torch.float32):.2f} GB")
    print(f"  bfloat16: {calc_memory_usage(model, torch.bfloat16):.2f} GB")
    
    # Download and load weights
    weights, repo_id = download_model_weights(MODEL_SIZE, USE_REASONING)
    load_pretrained_weights(model, config, weights)
    del weights  # Free memory
    print("✓ Model weights loaded successfully!")
    
    # Setup tokenizer
    if USE_REASONING:
        tokenizer_path = f"Qwen3-{MODEL_SIZE}/tokenizer.json"
    else:
        tokenizer_path = f"Qwen3-{MODEL_SIZE}-Base/tokenizer.json"
    
    tokenizer = Qwen3Tokenizer(
        tokenizer_path=tokenizer_path,
        repo_id=repo_id,
        add_gen_prompt=USE_REASONING,
        add_thinking=USE_REASONING
    )
    
    # Test generation
    test_prompt = "Give me a short introduction to large language models."
    print(f"\nGenerating text for: '{test_prompt}'")
    
    token_ids = tokenizer.encode(test_prompt)
    start_time = time.time()
    
    output_ids = generate_text(
        model=model,
        token_ids=torch.tensor(token_ids, device=device).unsqueeze(0),
        max_tokens=150,
        top_k=1,
        temp=0.0
    )
    
    gen_time = time.time() - start_time
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    
    print(f"\n✓ Generation completed!")
    print(f"  Time: {gen_time:.2f} seconds")
    print(f"  Tokens generated: {output_ids.shape[1] - len(token_ids)}")
    print(f"  Speed: {(output_ids.shape[1] - len(token_ids)) / gen_time:.1f} tokens/sec")
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Peak GPU memory: {max_memory:.2f} GB")
    
    print(f"\nGenerated text:\n{'-'*50}")
    print(output_text)
    print("-"*50)
    
    # Run benchmarks
    print("\n" + "="*50)
    print("RUNNING BENCHMARKS")
    print("="*50)
    
    parameter_analysis(model)
    speed_benchmark(model, tokenizer)
    memory_analysis(model, seq_len=512)
    
    print("\n" + "="*50)
    print("DEMO COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
