# QWEN3 MODEL IMPLEMENTATION

A clean, modular implementation of the Qwen3 language model family with support for multiple model sizes (0.6B to 32B parameters).

## FEATURES

- Multiple Model Sizes: Support for 0.6B, 1.7B, 4B, 8B, 14B, and 32B parameter models
- Grouped Query Attention: Memory-efficient attention mechanism
- Rotary Position Encoding: Advanced positional encoding for better sequence modeling
- Modular Architecture: Clean separation of concerns with well-organized code structure
- Performance Benchmarks: Built-in tools for speed and memory analysis

## INSTALLATION
First clone the repo
```
git clone https://github.com/namanraiyani/QWEN_From_Scratch.git
cd QWEN_From_Scratch
```
Then install the requirements
```
pip install -r requirements.txt
```
## QUICK START
```
from src.models.config import get_config
from src.models.transformer import Qwen3Model
from src.utils.tokenizer import Qwen3Tokenizer
from src.utils.generation import generate_text
```
**Load model**
```
config = get_config("0.6B")
model = Qwen3Model(config)
```
**Load tokenizer**
```
tokenizer = Qwen3Tokenizer("path/to/tokenizer.json")
```
**Generate text**
```
text = "What is machine learning?"
tokens = tokenizer.encode(text)
output = generate_text(model, tokens, max_tokens=100)
result = tokenizer.decode(output.tolist())
```
## STANDALONE NOTEBOOK
```
qwen3_implementation/notebook/QWENFrom_Scratch.ipynb
```
## PROJECT STRUCTURE
```
qwen3_implementation/
├── src/
│   ├── models/                 # Model architecture
│   ├── utils/                  # Utilities and helpers
│   └── evaluation/             # Benchmarking tools
├── scripts/                    # Demo and evaluation scripts
├── notebook/
│   ├── QWENFrom_Scratch.ipynb  # Complete Project as a standalone jupyter notebook
└── requirements.txt
```
## RUNNING THE DEMO
```
python scripts/demo.py
```
## MODEL SIZES
```
Size    Parameters    Embedding Dim    Layers    Heads
0.6B    ~600M         1024             28        16
1.7B    ~1.7B         2048             28        16
4B      ~4B           2560             36        32
8B      ~8B           4096             36        32
14B     ~14B          5120             40        40
32B     ~32B          5120             64        64
```
## USAGE EXAMPLES

**Basic Text Generation:**
    Load the 0.6B model and generate text for a simple prompt.

**Advanced Configuration:**
    Customize generation parameters like temperature, top-k sampling, and maximum tokens.

**Memory Optimization:**
    Use bfloat16 precision and appropriate batch sizes for your hardware.

**Benchmarking:**
    Run built-in speed and memory benchmarks to evaluate performance.

## REQUIREMENTS

- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- tokenizers 0.15.0 or higher
- huggingface_hub 0.20.0 or higher
- safetensors 0.4.0 or higher

## ARCHITECTURE DETAILS

The implementation follows the original Qwen3 architecture with:
- Grouped Query Attention for memory efficiency
- RMSNorm for improved training stability
- SwiGLU activation in feedforward layers
- Rotary Position Encoding for better positional understanding
