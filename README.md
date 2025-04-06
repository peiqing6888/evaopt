# EvaOpt - LLM Optimization Engine

High-performance optimization engine for Large Language Models (LLMs) on Apple Silicon, combining Rust's performance with Python's ease of use.

## Features

- 🚀 High-performance Rust core engine
- 🔢 Matrix optimization methods (SVD, Low-rank, Sparse)
- 📊 Model quantization (INT4/INT8)
- 💾 Smart memory management
- 🍎 Apple Silicon optimization

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- Rust 1.75+

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/evaopt.git
cd evaopt

# Setup environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from evaopt import Optimizer, ModelConfig

# Configure optimizer
config = ModelConfig(
    model_type="llama2",
    quantization_bits=8,
    use_fp16=True
)

# Create optimizer
optimizer = Optimizer(config)

# Optimize model
optimized_model = optimizer.optimize_model(model)
```

## License

MIT License 

## Test Models & Results 🧪

### Matrix Tests
```python
# Block-sparse optimization (1000x1000)
optimizer = Optimizer(config)
result = optimizer.optimize(matrix)
# Compression: 40.36%, Error: 0.000003
```

### Language Models
```python
# GPT-2 (Demo)
model = GPT2LMHeadModel(config)

# Llama-2-7b-chat
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```

### Performance Highlights
- Block-sparse: 40% memory reduction, 0.001s processing time
- Matrix Compression: Up to 97.74% for rank-10 approximation
- LLM Inference: 26% speed improvement
- Memory Usage: Stable under 2GB for optimized models

## Optimization Methods 🚀

### Block-sparse Optimization
- Block sizes: 16x16 to 128x128
- Adaptive threshold selection
- Fast block-wise processing
- Minimal accuracy loss

### Matrix Methods
- SVD (Full/Truncated/Randomized)
- Low-rank approximation
- Sparse optimization
- Block-sparse compression

For more examples, check `examples/` directory. 