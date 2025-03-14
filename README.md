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