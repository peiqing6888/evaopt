# EvaOpt - Local LLM Optimization Engine

EvaOpt is a high-performance optimization engine designed for Large Language Models (LLMs), specifically optimized for Apple Silicon architecture. This project combines Rust's performance with Python's ease of use to provide a complete optimization solution for local LLM deployment.

## Features

- 🚀 High-performance core optimization engine implemented in Rust
- 🐍 Python-friendly high-level interface
- 🍎 Optimized for Apple Silicon (M1/M2/M3)
- 📊 Model quantization support (INT4/INT8)
- 💾 Smart memory management and optimization
- 🔄 Dynamic tensor optimization
- 🛠 Comprehensive toolset and examples

## System Requirements

- macOS with Apple Silicon (M1/M2/M3) processor
- Python 3.9+
- Rust 1.75+
- Recommended memory: 16GB+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/evaopt.git
cd evaopt
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
```

3. Install dependencies:
```bash
# Install Rust toolchain (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install -r requirements.txt

# Install development version
pip install -e .
```

## Project Structure

```
evaopt/
├── rust/              # Rust core implementation
│   ├── src/          # Source code
│   └── build.rs      # Build script
├── python/           # Python bindings and high-level interface
│   └── evaopt/      # Python package
│       ├── core/    # Core functionality
│       └── utils/   # Utility functions
├── examples/         # Usage examples
└── benchmarks/       # Performance tests
```

## Quick Start

1. Basic optimization example:
```python
from evaopt import Optimizer, ModelConfig

# Configure optimizer
config = ModelConfig(
    model_type="llama2",
    quantization_bits=8,
    use_fp16=True,
    max_memory_gb=24.0,
    device="mps"
)

# Create optimizer
optimizer = Optimizer(config)

# Optimize model
optimized_model = optimizer.optimize_model(model)
```

2. Run complete example:
```bash
python examples/optimize_llm.py
```

## Core Features

- **Model Optimization**:
  - Intelligent tensor optimization
  - Automatic quantization (INT4/INT8)
  - Memory usage optimization
  - Device-specific optimization

- **Memory Management**:
  - Dynamic memory allocation
  - Smart cache management
  - Memory usage monitoring

- **Performance Optimization**:
  - MPS acceleration support
  - Parallel computation optimization
  - Low-precision inference

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Submit Pull Request

## License

MIT License

## Acknowledgments

Thanks to all developers who contributed to this project. Special thanks to:
- Rust Community
- PyTorch Team
- Hugging Face Team 