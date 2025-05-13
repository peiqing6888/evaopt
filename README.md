# EvaOpt - LLM Optimization Engine

High-performance optimization engine for Large Language Models (LLMs) on Apple Silicon, combining Rust's performance with Python's ease of use.

## Getting Started

Welcome to EvaOpt! This guide will help you get started with our project, even if you're new to programming.

### What is EvaOpt?

EvaOpt is a tool that helps make artificial intelligence models (specifically, Large Language Models like ChatGPT) run faster and use less memory on Apple computers with M1/M2/M3 chips. Think of it like a compression tool that makes these AI models more efficient without losing their capabilities.

### Before You Begin

You'll need:

1. A Mac computer with Apple Silicon (M1, M2, or M3 chip)
2. Internet connection
3. Basic familiarity with using the Terminal (don't worry, we'll guide you!)

### Step-by-Step Installation

1. **Install Required Software**

   - Install Python (version 3.9 or newer)

     - Download from: https://www.python.org/downloads/
     - During installation, make sure to check "Add Python to PATH"
   - Install Rust

     - Open Terminal
     - Copy and paste this command:
       ```bash
       curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
       ```
     - Follow the on-screen instructions
2. **Get the Project**

   - Open Terminal
   - Run these commands one by one:
     ```bash
     # Download the project
     git clone https://github.com/yourusername/evaopt.git

     # Go to project folder
     cd evaopt

     # Create a virtual environment (like a separate space for the project)
     python3 -m venv venv

     # Activate the virtual environment
     source venv/bin/activate

     # Install required packages
     pip install -r requirements.txt
     pip install -e .
     ```

### Running Your First Optimization

Here's a simple example to get started:

```python
# Create a new file called 'first_test.py' and add this code:
from evaopt import Optimizer, ModelConfig

# Set up basic configuration
config = ModelConfig(
    model_type="llama2",  # Type of AI model
    quantization_bits=8,  # How much to compress
    use_fp16=True        # Use faster processing
)

# Create an optimizer
optimizer = Optimizer(config)

# This is where you would optimize your model
# For example:
# optimized_model = optimizer.optimize_model(your_model)
```

### Common Issues and Solutions

1. **"Command not found" errors**

   - Problem: Python or Rust commands aren't recognized
   - Solution: Make sure you've installed Python and Rust correctly and restart your Terminal
2. **Installation errors**

   - Problem: Packages fail to install
   - Solution: Try running:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt --no-cache-dir
     ```
3. **Memory errors**

   - Problem: Process uses too much memory
   - Solution: Try reducing the model size or increasing quantization_bits in the config
4. **Import errors**

   - Problem: Can't import evaopt
   - Solution: Make sure you're in the virtual environment:
     ```bash
     source venv/bin/activate
     ```

### Need Help?

If you encounter any issues not covered here, please:

1. Check if your Mac has Apple Silicon (M1/M2/M3)
2. Make sure all software is up to date
3. Try restarting your computer
4. Create an issue on our GitHub page

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
