"""
Example: Optimize and run a large language model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaopt import Optimizer, ModelConfig
from evaopt.utils import optimize_memory

def main():
    # 1. Configure optimizer
    config = ModelConfig(
        model_type="llama2",
        quantization_bits=8,
        use_fp16=True,
        max_memory_gb=24.0,
        device="mps"
    )
    
    optimizer = Optimizer(config)
    
    # 2. Load model
    print("Loading model...")
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Requires HuggingFace access
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Optimize model
    print("Optimizing model...")
    model = optimizer.optimize_model(model)
    
    # 4. Run inference
    print("Running inference test...")
    prompt = "Please explain what machine learning is:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nInput: {prompt}")
    print(f"Output: {response}")
    
    # 5. Display memory usage
    memory_stats = optimizer.get_memory_stats()
    print("\nMemory Usage:")
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f} GB")

if __name__ == "__main__":
    main() 