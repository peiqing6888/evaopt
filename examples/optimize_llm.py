"""
示例：優化並運行大型語言模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaopt import Optimizer, ModelConfig
from evaopt.utils import optimize_memory

def main():
    # 1. 配置優化器
    config = ModelConfig(
        model_type="llama2",
        quantization_bits=8,
        use_fp16=True,
        max_memory_gb=24.0,
        device="mps"
    )
    
    optimizer = Optimizer(config)
    
    # 2. 加載模型
    print("正在加載模型...")
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # 需要 HuggingFace 訪問權限
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. 優化模型
    print("正在優化模型...")
    model = optimizer.optimize_model(model)
    
    # 4. 運行推理
    print("運行推理測試...")
    prompt = "請用中文回答：什麼是機器學習？"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n輸入: {prompt}")
    print(f"輸出: {response}")
    
    # 5. 顯示內存使用情況
    memory_stats = optimizer.get_memory_stats()
    print("\n內存使用情況:")
    for key, value in memory_stats.items():
        print(f"{key}: {value:.2f} GB")

if __name__ == "__main__":
    main() 