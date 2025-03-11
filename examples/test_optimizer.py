"""
測試優化引擎的基本功能
"""

import torch
from evaopt import Optimizer, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1. 初始化配置
    config = ModelConfig(
        model_type="llama2",
        quantization_bits=8,
        use_fp16=True,
        max_memory_gb=24.0,
        device="mps"
    )
    
    # 2. 創建優化器
    optimizer = Optimizer(config)
    
    # 3. 創建一個小型測試模型
    print("創建測試模型...")
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    
    # 4. 優化模型
    print("開始優化模型...")
    try:
        optimized_model = optimizer.optimize_model(model)
        print("模型優化成功！")
        
        # 5. 檢查內存使用情況
        memory_stats = optimizer.get_memory_stats()
        print("\n內存使用情況:")
        for key, value in memory_stats.items():
            print(f"{key}: {value:.2f} GB")
            
    except Exception as e:
        print(f"優化過程中出現錯誤: {e}")

if __name__ == "__main__":
    main() 