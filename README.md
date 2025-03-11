# EvaOpt - Local LLM Optimization Engine

EvaOpt 是一個為大型語言模型（LLM）優化設計的高性能引擎，專門針對 Apple Silicon 架構優化。本項目結合了 Rust 的高性能和 Python 的易用性，為本地 LLM 部署提供完整的優化解決方案。

## 特點

- 🚀 使用 Rust 實現的高性能核心優化引擎
- 🐍 Python 友好的高層接口
- 🍎 針對 Apple Silicon (M1/M2/M3) 優化
- 📊 支持模型量化（INT4/INT8）
- 💾 智能內存管理和優化
- 🔄 動態張量優化
- 🛠 完整的工具集和示例

## 系統要求

- macOS 運行 Apple Silicon (M1/M2/M3) 處理器
- Python 3.9+
- Rust 1.75+
- 建議內存 16GB+

## 安裝

1. 克隆倉庫：
```bash
git clone https://github.com/yourusername/evaopt.git
cd evaopt
```

2. 創建並激活虛擬環境：
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
```

3. 安裝依賴：
```bash
# 安裝 Rust 工具鏈（如果未安裝）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安裝 Python 依賴
pip install -r requirements.txt

# 安裝開發版本
pip install -e .
```

## 項目結構

```
evaopt/
├── rust/              # Rust 核心實現
│   ├── src/          # 源代碼
│   └── build.rs      # 構建腳本
├── python/           # Python 綁定和高層接口
│   └── evaopt/      # Python 包
│       ├── core/    # 核心功能
│       └── utils/   # 工具函數
├── examples/         # 使用示例
└── benchmarks/       # 性能測試
```

## 快速開始

1. 基本優化示例：
```python
from evaopt import Optimizer, ModelConfig

# 配置優化器
config = ModelConfig(
    model_type="llama2",
    quantization_bits=8,
    use_fp16=True,
    max_memory_gb=24.0,
    device="mps"
)

# 創建優化器
optimizer = Optimizer(config)

# 優化模型
optimized_model = optimizer.optimize_model(model)
```

2. 運行完整示例：
```bash
python examples/optimize_llm.py
```

## 主要功能

- **模型優化**：
  - 智能張量優化
  - 自動量化（INT4/INT8）
  - 內存使用優化
  - 設備特定優化

- **內存管理**：
  - 動態內存分配
  - 智能緩存管理
  - 內存使用監控

- **性能優化**：
  - MPS 加速支持
  - 並行計算優化
  - 低精度推理

## 貢獻指南

1. Fork 本倉庫
2. 創建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

## 許可證

MIT License

## 致謝

感謝所有為本項目做出貢獻的開發者。特別感謝：
- Rust 社區
- PyTorch 團隊
- Hugging Face 團隊 