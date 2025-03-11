from setuptools import setup, find_packages

setup(
    name="evaopt",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "transformers>=4.36.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.41.0",
        "safetensors>=0.4.0",
    ],
    author="EvaOpt Team",
    description="高性能本地化大模型優化引擎",
    python_requires=">=3.9",
) 