# EPW-A Framework Installation Guide

## 概述

EPW-A（增强型专家路径水印）框架是一个用于MoE（专家混合）大语言模型的水印系统。本指南将帮助您安装和配置所有必要的依赖项。

## 系统要求

### 硬件要求
- **GPU**: 推荐NVIDIA GPU，至少8GB显存（用于Mixtral-8x7B）
- **RAM**: 至少16GB系统内存
- **存储**: 至少50GB可用空间（用于模型下载）

### 软件要求
- **Python**: 3.8或更高版本
- **CUDA**: 11.8或更高版本（如果使用GPU）
- **操作系统**: Linux, Windows, macOS

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd epw
```

### 2. 创建虚拟环境（推荐）
```bash
# 使用conda
conda create -n epw-a python=3.9
conda activate epw-a

# 或使用venv
python -m venv epw-a-env
source epw-a-env/bin/activate  # Linux/macOS
# epw-a-env\Scripts\activate  # Windows
```

### 3. 安装基础依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; import transformers; import numpy; import sklearn; print('All dependencies installed successfully!')"
```

## 可选依赖项

### 量化支持
对于内存受限的环境，建议安装量化支持：
```bash
pip install bitsandbytes
```

### 性能优化
对于更好的性能，可以安装：
```bash
# 更好的注意力机制性能
pip install xformers

# Flash Attention支持（需要兼容的GPU）
pip install flash-attn
```

### 开发工具
```bash
# 代码格式化和检查
pip install black flake8

# 测试框架
pip install pytest
```

## 配置Hugging Face Token

1. 访问 [Hugging Face](https://huggingface.co/settings/tokens)
2. 创建新的访问令牌
3. 在代码中替换 `token = "hf_xxx"` 为您的实际令牌

## 模型下载

首次运行时，框架会自动下载Mixtral-8x7B模型。确保：
- 有稳定的网络连接
- 有足够的存储空间（约50GB）
- 已正确配置Hugging Face令牌

## 故障排除

### 常见问题

#### 1. CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

#### 2. 内存不足
- 启用量化：在代码中设置 `quantization_config`
- 减少批处理大小
- 使用更小的模型进行测试

#### 3. 依赖项冲突
```bash
# 重新安装特定版本
pip uninstall torch transformers
pip install torch==2.0.1 transformers==4.35.0
```

#### 4. scikit-learn不可用
如果scikit-learn安装失败，框架会自动使用简单的启发式分类器。

## 快速开始

安装完成后，运行：
```bash
python epw-enhance-1.py
```

这将演示：
- EPW-A水印生成（GSG和EWP模式）
- 灰盒检测（CSPV）
- 黑盒检测（PEPI）
- 性能对比分析

## 环境变量

可以设置以下环境变量来优化性能：

```bash
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 设置Hugging Face缓存目录
export HF_HOME=/path/to/cache

# 设置模型下载目录
export TRANSFORMERS_CACHE=/path/to/models
```

## 支持

如果遇到问题，请检查：
1. Python版本是否符合要求
2. 所有依赖项是否正确安装
3. GPU驱动和CUDA版本是否兼容
4. 网络连接是否稳定（用于模型下载）

## 许可证

请参考项目根目录的LICENSE文件了解使用条款。 