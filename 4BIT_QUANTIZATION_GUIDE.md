# EPW-A 4位量化使用指南

## 概述

EPW-A框架现在支持显式的4位量化控制，通过环境变量 `EPW_LOAD_IN_4BIT` 来实现。这提供了最快的模型加载速度，但会降低模型精度。

## 量化选项对比

| 选项 | 环境变量 | 加载速度 | 内存使用 | 精度损失 | 适用场景 |
|------|----------|----------|----------|----------|----------|
| 4位量化 | `EPW_LOAD_IN_4BIT=true` | 最快 | 最低 | 较大 | 快速测试、原型开发 |
| 8位量化 | `EPW_LOAD_IN_8BIT=true` | 较快 | 中等 | 较小 | 生产环境 |
| 快速加载 | `EPW_FAST_LOADING=true` | 快 | 低 | 中等 | 标准使用 |
| 标准模式 | 无 | 中等 | 中等 | 中等 | 大多数场景 |

## 使用方法

### 1. 环境变量设置

#### Linux/macOS (Bash)
```bash
# 4位量化
export EPW_LOAD_IN_4BIT=true
python epw-enhance-1.py

# 8位量化
export EPW_LOAD_IN_8BIT=true
python epw-enhance-1.py

# 快速加载模式
export EPW_FAST_LOADING=true
python epw-enhance-1.py

# CPU模式
export EPW_USE_CPU=true
python epw-enhance-1.py

# 组合使用
export EPW_LOAD_IN_4BIT=true
export EPW_USE_CPU=true
python epw-enhance-1.py
```

#### Windows (PowerShell)
```powershell
# 4位量化
$env:EPW_LOAD_IN_4BIT="true"
python epw-enhance-1.py

# 8位量化
$env:EPW_LOAD_IN_8BIT="true"
python epw-enhance-1.py

# 快速加载模式
$env:EPW_FAST_LOADING="true"
python epw-enhance-1.py

# CPU模式
$env:EPW_USE_CPU="true"
python epw-enhance-1.py

# 组合使用
$env:EPW_LOAD_IN_4BIT="true"
$env:EPW_USE_CPU="true"
python epw-enhance-1.py
```

### 2. 便捷脚本

#### Linux/macOS
```bash
# 使用bash脚本
./run_4bit.sh
```

#### Windows
```powershell
# 使用PowerShell脚本
.\run_4bit.ps1
```

### 3. 测试脚本

```bash
# 测试所有配置的性能
python test_loading_speed.py

# 4位量化使用示例
python example_4bit_usage.py
```

## 量化优先级

系统按以下优先级选择量化策略：

1. **最高优先级**：`EPW_LOAD_IN_4BIT=true` - 强制4位量化
2. **高优先级**：`EPW_LOAD_IN_8BIT=true` - 强制8位量化
3. **中优先级**：`EPW_FAST_LOADING=true` - 快速加载模式（默认4位）
4. **低优先级**：标准模式（默认4位，仅限Mixtral模型）

## 性能优化建议

### 硬件配置建议

#### 4位量化适用场景
- **内存**：8GB+ RAM
- **存储**：SSD推荐
- **GPU**：可选，显存2GB+
- **用途**：快速测试、原型开发、演示

#### 8位量化适用场景
- **内存**：16GB+ RAM
- **存储**：SSD推荐
- **GPU**：推荐，显存4GB+
- **用途**：生产环境、精度要求较高的场景

### 组合优化策略

1. **最快速度**：`EPW_LOAD_IN_4BIT=true` + `EPW_USE_CPU=true`
2. **平衡性能**：`EPW_LOAD_IN_8BIT=true`
3. **标准使用**：`EPW_FAST_LOADING=true`
4. **最大兼容性**：无环境变量（标准模式）

## 故障排除

### 常见问题

1. **量化失败**
   ```
   Warning: 4-bit quantization failed: ...
   ```
   - 检查bitsandbytes是否正确安装
   - 尝试使用8位量化或标准模式

2. **内存不足**
   ```
   CUDA out of memory
   ```
   - 使用CPU模式：`EPW_USE_CPU=true`
   - 使用4位量化：`EPW_LOAD_IN_4BIT=true`

3. **加载时间过长**
   - 使用4位量化：`EPW_LOAD_IN_4BIT=true`
   - 使用快速加载：`EPW_FAST_LOADING=true`
   - 使用CPU模式：`EPW_USE_CPU=true`

### 性能监控

脚本会自动显示加载时间统计：
```
=== 加载时间统计 ===
Tokenizer: 2.34秒
Model: 45.67秒
Router Hash: 1.23秒
总加载时间: 49.24秒
==================
```

## 技术细节

### 量化配置

4位量化使用 `BitsAndBytesConfig(load_in_4bit=True)`，这会：
- 将模型权重从16位压缩到4位
- 减少约75%的内存使用
- 可能降低模型精度
- 显著加快加载速度

### 兼容性

- **模型支持**：主要针对Mixtral-8x7B优化
- **硬件要求**：需要bitsandbytes库
- **Python版本**：3.8+
- **操作系统**：Windows、Linux、macOS

## 更新日志

### v1.1.0 (当前版本)
- ✅ 添加 `EPW_LOAD_IN_4BIT` 环境变量
- ✅ 实现量化优先级逻辑
- ✅ 添加便捷脚本 (`run_4bit.sh`, `run_4bit.ps1`)
- ✅ 更新测试脚本支持4位量化
- ✅ 完善文档和使用指南

### 未来计划
- 🔄 支持更多模型类型的量化
- 🔄 添加量化质量评估工具
- 🔄 实现动态量化策略选择 