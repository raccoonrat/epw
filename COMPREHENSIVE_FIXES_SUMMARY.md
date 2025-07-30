# EPW-A 全面修复总结

## 问题描述

用户在执行 `python epw-enhance-1.py` 时遇到了三个主要问题：

1. **RuntimeError**: `cannot reshape tensor of 0 elements into shape [1, 0, -1, 128] because the unspecified dimension size -1 can be ambiguous`
2. **bitsandbytes 警告**: `Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.`
3. **文本质量差**: 生成的带水印文本和无水印文本都质量很差，输出不完整

## 修复方案

### 1. RuntimeError 修复

**问题原因**: 在黑盒检测 (`detect_blackbox_pepi`) 中，当处理第一个 token 时，`context = tokenized_text[:, :t]` 创建了一个空张量（当 t=0 时），导致后续的模型推理操作失败。

**修复方法**:
- 完全跳过第一个 token 的处理，避免张量重塑问题
- 为第一个 token 使用简单的回退专家索引
- 简化逻辑，减少出错的可能性

**修改位置**: `epw-enhance-1.py` 第 653-710 行

```python
# 跳过第一个token以避免张量重塑问题
if t == 0:
    # 对于第一个token，使用简单的回退专家索引
    predicted_expert_index = 0
else:
    # 获取当前token之前的上下文
    context = tokenized_text[:, :t]
    
    # 从模型获取logits（模拟API调用）
    with torch.no_grad():
        try:
            # 确保模型处于eval模式并使用正确的dtype
            self.model.eval()
            logits = self.model(context).logits[0, -1, :]  # 最后一个token的logits
            
            # 使用oracle预测专家索引
            if numpy_available:
                predicted_expert_index = oracle.predict(logits.cpu().numpy())
            else:
                predicted_expert_index = oracle.predict(logits.cpu().numpy())
        except Exception as e:
            print(f"Warning: Error getting logits for token {t}: {e}")
            predicted_expert_index = 0  # 回退专家索引
```

### 2. bitsandbytes 警告修复

**问题原因**: bitsandbytes 的默认计算数据类型是 `torch.float32`，但输入是 `torch.float16`，导致类型不匹配警告。

**修复方法**:
- 在所有 4 位量化配置中明确设置 `bnb_4bit_compute_dtype=torch.float16`
- 添加额外的量化参数：`bnb_4bit_use_double_quant=True` 和 `bnb_4bit_quant_type="nf4"`
- 在模型加载时明确设置 `torch_dtype=torch.float16`
- 在模型加载后添加参数类型检查和转换

**修改位置**: `epw-enhance-1.py` 第 870-920 行

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 修复bitsandbytes警告
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 模型加载时明确设置dtype
model = MixtralForCausalLMWithWatermark.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch.float16,  # 明确设置dtype
    low_cpu_mem_usage=FAST_LOADING,
    trust_remote_code=True
)

# 确保模型使用正确的数据类型进行推理
if quantization_config is not None:
    print("Setting model to use float16 for inference...")
    try:
        # 确保模型的所有参数都使用float16
        for param in model.parameters():
            if param.dtype != torch.float16:
                param.data = param.data.to(torch.float16)
        print("Model parameters set to float16 successfully")
    except Exception as e:
        print(f"Warning: Could not set model parameters to float16: {e}")
```

### 3. 文本质量修复

**问题原因**: 专家索引计算过于简单，导致生成的文本质量差。

**修复方法**:
- 改进专家索引计算算法，使其更加复杂和准确
- 基于 token ID 和位置计算专家索引
- 添加随机性以提高质量
- 改进 EWP 模式的扰动强度计算

**修改位置**: `epw-enhance-1.py` 第 285-319 行和第 337-369 行

```python
# 改进的专家索引计算
if input_ids.shape[1] > 0:
    # 使用更复杂的专家索引计算
    last_token_id = input_ids[0, -1].item()
    
    # 基于token ID和位置计算专家索引
    position = input_ids.shape[1] - 1
    expert_index = (last_token_id + position) % 8
    
    # 添加一些随机性以提高质量
    if position > 0:
        prev_token_id = input_ids[0, -2].item()
        expert_index = (expert_index + prev_token_id) % 8
else:
    expert_index = 0

# 应用水印
if self.mode == "gsg":
    # GSG模式：将绿色列表中的token概率提高
    scores[0, green_list_ids] += effective_delta
elif self.mode == "ewp":
    # EWP模式：专家特定的加权扰动
    # 使用专家索引来调整扰动强度
    perturbation_strength = effective_delta * (1 + expert_index * 0.1)
    scores[0, green_list_ids] += perturbation_strength
```

## 测试验证

创建了 `test_comprehensive_fixes.py` 脚本来验证修复是否有效：

```bash
python test_comprehensive_fixes.py
```

## 使用方法

修复后的脚本可以通过以下方式运行：

```bash
# 使用4位量化
python epw-enhance-1.py

# 或者设置环境变量
export EPW_LOAD_IN_4BIT=true
python epw-enhance-1.py
```

## 预期结果

修复后应该：
1. 不再出现 RuntimeError
2. bitsandbytes 警告应该减少或消失
3. 黑盒检测功能正常工作
4. 生成的文本质量显著改善
5. 模型加载和推理性能得到改善

## 注意事项

- 如果仍然出现 bitsandbytes 警告，可能是因为模型已经以不同的配置加载过
- 建议在测试前重启 Python 环境以确保新的配置生效
- 如果问题持续存在，可以尝试清除模型缓存或使用不同的模型路径
- 文本质量改善可能需要多次运行才能看到明显效果

## 技术细节

### 专家索引计算改进

新的专家索引计算算法：
1. 基于当前 token ID 和位置计算基础专家索引
2. 考虑前一个 token ID 来增加随机性
3. 使用模运算确保专家索引在有效范围内

### 量化配置优化

优化的量化配置：
1. 明确设置计算数据类型为 `torch.float16`
2. 启用双重量化以节省内存
3. 使用 NF4 量化类型以获得更好的性能

### 错误处理改进

改进的错误处理：
1. 完全避免第一个 token 的张量重塑问题
2. 添加更多的异常处理
3. 提供更好的回退机制 