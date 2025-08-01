# EPW-A Shape 修复说明

## 问题描述

在运行修复后的 `epw-enhance-2.py` 时遇到以下错误：

```
TypeError: '<=' not supported between instances of 'torch.Size' and 'int'
```

## 问题原因

原始代码试图直接比较 `torch.Size` 对象和整数：

```python
if token_ids.shape <= 1: return 0.0
```

但是 `token_ids.shape` 是一个 `torch.Size` 对象，不能直接与整数进行比较。

## 修复方案

### 修复前的问题代码

```python
# 错误的比较方式
if token_ids.shape <= 1: return 0.0
for t in range(1, token_ids.shape):
    # ...

# 错误的计算方式
num_tokens = token_ids.shape - 1
```

### 修复后的正确代码

```python
# 正确的比较方式 - 使用shape[1]获取序列长度
if token_ids.shape[1] <= 1: return 0.0
for t in range(1, token_ids.shape[1]):
    # ...

# 正确的计算方式
num_tokens = token_ids.shape[1] - 1
```

## 具体修复位置

### 1. `detect_graybox_cspv` 方法 (第252-256行)

**修复前：**
```python
token_ids = self.wrapper.tokenizer.encode(text, return_tensors='pt')
if token_ids.shape <= 1: return 0.0

# 1. 单次前向传播获取所有路由置信度
all_confidences = []
for t in range(1, token_ids.shape):
```

**修复后：**
```python
token_ids = self.wrapper.tokenizer.encode(text, return_tensors='pt')
if token_ids.shape[1] <= 1: return 0.0

# 1. 单次前向传播获取所有路由置信度
all_confidences = []
for t in range(1, token_ids.shape[1]):
```

### 2. `detect_blackbox_pepi` 方法 (第294-299行)

**修复前：**
```python
token_ids = self.wrapper.tokenizer.encode(text, return_tensors='pt')
if token_ids.shape <= 1: return 0.0

green_token_count = 0
num_tokens = token_ids.shape - 1

for t in range(1, token_ids.shape):
```

**修复后：**
```python
token_ids = self.wrapper.tokenizer.encode(text, return_tensors='pt')
if token_ids.shape[1] <= 1: return 0.0

green_token_count = 0
num_tokens = token_ids.shape[1] - 1

for t in range(1, token_ids.shape[1]):
```

### 3. `train_pepi_oracle` 函数 (第329行)

**修复前：**
```python
for text in training_corpus:
    token_ids = model_wrapper.tokenizer.encode(text, return_tensors='pt')
    for t in range(1, token_ids.shape):
```

**修复后：**
```python
for text in training_corpus:
    token_ids = model_wrapper.tokenizer.encode(text, return_tensors='pt')
    for t in range(1, token_ids.shape[1]):
```

## 修复原理

### PyTorch 张量的 shape 属性

- `token_ids.shape` 返回一个 `torch.Size` 对象，例如 `torch.Size([1, 10])`
- `token_ids.shape[0]` 是批次大小 (batch size)
- `token_ids.shape[1]` 是序列长度 (sequence length)

### 正确的访问方式

```python
# 对于形状为 [batch_size, sequence_length] 的张量
token_ids = torch.tensor([[1, 2, 3, 4, 5]])  # shape: [1, 5]

print(token_ids.shape)      # torch.Size([1, 5])
print(token_ids.shape[0])   # 1 (batch_size)
print(token_ids.shape[1])   # 5 (sequence_length)

# 正确的比较
if token_ids.shape[1] <= 1:  # 检查序列长度是否太短
    return 0.0

# 正确的循环
for t in range(1, token_ids.shape[1]):  # 从1到序列长度
    context = token_ids[:, :t]  # 取前t个token作为上下文
```

## 测试验证

创建了 `test_shape_fix.py` 脚本来验证修复：

```bash
python test_shape_fix.py
```

预期输出：
```
=== 测试shape比较修复 ===
token_ids.shape: torch.Size([1, 5])
token_ids.shape[1]: 5
✓ shape[1] > 1 比较成功
✓ 循环 t=1 成功
✓ 循环 t=2 成功
✓ 循环 t=3 成功
✓ 循环 t=4 成功
✓ 所有shape相关操作都成功

=== 测试token处理逻辑 ===
文本长度: 5
上下文长度 1: torch.Size([1, 1])
上下文长度 2: torch.Size([1, 2])
上下文长度 3: torch.Size([1, 3])
上下文长度 4: torch.Size([1, 4])
✓ token处理逻辑测试成功

✓ 所有shape修复测试通过！
```

## 注意事项

1. **张量维度**：确保理解张量的维度含义
2. **批次处理**：代码支持批次处理，但通常 batch_size = 1
3. **序列长度**：使用 `shape[1]` 获取实际的序列长度
4. **边界检查**：在访问张量之前进行适当的边界检查

## 预期结果

修复后，`epw-enhance-2.py` 应该能够正常运行，不再出现 `TypeError` 错误，并且能够成功进行水印检测。 