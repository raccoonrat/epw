# EPW-A 设备修复说明

## 问题描述

在运行修复后的 `epw-enhance-2.py` 时遇到以下错误：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

## 问题原因

模型在GPU上运行，但输入的token_ids张量在CPU上，导致设备不匹配错误。

## 修复方案

### 修复前的问题代码

```python
def get_logits_and_route_info(self, input_ids: torch.Tensor) -> Tuple:
    with torch.no_grad():
        outputs = self.model(input_ids, output_router_logits=True)  # 设备不匹配
    # ...
```

### 修复后的正确代码

```python
def get_logits_and_route_info(self, input_ids: torch.Tensor) -> Tuple:
    # 确保输入张量在正确的设备上
    if hasattr(self.model, 'device'):
        device = self.model.device
    else:
        device = next(self.model.parameters()).device
    
    input_ids = input_ids.to(device)  # 移动到正确设备
    
    with torch.no_grad():
        outputs = self.model(input_ids, output_router_logits=True)
    # ...
```

## 具体修复位置

### 1. `MoEModelWrapper.get_logits_and_route_info` 方法

**修复前：**
```python
def get_logits_and_route_info(self, input_ids: torch.Tensor) -> Tuple:
    with torch.no_grad():
        outputs = self.model(input_ids, output_router_logits=True)
```

**修复后：**
```python
def get_logits_and_route_info(self, input_ids: torch.Tensor) -> Tuple:
    # 确保输入张量在正确的设备上
    if hasattr(self.model, 'device'):
        device = self.model.device
    else:
        device = next(self.model.parameters()).device
    
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = self.model(input_ids, output_router_logits=True)
```

### 2. `MoEModelWrapper.get_logits_blackbox` 方法

**修复前：**
```python
def get_logits_blackbox(self, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = self.model(input_ids)
    return outputs.logits[:, -1, :]
```

**修复后：**
```python
def get_logits_blackbox(self, input_ids: torch.Tensor) -> torch.Tensor:
    # 确保输入张量在正确的设备上
    if hasattr(self.model, 'device'):
        device = self.model.device
    else:
        device = next(self.model.parameters()).device
    
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = self.model(input_ids)
    return outputs.logits[:, -1, :]
```

## 修复原理

### PyTorch 设备管理

1. **模型设备检测**：
   - 优先检查 `model.device` 属性
   - 如果没有，则从模型参数中获取设备

2. **张量设备移动**：
   - 使用 `.to(device)` 方法将张量移动到正确设备
   - 确保所有张量在同一设备上

### 设备检测策略

```python
# 方法1: 检查模型是否有device属性
if hasattr(self.model, 'device'):
    device = self.model.device

# 方法2: 从模型参数获取设备
else:
    device = next(self.model.parameters()).device

# 移动张量到正确设备
input_ids = input_ids.to(device)
```

## 测试验证

创建了 `test_device_fix.py` 脚本来验证修复：

```bash
python test_device_fix.py
```

预期输出：
```
=== 测试设备处理逻辑 ===
原始输入设备: cpu
模型设备: cuda:0
输入设备: cpu
移动后输入设备: cuda:0
✓ 灰盒方法设备处理成功
✓ 黑盒方法设备处理成功

=== 测试设备检测逻辑 ===

测试: 有device属性的模型
使用model.device: cuda:0
✓ 设备检测成功: cuda:0

测试: 有参数的模型
使用参数设备: cuda:0
✓ 设备检测成功: cuda:0

✓ 所有设备修复测试通过！
```

## 注意事项

1. **设备兼容性**：代码同时支持CPU和GPU
2. **设备检测**：自动检测模型所在设备
3. **内存管理**：张量移动是安全的，不会影响原始数据
4. **性能考虑**：避免频繁的设备间移动

## 预期结果

修复后，`epw-enhance-2.py` 应该能够：
1. ✅ 正确加载模型到GPU
2. ✅ 自动检测模型设备
3. ✅ 将输入张量移动到正确设备
4. ✅ 成功进行水印生成和检测

不再出现设备不匹配的 `RuntimeError` 错误。 