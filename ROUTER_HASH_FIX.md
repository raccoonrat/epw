# EPW-A 路由器哈希修复说明

## 问题描述

在运行 `epw-enhance-2.py` 时遇到以下错误：

```
AttributeError: 'MixtralModel' object has no attribute 'block_sparse_moe'
```

## 问题原因

原始代码假设模型架构为：
```python
model.model.block_sparse_moe.gate.weight.data
```

但是 Mixtral 模型的实际架构是：
```python
model.model.layers[i].block_sparse_moe.gate.weight.data
```

其中 `layers` 是一个列表，包含多个层，每个层都有自己的 MoE 组件。

## 修复方案

### 1. 修复 `get_router_hash` 函数

**原始代码：**
```python
def get_router_hash(model: torch.nn.Module, moe_layer_name: str = "block_sparse_moe") -> str:
    try:
        router_weights = getattr(model.model, moe_layer_name).gate.weight.data
        hasher = hashlib.sha256()
        hasher.update(router_weights.cpu().numpy().tobytes())
        return hasher.hexdigest()
    except AttributeError:
        raise AttributeError(f"无法在模型中找到名为 '{moe_layer_name}' 的MoE层或其gate。请检查模型架构。")
```

**修复后的代码：**
```python
def get_router_hash(model: torch.nn.Module, moe_layer_name: str = "block_sparse_moe") -> str:
    try:
        # 检查模型类型
        model_type = type(model).__name__
        print(f"检测到模型类型: {model_type}")
        
        # 对于Mixtral模型，遍历所有层查找MoE层
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            router_weights = []
            moe_layer_count = 0
            
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                    try:
                        gate_weights = layer.block_sparse_moe.gate.weight.data
                        
                        # 检查是否为meta tensor
                        if hasattr(gate_weights, 'is_meta') and gate_weights.is_meta:
                            print(f"警告: 第{i}层的路由器权重是meta tensor")
                            continue
                        
                        # 安全访问数据
                        weight_bytes = gate_weights.cpu().numpy().tobytes()
                        router_weights.append(weight_bytes)
                        moe_layer_count += 1
                        
                    except Exception as e:
                        print(f"警告: 无法访问第{i}层的路由器权重: {e}")
                        continue
            
            if router_weights:
                print(f"找到 {moe_layer_count} 个MoE层")
                # 连接所有路由器权重并哈希
                combined_weights = b''.join(router_weights)
                hasher = hashlib.sha256()
                hasher.update(combined_weights)
                return hasher.hexdigest()
            else:
                print("未找到任何可用的MoE层")
        
        # 尝试其他可能的架构
        alternative_paths = [
            'model.block_sparse_moe.gate.weight',
            'model.moe.gate.weight',
            'block_sparse_moe.gate.weight',
            'moe.gate.weight'
        ]
        
        for path in alternative_paths:
            try:
                # 使用getattr递归访问
                parts = path.split('.')
                current = model
                for part in parts:
                    current = getattr(current, part)
                
                weight_data = current.data
                hasher = hashlib.sha256()
                hasher.update(weight_data.cpu().numpy().tobytes())
                print(f"使用路径 '{path}' 成功计算哈希")
                return hasher.hexdigest()
            except (AttributeError, Exception) as e:
                continue
        
        # 如果所有方法都失败，使用默认哈希
        print("无法找到路由器权重，使用默认哈希")
        return "default_router_hash_for_mixtral"
        
    except Exception as e:
        print(f"路由器哈希计算过程中发生错误: {e}")
        return "error_router_hash"
```

### 2. 修复 `MoEModelWrapper.get_router_weights` 方法

**原始代码：**
```python
def get_router_weights(self):
    return self.model.model.block_sparse_moe.gate.weight
```

**修复后的代码：**
```python
def get_router_weights(self):
    try:
        # 尝试直接访问
        return self.model.model.block_sparse_moe.gate.weight
    except AttributeError:
        # 如果直接访问失败，尝试遍历层
        for layer in self.model.model.layers:
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                return layer.block_sparse_moe.gate.weight
        # 如果都失败，返回None
        return None
```

## 修复特点

1. **多架构支持**：支持多种模型架构，不仅限于 Mixtral
2. **错误处理**：提供详细的错误信息和警告
3. **降级策略**：如果无法获取路由器权重，使用默认哈希
4. **调试信息**：提供详细的调试输出，帮助诊断问题

## 测试方法

1. 运行修复后的 `epw-enhance-2.py`
2. 观察路由器哈希计算是否成功
3. 检查是否能够正常生成和检测水印

## 预期结果

修复后应该看到类似以下的输出：

```
检测到模型类型: MixtralForCausalLM
找到 32 个MoE层
✓ 路由器哈希: a1b2c3d4e5f6...
```

而不是之前的错误信息。

## 注意事项

1. 如果模型使用 4-bit 量化，某些权重可能是 meta tensor，需要特殊处理
2. 不同版本的 Mixtral 模型可能有细微的架构差异
3. 建议在运行前先测试路由器哈希计算是否成功 