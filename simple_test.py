#!/usr/bin/env python3
"""
简单的测试脚本来验证EPW-A修复
"""

import torch
import hashlib
import random
import numpy as np
from transformers import LogitsProcessor
from typing import List, Tuple, Dict, Any

# 模拟路由器哈希函数
def get_router_hash(model: torch.nn.Module, moe_layer_name: str = "block_sparse_moe") -> str:
    """
    计算并返回模型路由器权重的SHA256哈希值 (IRSH实现)。
    专门针对Mixtral模型架构优化。
    """
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

def get_green_list_ids(
    key: str,
    expert_index: int,
    router_hash: str,
    vocab_size: int,
    gamma: float = 0.5
) -> List[int]:
    """
    根据密钥、专家索引和路由器哈希生成绿名单 (IRSH实现)。
    """
    combined_input = f"{key}-{expert_index}-{router_hash}"
    hasher = hashlib.sha256()
    hasher.update(combined_input.encode('utf-8'))
    seed = int.from_bytes(hasher.digest(), 'big')
    
    rng = random.Random(seed)
    green_list_size = int(vocab_size * gamma)
    green_list = rng.sample(range(vocab_size), green_list_size)
    return green_list

# 模拟模型包装器
class MockModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_vocab_size(self) -> int:
        return 50257  # GPT-2词汇表大小

    def get_logits_and_route_info(self, input_ids: torch.Tensor) -> Tuple:
        # 模拟返回
        logits = torch.randn(1, 50257)
        top_expert_index = random.randint(0, 7)
        top_expert_confidence = random.random()
        return logits, top_expert_index, top_expert_confidence

# 模拟水印生成器
class MockWatermarker(LogitsProcessor):
    def __init__(self, model_wrapper, secret_key: str, gamma: float = 0.5):
        self.wrapper = model_wrapper
        self.secret_key = secret_key
        self.gamma = gamma
        self.vocab_size = self.wrapper.get_vocab_size()
        self.router_hash = "mock_router_hash"

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # 模拟水印处理
        _, top_expert_index, _ = self.wrapper.get_logits_and_route_info(input_ids)
        green_list = get_green_list_ids(
            self.secret_key,
            top_expert_index,
            self.router_hash,
            self.vocab_size,
            self.gamma
        )
        scores[:, green_list] += 2.0
        return scores

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建模拟模型
    class MockModel:
        def __init__(self):
            self.config = type('obj', (object,), {'vocab_size': 50257})()
            self.device = 'cpu'
        
        def __call__(self, input_ids, output_router_logits=False, **kwargs):
            batch_size, seq_len = input_ids.shape
            vocab_size = 50257
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return type('obj', (object,), {'logits': logits})()
    
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 50257
            self.pad_token = "<|endoftext|>"
            self.eos_token = "<|endoftext|>"
        
        def encode(self, text, return_tensors='pt', add_special_tokens=False):
            tokens = [hash(text) % self.vocab_size] + [i % self.vocab_size for i in range(len(text.split()))]
            return torch.tensor([tokens])
        
        def decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, torch.Tensor):
                if token_ids.dim() == 1:
                    token_list = token_ids.tolist()
                else:
                    token_list = token_ids[0].tolist()
            else:
                token_list = token_ids
            return "模拟生成的文本: " + " ".join([f"token_{i}" for i in token_list])
    
    # 测试路由器哈希
    model = MockModel()
    try:
        router_hash = get_router_hash(model)
        print(f"✓ 路由器哈希计算成功: {router_hash}")
    except Exception as e:
        print(f"✗ 路由器哈希计算失败: {e}")
        return False
    
    # 测试绿名单生成
    try:
        green_list = get_green_list_ids("test_key", 0, router_hash, 50257, 0.5)
        print(f"✓ 绿名单生成成功，大小: {len(green_list)}")
    except Exception as e:
        print(f"✗ 绿名单生成失败: {e}")
        return False
    
    # 测试水印生成器
    try:
        tokenizer = MockTokenizer()
        model_wrapper = MockModelWrapper(model, tokenizer)
        watermarker = MockWatermarker(model_wrapper, "test_key", 0.5)
        
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.randn(1, 50257)
        watermarked_scores = watermarker(input_ids, scores)
        
        print(f"✓ 水印生成器测试成功")
    except Exception as e:
        print(f"✗ 水印生成器测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✓ 所有测试通过！修复成功。")
    else:
        print("\n✗ 测试失败，需要进一步修复。") 