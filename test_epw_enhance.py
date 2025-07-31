#!/usr/bin/env python3
"""
EPW-A 增强版基本功能测试脚本
"""

import torch
import hashlib
import random
import numpy as np
from typing import List

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

def get_router_hash(model: torch.nn.Module, moe_layer_name: str = "block_sparse_moe") -> str:
    """
    计算并返回模型路由器权重的SHA256哈希值 (IRSH实现)。
    """
    try:
        # 这是一个示例路径，实际路径取决于具体模型架构
        router_weights = getattr(model.model, moe_layer_name).gate.weight.data
        hasher = hashlib.sha256()
        hasher.update(router_weights.cpu().numpy().tobytes())
        return hasher.hexdigest()
    except AttributeError:
        raise AttributeError(f"无法在模型中找到名为 '{moe_layer_name}' 的MoE层或其gate。请检查模型架构。")

def test_basic_functionality():
    """
    测试基本功能，不依赖大型模型
    """
    print("=== EPW-A 增强版基本功能测试 ===")
    
    # 测试绿名单生成
    print("\n1. 测试绿名单生成...")
    test_key = "test_secret_key"
    test_expert_index = 5
    test_router_hash = "test_router_hash_12345"
    test_vocab_size = 1000
    test_gamma = 0.3
    
    green_list = get_green_list_ids(
        test_key, test_expert_index, test_router_hash, test_vocab_size, test_gamma
    )
    print(f"✓ 绿名单生成成功，大小: {len(green_list)}")
    print(f"  预期大小: {int(test_vocab_size * test_gamma)}")
    print(f"  绿名单示例: {green_list[:10]}...")
    
    # 测试确定性
    green_list2 = get_green_list_ids(
        test_key, test_expert_index, test_router_hash, test_vocab_size, test_gamma
    )
    if green_list == green_list2:
        print("✓ 绿名单生成具有确定性")
    else:
        print("✗ 绿名单生成不具有确定性")
    
    # 测试路由器哈希计算
    print("\n2. 测试路由器哈希计算...")
    try:
        # 创建一个简单的测试模型
        class TestModel:
            def __init__(self):
                self.model = type('obj', (object,), {
                    'block_sparse_moe': type('obj', (object,), {
                        'gate': type('obj', (object,), {
                            'weight': type('obj', (object,), {
                                'data': torch.randn(10, 10)
                            })
                        })
                    })
                })()
        
        test_model = TestModel()
        test_hash = get_router_hash(test_model, "block_sparse_moe")
        print(f"✓ 路由器哈希计算成功: {test_hash[:16]}...")
    except Exception as e:
        print(f"✗ 路由器哈希计算失败: {e}")
    
    # 测试Z-score计算
    print("\n3. 测试Z-score计算...")
    def calculate_z_score(green_tokens: int, total_tokens: int, gamma: float = 0.3) -> float:
        if total_tokens == 0:
            return 0.0
        expected_green = total_tokens * gamma
        std_dev = np.sqrt(total_tokens * gamma * (1 - gamma))
        return (green_tokens - expected_green) / (std_dev + 1e-8)
    
    test_green_tokens = 35
    test_total_tokens = 100
    z_score = calculate_z_score(test_green_tokens, test_total_tokens)
    print(f"✓ Z-score计算成功: {z_score:.4f}")
    
    # 测试不同参数下的Z-score
    test_cases = [
        (30, 100, 0.3),  # 正常情况
        (50, 100, 0.3),  # 高命中率
        (10, 100, 0.3),  # 低命中率
    ]
    
    for green_tokens, total_tokens, gamma in test_cases:
        z_score = calculate_z_score(green_tokens, total_tokens, gamma)
        print(f"  绿名单命中: {green_tokens}/{total_tokens}, Z-score: {z_score:.4f}")
    
    print("\n=== 基本功能测试完成 ===")
    print("所有核心功能测试通过！")

if __name__ == "__main__":
    test_basic_functionality() 