#!/usr/bin/env python3
"""
测试索引修复的简单脚本
"""

import torch
import random

def test_index_handling():
    """测试索引处理逻辑"""
    print("=== 测试索引处理逻辑 ===")
    
    # 模拟token_ids
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    print(f"token_ids.shape: {token_ids.shape}")
    
    # 模拟all_confidences列表
    all_confidences = [(t, random.random()) for t in range(1, token_ids.shape[1])]
    print(f"all_confidences: {all_confidences[:5]}...")  # 显示前5个
    
    # 测试抽样逻辑
    sample_size = 5
    if len(all_confidences) <= sample_size:
        sampled_indices = all_confidences
    else:
        all_confidences.sort(key=lambda x: x[1])  # 按置信度排序
        k = sample_size // 3
        low_conf = all_confidences[:k]
        high_conf = all_confidences[-k:]
        remaining = all_confidences[k:-k]
        random_conf = random.sample(remaining, sample_size - 2 * k)
        sampled_indices = low_conf + high_conf + random_conf
    
    print(f"sampled_indices: {sampled_indices}")
    
    # 测试正确的循环方式
    try:
        green_token_count = 0
        for t, confidence in sampled_indices:
            context = token_ids[:, :t]
            print(f"索引 {t}, 置信度 {confidence:.3f}, 上下文形状: {context.shape}")
            
            # 模拟检查绿名单
            if random.random() > 0.5:  # 模拟50%概率命中
                green_token_count += 1
        
        print(f"✓ 索引处理成功，绿名单命中数: {green_token_count}")
        return True
        
    except Exception as e:
        print(f"✗ 索引处理失败: {e}")
        return False

def test_tuple_unpacking():
    """测试元组解包逻辑"""
    print("\n=== 测试元组解包逻辑 ===")
    
    # 创建包含元组的列表
    test_data = [(1, 0.8), (2, 0.9), (3, 0.7), (4, 0.6), (5, 0.5)]
    
    try:
        for t, confidence in test_data:
            print(f"索引: {t}, 置信度: {confidence}")
        
        print("✓ 元组解包成功")
        return True
        
    except Exception as e:
        print(f"✗ 元组解包失败: {e}")
        return False

def test_slice_operations():
    """测试切片操作"""
    print("\n=== 测试切片操作 ===")
    
    # 创建测试张量
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    try:
        for t in range(1, token_ids.shape[1]):
            context = token_ids[:, :t]
            print(f"t={t}, context.shape={context.shape}, context={context}")
        
        print("✓ 切片操作成功")
        return True
        
    except Exception as e:
        print(f"✗ 切片操作失败: {e}")
        return False

if __name__ == "__main__":
    success1 = test_index_handling()
    success2 = test_tuple_unpacking()
    success3 = test_slice_operations()
    
    if success1 and success2 and success3:
        print("\n✓ 所有索引修复测试通过！")
    else:
        print("\n✗ 部分测试失败，需要进一步修复。") 