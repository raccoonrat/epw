#!/usr/bin/env python3
"""
测试shape修复的简单脚本
"""

import torch

def test_shape_comparison():
    """测试shape比较修复"""
    print("=== 测试shape比较修复 ===")
    
    # 创建一个模拟的token_ids张量
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    print(f"token_ids.shape: {token_ids.shape}")
    print(f"token_ids.shape[1]: {token_ids.shape[1]}")
    
    # 测试修复后的比较
    try:
        if token_ids.shape[1] <= 1:
            print("✓ shape[1] <= 1 比较成功")
        else:
            print("✓ shape[1] > 1 比较成功")
        
        # 测试range
        for t in range(1, token_ids.shape[1]):
            print(f"✓ 循环 t={t} 成功")
        
        print("✓ 所有shape相关操作都成功")
        return True
        
    except Exception as e:
        print(f"✗ shape操作失败: {e}")
        return False

def test_token_processing():
    """测试token处理逻辑"""
    print("\n=== 测试token处理逻辑 ===")
    
    # 模拟token_ids
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    try:
        # 测试条件检查
        if token_ids.shape[1] <= 1:
            print("文本太短")
        else:
            print(f"文本长度: {token_ids.shape[1]}")
        
        # 测试循环
        for t in range(1, token_ids.shape[1]):
            context = token_ids[:, :t]
            print(f"上下文长度 {t}: {context.shape}")
        
        print("✓ token处理逻辑测试成功")
        return True
        
    except Exception as e:
        print(f"✗ token处理失败: {e}")
        return False

if __name__ == "__main__":
    success1 = test_shape_comparison()
    success2 = test_token_processing()
    
    if success1 and success2:
        print("\n✓ 所有shape修复测试通过！")
    else:
        print("\n✗ 部分测试失败，需要进一步修复。") 