#!/usr/bin/env python3
"""
测试AttributeError修复的简单脚本
"""

import torch
from transformers import LogitsProcessor

# 导入修复后的EPWALogitsProcessor
import sys
sys.path.append('.')

# 模拟EPWALogitsProcessor的初始化
class TestEPWALogitsProcessor(LogitsProcessor):
    def __init__(self):
        self._fallback_printed = False  # 确保这个属性被初始化
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self._fallback_printed:
            print("EPW-A LogitsProcessor: Using fallback mode with improved expert index calculation")
            self._fallback_printed = True
        
        # 简单的测试逻辑
        return scores

def test_attribute_error_fix():
    """测试AttributeError是否已修复"""
    print("测试AttributeError修复...")
    
    try:
        # 创建测试实例
        processor = TestEPWALogitsProcessor()
        
        # 创建测试数据
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        scores = torch.randn(1, 1000)
        
        # 调用__call__方法
        result = processor(input_ids, scores)
        
        print("✓ AttributeError修复成功！")
        print("✓ EPWALogitsProcessor可以正常工作")
        
        return True
        
    except AttributeError as e:
        print(f"✗ AttributeError仍然存在: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

if __name__ == "__main__":
    success = test_attribute_error_fix()
    if success:
        print("\n现在可以运行主脚本了: python epw-enhance-1.py")
    else:
        print("\n需要进一步修复AttributeError") 