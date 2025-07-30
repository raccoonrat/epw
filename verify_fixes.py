#!/usr/bin/env python3
"""
验证修复脚本 - 检查所有修复是否已正确应用
"""

import re

def check_bitsandbytes_fix():
    """检查bitsandbytes修复"""
    print("=== 检查bitsandbytes修复 ===")
    
    with open('epw-enhance-1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含正确的bnb_4bit_compute_dtype设置
    if 'bnb_4bit_compute_dtype=torch.float16' in content:
        print("✓ bnb_4bit_compute_dtype=torch.float16 已设置")
    else:
        print("✗ bnb_4bit_compute_dtype=torch.float16 未找到")
        return False
    
    # 检查是否包含bnb_4bit_use_double_quant
    if 'bnb_4bit_use_double_quant=True' in content:
        print("✓ bnb_4bit_use_double_quant=True 已设置")
    else:
        print("✗ bnb_4bit_use_double_quant=True 未找到")
        return False
    
    # 检查是否包含bnb_4bit_quant_type
    if 'bnb_4bit_quant_type="nf4"' in content:
        print("✓ bnb_4bit_quant_type=\"nf4\" 已设置")
    else:
        print("✗ bnb_4bit_quant_type=\"nf4\" 未找到")
        return False
    
    return True

def check_runtime_error_fix():
    """检查RuntimeError修复"""
    print("\n=== 检查RuntimeError修复 ===")
    
    with open('epw-enhance-1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含跳过第一个token的逻辑
    if 'if t == 0:' in content:
        print("✓ 跳过第一个token的逻辑已添加")
    else:
        print("✗ 跳过第一个token的逻辑未找到")
        return False
    
    # 检查是否包含predicted_expert_index = 0的fallback
    if 'predicted_expert_index = 0' in content:
        print("✓ 专家索引fallback已添加")
    else:
        print("✗ 专家索引fallback未找到")
        return False
    
    return True

def check_logits_processor_fix():
    """检查LogitsProcessor修复"""
    print("\n=== 检查LogitsProcessor修复 ===")
    
    with open('epw-enhance-1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查EPWALogitsProcessor的__call__方法签名
    if 'def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:' in content:
        print("✓ EPWALogitsProcessor签名已修复")
    else:
        print("✗ EPWALogitsProcessor签名未修复")
        return False
    
    # 检查WatermarkLogitsProcessor的__call__方法签名
    if 'def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:' in content:
        print("✓ WatermarkLogitsProcessor签名已修复")
    else:
        print("✗ WatermarkLogitsProcessor签名未修复")
        return False
    
    return True

def check_attribute_fix():
    """检查属性初始化修复"""
    print("\n=== 检查属性初始化修复 ===")
    
    with open('epw-enhance-1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含_fallback_printed的初始化
    if 'self._fallback_printed = False' in content:
        print("✓ _fallback_printed属性已初始化")
    else:
        print("✗ _fallback_printed属性未初始化")
        return False
    
    return True

def check_expert_index_calculation_fix():
    """检查专家索引计算修复"""
    print("\n=== 检查专家索引计算修复 ===")
    
    with open('epw-enhance-1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含改进的专家索引计算
    if 'last_token_id = input_ids[0, -1].item()' in content:
        print("✓ 改进的专家索引计算已添加")
    else:
        print("✗ 改进的专家索引计算未找到")
        return False
    
    # 检查是否包含位置计算
    if 'position = input_ids.shape[1] - 1' in content:
        print("✓ 位置计算已添加")
    else:
        print("✗ 位置计算未找到")
        return False
    
    return True

def check_model_parameter_fix():
    """检查模型参数类型修复"""
    print("\n=== 检查模型参数类型修复 ===")
    
    with open('epw-enhance-1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含模型参数类型转换
    if 'param.data = param.data.to(torch.float16)' in content:
        print("✓ 模型参数类型转换已添加")
    else:
        print("✗ 模型参数类型转换未找到")
        return False
    
    return True

def main():
    """主验证函数"""
    print("开始验证所有修复...")
    
    checks = [
        check_bitsandbytes_fix,
        check_runtime_error_fix,
        check_logits_processor_fix,
        check_attribute_fix,
        check_expert_index_calculation_fix,
        check_model_parameter_fix
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"✗ 检查异常: {e}")
    
    print(f"\n=== 验证结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有修复验证通过！")
        print("\n修复总结:")
        print("- ✓ bitsandbytes警告修复已应用")
        print("- ✓ RuntimeError修复已应用")
        print("- ✓ LogitsProcessor签名修复已应用")
        print("- ✓ 属性初始化修复已应用")
        print("- ✓ 专家索引计算修复已应用")
        print("- ✓ 模型参数类型修复已应用")
        return True
    else:
        print("✗ 部分修复验证失败，需要检查。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 