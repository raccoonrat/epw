#!/usr/bin/env python3
"""
简单的DeepSeek MoE模型测试脚本
验证DynamicCache兼容性修复
"""

import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 设置环境变量
os.environ['EPW_FAST_LOADING'] = 'true'
os.environ['EPW_LOAD_IN_4BIT'] = 'true'

def test_simple_generation():
    """简单的文本生成测试"""
    print("=== 简单文本生成测试 ===")
    
    model_name = "deepseek-ai/deepseek-moe-16b-chat"
    print(f"模型: {model_name}")
    
    try:
        # 1. 加载tokenizer
        print("1. 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Tokenizer加载成功")
        
        # 2. 配置量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 3. 加载模型
        print("2. 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✓ 模型加载成功")
        
        # 4. 准备输入
        test_prompt = "Hello"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # 5. 确保输入在正确设备上
        model_device = next(model.parameters()).device
        print(f"模型设备: {model_device}")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # 6. 生成文本
        print("3. 生成文本...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 生成成功: {generated_text}")
        
        print("\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始简单测试...")
    success = test_simple_generation()
    
    if success:
        print("\n✅ 修复成功！现在可以运行主程序了。")
    else:
        print("\n❌ 修复失败，需要进一步调试。") 