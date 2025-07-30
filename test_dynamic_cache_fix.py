#!/usr/bin/env python3
"""
测试DynamicCache修复的脚本
"""

import os
import sys

# 设置环境变量
os.environ['EPW_FAST_LOADING'] = 'true'
os.environ['EPW_LOAD_IN_4BIT'] = 'true'

def test_dynamic_cache_fix():
    """测试DynamicCache修复"""
    print("=== 测试DynamicCache修复 ===")
    
    try:
        # 1. 测试DynamicCache修复
        print("1. 测试DynamicCache修复...")
        
        # 导入修复函数
        from epw_enhance_1 import fix_dynamic_cache_compatibility
        
        # 应用修复
        fix_dynamic_cache_compatibility()
        
        # 验证修复
        try:
            from transformers.cache import DynamicCache
            cache = DynamicCache()
            
            # 测试get_usable_length方法是否存在
            if hasattr(cache, 'get_usable_length'):
                result = cache.get_usable_length()
                print(f"✓ get_usable_length方法可用，返回: {result}")
            else:
                print("❌ get_usable_length方法不可用")
                return False
                
        except Exception as e:
            print(f"❌ DynamicCache测试失败: {e}")
            return False
        
        # 2. 测试简单的模型加载
        print("\n2. 测试简单模型加载...")
        
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        
        model_name = "deepseek-ai/deepseek-moe-16b-chat"
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Tokenizer加载成功")
        
        # 配置量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 加载模型
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✓ 模型加载成功")
        
        # 3. 测试简单的前向传播
        print("\n3. 测试前向传播...")
        
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # 确保输入在正确设备上
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)
        
        print("✓ 前向传播成功")
        
        # 4. 测试文本生成
        print("\n4. 测试文本生成...")
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✓ 文本生成成功: {generated_text}")
        
        print("\n🎉 所有测试通过！DynamicCache修复成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试DynamicCache修复...")
    success = test_dynamic_cache_fix()
    
    if success:
        print("\n✅ 修复成功！现在可以运行主程序了。")
        print("运行命令: python epw-enhance-1.py")
    else:
        print("\n❌ 修复失败，需要进一步调试。") 