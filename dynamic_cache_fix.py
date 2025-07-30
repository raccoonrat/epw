#!/usr/bin/env python3
"""
DynamicCache修复脚本
在运行主程序之前先运行此脚本
"""

import sys
import os

def apply_dynamic_cache_fix():
    """应用DynamicCache修复"""
    print("=== 应用DynamicCache修复 ===")
    
    try:
        # 导入transformers
        import transformers
        
        # 检查是否有cache模块
        if hasattr(transformers, 'cache'):
            from transformers.cache import DynamicCache
            
            # 检查是否需要添加get_usable_length方法
            if not hasattr(DynamicCache, 'get_usable_length'):
                def get_usable_length(self):
                    """兼容性方法，返回序列长度"""
                    return self.get_seq_length()
                
                # 动态添加方法
                DynamicCache.get_usable_length = get_usable_length
                print("✓ 已为DynamicCache添加get_usable_length方法")
            else:
                print("✓ DynamicCache已包含get_usable_length方法")
            
            # 测试修复是否有效
            cache = DynamicCache()
            try:
                result = cache.get_usable_length()
                print(f"✓ 测试成功，get_usable_length返回: {result}")
                return True
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                return False
                
        else:
            print("⚠️ transformers没有cache模块")
            return False
            
    except ImportError as e:
        print(f"❌ 无法导入transformers: {e}")
        return False
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

def test_model_generation():
    """测试模型生成功能"""
    print("\n=== 测试模型生成功能 ===")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # 设置环境变量
        os.environ['EPW_FAST_LOADING'] = 'true'
        os.environ['EPW_LOAD_IN_4BIT'] = 'true'
        
        model_name = "deepseek-ai/deepseek-moe-16b-chat"
        print(f"测试模型: {model_name}")
        
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✓ 模型加载成功")
        
        # 测试生成
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # 确保输入在正确设备上
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        print("测试文本生成...")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✓ 生成成功: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始DynamicCache修复...")
    
    # 应用修复
    fix_success = apply_dynamic_cache_fix()
    
    if fix_success:
        print("\n✅ DynamicCache修复成功！")
        
        # 测试模型生成
        model_success = test_model_generation()
        
        if model_success:
            print("\n🎉 所有测试通过！现在可以运行主程序了。")
            print("运行命令: python epw-enhance-1.py")
        else:
            print("\n⚠️ 模型测试失败，但DynamicCache修复成功。")
    else:
        print("\n❌ DynamicCache修复失败。") 