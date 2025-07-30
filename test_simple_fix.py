#!/usr/bin/env python3
"""
简单的DynamicCache修复测试
"""

import os
import sys

def test_dynamic_cache_import():
    """测试DynamicCache导入"""
    print("=== 测试DynamicCache导入 ===")
    
    try:
        import transformers
        print(f"transformers版本: {transformers.__version__}")
        
        # 尝试不同的导入方法
        DynamicCache = None
        
        # 方法1: 直接从transformers导入
        try:
            from transformers import DynamicCache
            print("✓ 方法1成功: from transformers import DynamicCache")
        except ImportError as e:
            print(f"✗ 方法1失败: {e}")
        
        # 方法2: 从transformers.cache导入
        if DynamicCache is None:
            try:
                from transformers.cache import DynamicCache
                print("✓ 方法2成功: from transformers.cache import DynamicCache")
            except ImportError as e:
                print(f"✗ 方法2失败: {e}")
        
        # 方法3: 从transformers.utils导入
        if DynamicCache is None:
            try:
                from transformers.utils import DynamicCache
                print("✓ 方法3成功: from transformers.utils import DynamicCache")
            except ImportError as e:
                print(f"✗ 方法3失败: {e}")
        
        # 方法4: 从transformers.generation导入
        if DynamicCache is None:
            try:
                from transformers.generation import DynamicCache
                print("✓ 方法4成功: from transformers.generation import DynamicCache")
            except ImportError as e:
                print(f"✗ 方法4失败: {e}")
        
        # 方法5: 动态查找
        if DynamicCache is None:
            print("尝试动态查找DynamicCache...")
            for attr_name in dir(transformers):
                attr = getattr(transformers, attr_name)
                if hasattr(attr, '__name__') and attr.__name__ == 'DynamicCache':
                    DynamicCache = attr
                    print(f"✓ 动态找到DynamicCache: {attr_name}")
                    break
        
        if DynamicCache is not None:
            print(f"✓ DynamicCache类找到: {DynamicCache}")
            
            # 检查方法
            if hasattr(DynamicCache, 'get_usable_length'):
                print("✓ DynamicCache已有get_usable_length方法")
            else:
                print("⚠️ DynamicCache缺少get_usable_length方法")
                
            if hasattr(DynamicCache, 'get_seq_length'):
                print("✓ DynamicCache已有get_seq_length方法")
            else:
                print("⚠️ DynamicCache缺少get_seq_length方法")
                
            return True
        else:
            print("❌ 无法找到DynamicCache类")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_model():
    """测试简单模型加载"""
    print("\n=== 测试简单模型加载 ===")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
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
        
        # 测试简单的前向传播
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # 确保输入在正确设备上
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)
        
        print("✓ 前向传播成功")
        
        # 测试生成
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=3,
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
    print("开始简单测试...")
    
    # 测试DynamicCache导入
    import_success = test_dynamic_cache_import()
    
    if import_success:
        print("\n✅ DynamicCache导入测试成功！")
        
        # 测试模型加载
        model_success = test_simple_model()
        
        if model_success:
            print("\n🎉 所有测试通过！")
            print("现在可以运行主程序: python epw-enhance-1.py")
        else:
            print("\n⚠️ 模型测试失败，但DynamicCache导入成功。")
    else:
        print("\n❌ DynamicCache导入测试失败。") 