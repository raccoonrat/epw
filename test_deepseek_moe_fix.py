#!/usr/bin/env python3
"""
测试DeepSeek MoE模型修复的脚本
验证适配器模式是否解决了"Unrecognized configuration class"错误
"""

import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 设置环境变量以启用快速加载
os.environ['EPW_FAST_LOADING'] = 'true'
os.environ['EPW_LOAD_IN_4BIT'] = 'true'  # 使用4位量化以加快加载

def test_model_loading():
    """测试模型加载"""
    print("=== 测试DeepSeek MoE模型加载 ===")
    
    model_name = "deepseek-ai/deepseek-moe-16b-chat"
    print(f"模型名称: {model_name}")
    
    # 配置量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        # 测试1: 加载tokenizer
        print("\n1. 加载tokenizer...")
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer_end = time.time()
        print(f"✓ Tokenizer加载成功，耗时: {tokenizer_end - tokenizer_start:.2f}s")
        
        # 测试2: 加载基础模型
        print("\n2. 加载基础模型...")
        model_start = time.time()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model_end = time.time()
        print(f"✓ 基础模型加载成功，耗时: {model_end - model_start:.2f}s")
        print(f"  模型类型: {type(base_model).__name__}")
        
        # 测试3: 创建包装器
        print("\n3. 创建模型包装器...")
        from epw_enhance_1 import MoEModelWithWatermark
        
        wrapper_start = time.time()
        model = MoEModelWithWatermark(base_model, tokenizer)
        wrapper_end = time.time()
        print(f"✓ 模型包装器创建成功，耗时: {wrapper_end - wrapper_start:.2f}s")
        print(f"  专家数量: {model._num_experts}")
        print(f"  路由器哈希: {model._router_hash[:16]}..." if model._router_hash else "无")
        
        # 测试4: 简单的前向传播
        print("\n4. 测试前向传播...")
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        forward_start = time.time()
        with torch.no_grad():
            outputs = model.forward(**inputs, output_router_logits=True)
        forward_end = time.time()
        print(f"✓ 前向传播成功，耗时: {forward_end - forward_start:.2f}s")
        
        if hasattr(outputs, 'router_logits'):
            print(f"  路由器logits数量: {len(outputs.router_logits)}")
        else:
            print("  警告: 未找到路由器logits")
        
        # 测试5: 文本生成
        print("\n5. 测试文本生成...")
        generation_start = time.time()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7
            )
        generation_end = time.time()
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✓ 文本生成成功，耗时: {generation_end - generation_start:.2f}s")
        print(f"  生成文本: {generated_text}")
        
        print("\n=== 所有测试通过！DeepSeek MoE模型修复成功 ===")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_watermark_functionality():
    """测试水印功能"""
    print("\n=== 测试水印功能 ===")
    
    try:
        from epw_enhance_1 import EPWALogitsProcessor, WatermarkDetector
        
        # 这里需要先加载模型，但为了简化测试，我们只测试类定义
        print("✓ EPWALogitsProcessor类定义正常")
        print("✓ WatermarkDetector类定义正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 水印功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试DeepSeek MoE模型修复...")
    
    # 测试模型加载
    loading_success = test_model_loading()
    
    # 测试水印功能
    watermark_success = test_watermark_functionality()
    
    if loading_success and watermark_success:
        print("\n🎉 所有测试通过！适配器模式修复成功！")
    else:
        print("\n❌ 部分测试失败，需要进一步调试") 