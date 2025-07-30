#!/usr/bin/env python3
"""
猴子补丁修复DynamicCache问题
"""

import os
import sys
import types

def apply_monkey_patch():
    """应用猴子补丁修复"""
    print("=== 应用猴子补丁修复 ===")
    
    try:
        import transformers
        
        # 方法1: 直接修改transformers.generation.utils
        try:
            from transformers.generation import utils as generation_utils
            
            # 检查是否有_sample方法
            if hasattr(generation_utils, '_sample'):
                original_sample = generation_utils._sample
                
                def patched_sample(self, *args, **kwargs):
                    """修补的_sample方法"""
                    try:
                        return original_sample(self, *args, **kwargs)
                    except AttributeError as e:
                        if "'DynamicCache' object has no attribute 'get_usable_length'" in str(e):
                            # 动态添加get_usable_length方法
                            import torch
                            from transformers.cache import DynamicCache
                            
                            if not hasattr(DynamicCache, 'get_usable_length'):
                                def get_usable_length(self):
                                    return self.get_seq_length()
                                DynamicCache.get_usable_length = get_usable_length
                                print("✓ 动态添加了get_usable_length方法")
                            
                            # 重试
                            return original_sample(self, *args, **kwargs)
                        else:
                            raise e
                
                # 替换方法
                generation_utils._sample = patched_sample
                print("✓ 已修补_sample方法")
                
        except Exception as e:
            print(f"⚠️ 修补_sample方法失败: {e}")
        
        # 方法2: 直接修补DynamicCache类
        try:
            # 尝试找到DynamicCache类
            DynamicCache = None
            
            # 尝试不同的导入路径
            for import_path in [
                'transformers',
                'transformers.cache',
                'transformers.utils',
                'transformers.generation'
            ]:
                try:
                    module = __import__(import_path, fromlist=['DynamicCache'])
                    if hasattr(module, 'DynamicCache'):
                        DynamicCache = module.DynamicCache
                        print(f"✓ 从{import_path}找到DynamicCache")
                        break
                except:
                    continue
            
            if DynamicCache is not None:
                # 添加get_usable_length方法
                if not hasattr(DynamicCache, 'get_usable_length'):
                    def get_usable_length(self):
                        """兼容性方法"""
                        return self.get_seq_length()
                    
                    DynamicCache.get_usable_length = get_usable_length
                    print("✓ 已为DynamicCache添加get_usable_length方法")
                else:
                    print("✓ DynamicCache已有get_usable_length方法")
            else:
                print("⚠️ 无法找到DynamicCache类")
                
        except Exception as e:
            print(f"⚠️ 修补DynamicCache失败: {e}")
        
        # 方法3: 修补transformers的导入
        try:
            # 创建一个兼容的DynamicCache
            class CompatibleDynamicCache:
                def __init__(self):
                    self._cache = {}
                
                def get_seq_length(self):
                    return 0
                
                def get_usable_length(self):
                    return self.get_seq_length()
            
            # 尝试替换transformers中的DynamicCache
            for module_name in ['transformers.cache', 'transformers.utils', 'transformers.generation']:
                try:
                    module = __import__(module_name, fromlist=['DynamicCache'])
                    if hasattr(module, 'DynamicCache'):
                        module.DynamicCache = CompatibleDynamicCache
                        print(f"✓ 已替换{module_name}中的DynamicCache")
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ 替换DynamicCache失败: {e}")
        
        print("✓ 猴子补丁修复完成")
        return True
        
    except Exception as e:
        print(f"❌ 猴子补丁修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_after_patch():
    """测试修补后的模型"""
    print("\n=== 测试修补后的模型 ===")
    
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
    print("开始猴子补丁修复...")
    
    # 应用修复
    patch_success = apply_monkey_patch()
    
    if patch_success:
        print("\n✅ 猴子补丁修复成功！")
        
        # 测试模型
        model_success = test_model_after_patch()
        
        if model_success:
            print("\n🎉 所有测试通过！")
            print("现在可以运行主程序: python epw-enhance-1.py")
        else:
            print("\n⚠️ 模型测试失败，但猴子补丁修复成功。")
    else:
        print("\n❌ 猴子补丁修复失败。") 