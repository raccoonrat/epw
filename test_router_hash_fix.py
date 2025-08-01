#!/usr/bin/env python3
"""
测试路由器哈希修复的简单脚本
"""

import torch
import hashlib
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def get_router_hash(model: torch.nn.Module, moe_layer_name: str = "block_sparse_moe") -> str:
    """
    计算并返回模型路由器权重的SHA256哈希值 (IRSH实现)。
    """
    try:
        # 尝试多种可能的模型架构路径
        router_weights = []
        
        # 方法1: 直接访问model.model.block_sparse_moe (原始假设)
        try:
            router_weights = getattr(model.model, moe_layer_name).gate.weight.data
            hasher = hashlib.sha256()
            hasher.update(router_weights.cpu().numpy().tobytes())
            return hasher.hexdigest()
        except AttributeError:
            pass
        
        # 方法2: 遍历所有层，查找MoE层 (Mixtral架构)
        try:
            for layer in model.model.layers:
                if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                    gate_weights = layer.block_sparse_moe.gate.weight.data
                    
                    # 检查是否为meta tensor
                    if hasattr(gate_weights, 'is_meta') and gate_weights.is_meta:
                        print("警告: 路由器权重是meta tensor，使用备用哈希")
                        raise RuntimeError("检测到meta tensor")
                    
                    # 安全访问数据
                    try:
                        weight_bytes = gate_weights.cpu().numpy().tobytes()
                        router_weights.append(weight_bytes)
                    except Exception as e:
                        print(f"警告: 无法访问路由器权重数据: {e}")
                        raise RuntimeError(f"无法访问路由器权重数据: {e}")
            
            if router_weights:
                # 连接所有路由器权重并哈希
                combined_weights = b''.join(router_weights)
                hasher = hashlib.sha256()
                hasher.update(combined_weights)
                return hasher.hexdigest()
        except Exception as e:
            print(f"方法2失败: {e}")
        
        # 方法3: 尝试其他常见的MoE层名称
        alternative_names = ['moe', 'experts', 'router', 'gate']
        for name in alternative_names:
            try:
                if hasattr(model.model, name):
                    router_weights = getattr(model.model, name).weight.data
                    hasher = hashlib.sha256()
                    hasher.update(router_weights.cpu().numpy().tobytes())
                    return hasher.hexdigest()
            except AttributeError:
                continue
        
        # 如果所有方法都失败，抛出错误
        raise AttributeError(f"无法在模型中找到MoE层或其gate。请检查模型架构。")
        
    except Exception as e:
        raise AttributeError(f"路由器哈希计算失败: {e}")

def test_router_hash():
    """测试路由器哈希计算功能"""
    print("=== 测试路由器哈希计算 ===")
    
    # 设置模型路径
    model_paths = [
        "/root/private_data/model/mixtral-8x7b", 
        "/work/home/scnttrxbp8/wangyh/Mixtral-8x7B-Instruct-v0.1",
        "microsoft/DialoGPT-small",  # 小型模型，适合测试
        "gpt2",  # 标准GPT-2模型
    ]
    
    # 4位量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True
    )
    
    model_id = None
    model = None
    
    for path in model_paths:
        try:
            print(f"尝试加载模型: {path}")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )
            model_id = path
            print(f"✓ 成功加载模型: {path}")
            break
        except Exception as e:
            print(f"✗ 无法加载模型 {path}: {e}")
            continue
    
    if model_id is None:
        print("✗ 所有模型路径都无法加载，使用模拟模式")
        return
    
    # 测试路由器哈希计算
    print("\n测试路由器哈希计算...")
    try:
        router_hash = get_router_hash(model)
        print(f"✓ 路由器哈希计算成功: {router_hash[:16]}...")
        return True
    except Exception as e:
        print(f"✗ 路由器哈希计算失败: {e}")
        return False

if __name__ == "__main__":
    success = test_router_hash()
    if success:
        print("\n✓ 路由器哈希修复测试通过")
    else:
        print("\n✗ 路由器哈希修复测试失败") 