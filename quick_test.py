#!/usr/bin/env python3
"""
快速测试脚本 - 验证修复
"""

import torch

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 50257
    
    def encode(self, text, return_tensors='pt', add_special_tokens=False):
        tokens = [hash(text) % self.vocab_size] + [i % self.vocab_size for i in range(len(text.split()))]
        return torch.tensor([tokens])
    
    def decode(self, token_ids, skip_special_tokens=True):
        # 修复后的decode方法
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                token_list = token_ids.tolist()
            else:
                token_list = token_ids[0].tolist()
        else:
            token_list = token_ids
        return "模拟生成的文本: " + " ".join([f"token_{i}" for i in token_list])

def test_decode():
    print("=== 测试decode方法修复 ===")
    
    tokenizer = MockTokenizer()
    
    # 测试编码
    text = "Hello world"
    encoded = tokenizer.encode(text)
    print(f"编码结果: {encoded}")
    print(f"编码形状: {encoded.shape}")
    
    # 测试解码
    try:
        decoded = tokenizer.decode(encoded[0])
        print(f"解码结果: {decoded}")
        print("✓ decode方法修复成功！")
    except Exception as e:
        print(f"✗ decode方法仍有问题: {e}")
    
    # 测试生成序列
    print("\n=== 测试生成序列 ===")
    generated_ids = encoded.clone()
    
    # 模拟添加新token
    for i in range(5):
        new_token = torch.tensor([[i * 100]])
        generated_ids = torch.cat([generated_ids, new_token], dim=1)
        print(f"步骤 {i+1}: 形状 {generated_ids.shape}")
        
        try:
            decoded = tokenizer.decode(generated_ids[0])
            print(f"  解码: {decoded}")
        except Exception as e:
            print(f"  错误: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_decode() 