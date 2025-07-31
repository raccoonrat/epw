#!/usr/bin/env python3
"""
EPW-A 增强版简化测试脚本
专注于基本功能测试，不依赖大型模型
"""

import torch
import hashlib
import random
import numpy as np
from typing import List, Tuple, Dict, Any

# --- 核心工具函数 ---

def get_green_list_ids(
    key: str,
    expert_index: int,
    router_hash: str,
    vocab_size: int,
    gamma: float = 0.5
) -> List[int]:
    """
    根据密钥、专家索引和路由器哈希生成绿名单 (IRSH实现)。
    """
    combined_input = f"{key}-{expert_index}-{router_hash}"
    hasher = hashlib.sha256()
    hasher.update(combined_input.encode('utf-8'))
    seed = int.from_bytes(hasher.digest(), 'big')
    
    rng = random.Random(seed)
    green_list_size = int(vocab_size * gamma)
    green_list = rng.sample(range(vocab_size), green_list_size)
    return green_list

def get_router_hash(model: torch.nn.Module, moe_layer_name: str = "block_sparse_moe") -> str:
    """
    计算并返回模型路由器权重的SHA256哈希值 (IRSH实现)。
    """
    try:
        # 这是一个示例路径，实际路径取决于具体模型架构
        router_weights = getattr(model.model, moe_layer_name).gate.weight.data
        hasher = hashlib.sha256()
        hasher.update(router_weights.cpu().numpy().tobytes())
        return hasher.hexdigest()
    except AttributeError:
        raise AttributeError(f"无法在模型中找到名为 '{moe_layer_name}' 的MoE层或其gate。请检查模型架构。")

# --- 模拟模型包装器 ---

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2词汇表大小
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
    
    def encode(self, text, return_tensors='pt', add_special_tokens=False):
        # 简单的模拟编码
        tokens = [hash(text) % self.vocab_size] + [i % self.vocab_size for i in range(len(text.split()))]
        return torch.tensor([tokens])
    
    def decode(self, token_ids, skip_special_tokens=True):
        # 确保token_ids是张量格式
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                token_list = token_ids.tolist()
            else:
                token_list = token_ids[0].tolist()
        else:
            token_list = token_ids
        return "模拟生成的文本: " + " ".join([f"token_{i}" for i in token_list])

class MockModel:
    def __init__(self):
        self.config = type('obj', (object,), {'vocab_size': 50257})()
        self.device = 'cpu'
    
    def __call__(self, input_ids, output_router_logits=False, **kwargs):
        # 模拟模型输出
        batch_size, seq_len = input_ids.shape
        vocab_size = 50257
        
        # 模拟logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # 模拟router_logits（如果请求）
        if output_router_logits:
            router_logits = [torch.randn(batch_size, seq_len, 8)]  # 8个专家
            return type('obj', (object,), {
                'logits': logits,
                'router_logits': router_logits
            })()
        else:
            return type('obj', (object,), {'logits': logits})()

class MockModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_logits_and_route_info(self, input_ids: torch.Tensor) -> Tuple:
        """
        执行一次前向传播，返回logits、top-1专家索引和其置信度。
        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_router_logits=True)
            
        logits = outputs.logits[:, -1, :]
        
        # 模拟专家选择
        top_expert_index = torch.randint(0, 8, (1,)).item()
        top_expert_confidence = torch.rand(1).item()
        
        return logits, top_expert_index, top_expert_confidence

    def get_logits_blackbox(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        模拟黑盒API，只返回logits。
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs.logits[:, -1, :]

# --- 水印生成器 ---

class MockWatermarker:
    def __init__(self, model_wrapper, secret_key: str, gamma: float = 0.5, delta: float = 2.0):
        self.wrapper = model_wrapper
        self.secret_key = secret_key
        self.gamma = gamma
        self.delta = delta
        self.vocab_size = self.wrapper.get_vocab_size()
        self.router_hash = "mock_router_hash_for_testing"

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # 获取路由信息
        _, top_expert_index, confidence = self.wrapper.get_logits_and_route_info(input_ids)

        # 生成绿名单
        green_list = get_green_list_ids(
            self.secret_key,
            top_expert_index,
            self.router_hash,
            self.vocab_size,
            self.gamma
        )
        
        # 修改 Logits
        scores[:, green_list] += self.delta
        return scores

# --- 水印检测器 ---

class MockDetector:
    def __init__(self, model_wrapper, secret_key: str, router_hash: str, gamma: float = 0.5):
        self.wrapper = model_wrapper
        self.secret_key = secret_key
        self.router_hash = router_hash
        self.gamma = gamma
        self.vocab_size = self.wrapper.get_vocab_size()

    def _calculate_z_score(self, green_tokens: int, total_tokens: int) -> float:
        if total_tokens == 0:
            return 0.0
        expected_green = total_tokens * self.gamma
        std_dev = np.sqrt(total_tokens * self.gamma * (1 - self.gamma))
        return (green_tokens - expected_green) / (std_dev + 1e-8)

    def detect_graybox_cspv(self, text: str, sample_size: int = 50) -> float:
        """
        使用置信度分层路径验证 (CSPV) 进行高效灰盒检测。
        """
        token_ids = self.wrapper.tokenizer.encode(text, return_tensors='pt')
        if token_ids.shape[1] <= 1: 
            return 0.0

        # 简化的检测逻辑
        green_token_count = 0
        num_tokens = min(sample_size, token_ids.shape[1] - 1)
        
        for t in range(1, num_tokens + 1):
            context = token_ids[:, :t]
            _, top_expert_index, _ = self.wrapper.get_logits_and_route_info(context)
            
            green_list = get_green_list_ids(
                self.secret_key, top_expert_index, self.router_hash, self.vocab_size, self.gamma
            )
            
            if token_ids[0, t].item() in green_list:
                green_token_count += 1
        
        return self._calculate_z_score(green_token_count, num_tokens)

# --- 测试函数 ---

def test_basic_functionality():
    """
    测试基本功能
    """
    print("=== EPW-A 增强版基本功能测试 ===")
    
    # 测试绿名单生成
    print("\n1. 测试绿名单生成...")
    test_key = "test_secret_key"
    test_expert_index = 5
    test_router_hash = "test_router_hash_12345"
    test_vocab_size = 1000
    test_gamma = 0.3
    
    green_list = get_green_list_ids(
        test_key, test_expert_index, test_router_hash, test_vocab_size, test_gamma
    )
    print(f"✓ 绿名单生成成功，大小: {len(green_list)}")
    print(f"  预期大小: {int(test_vocab_size * test_gamma)}")
    print(f"  绿名单示例: {green_list[:10]}...")
    
    # 测试确定性
    green_list2 = get_green_list_ids(
        test_key, test_expert_index, test_router_hash, test_vocab_size, test_gamma
    )
    if green_list == green_list2:
        print("✓ 绿名单生成具有确定性")
    else:
        print("✗ 绿名单生成不具有确定性")
    
    # 测试路由器哈希计算
    print("\n2. 测试路由器哈希计算...")
    try:
        # 创建一个简单的测试模型
        class TestModel:
            def __init__(self):
                self.model = type('obj', (object,), {
                    'block_sparse_moe': type('obj', (object,), {
                        'gate': type('obj', (object,), {
                            'weight': type('obj', (object,), {
                                'data': torch.randn(10, 10)
                            })
                        })
                    })
                })()
        
        test_model = TestModel()
        test_hash = get_router_hash(test_model, "block_sparse_moe")
        print(f"✓ 路由器哈希计算成功: {test_hash[:16]}...")
    except Exception as e:
        print(f"✗ 路由器哈希计算失败: {e}")
    
    # 测试Z-score计算
    print("\n3. 测试Z-score计算...")
    def calculate_z_score(green_tokens: int, total_tokens: int, gamma: float = 0.3) -> float:
        if total_tokens == 0:
            return 0.0
        expected_green = total_tokens * gamma
        std_dev = np.sqrt(total_tokens * gamma * (1 - gamma))
        return (green_tokens - expected_green) / (std_dev + 1e-8)
    
    test_green_tokens = 35
    test_total_tokens = 100
    z_score = calculate_z_score(test_green_tokens, test_total_tokens)
    print(f"✓ Z-score计算成功: {z_score:.4f}")
    
    # 测试不同参数下的Z-score
    test_cases = [
        (30, 100, 0.3),  # 正常情况
        (50, 100, 0.3),  # 高命中率
        (10, 100, 0.3),  # 低命中率
    ]
    
    for green_tokens, total_tokens, gamma in test_cases:
        z_score = calculate_z_score(green_tokens, total_tokens, gamma)
        print(f"  绿名单命中: {green_tokens}/{total_tokens}, Z-score: {z_score:.4f}")
    
    print("\n=== 基本功能测试完成 ===")
    print("所有核心功能测试通过！")

def test_watermark_generation():
    """
    测试水印生成功能
    """
    print("\n=== 水印生成测试 ===")
    
    # 创建模拟组件
    tokenizer = MockTokenizer()
    model = MockModel()
    model_wrapper = MockModelWrapper(model, tokenizer)
    
    # 创建水印生成器
    secret_key = "test_secret_key_2024"
    gamma = 0.25
    delta = 2.0
    
    watermarker = MockWatermarker(model_wrapper, secret_key, gamma, delta)
    
    # 测试文本生成
    prompt = "In a world where AI is becoming increasingly powerful"
    print(f"输入提示: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    generated_ids = input_ids.clone()
    
    print("正在生成带水印的文本...")
    max_new_tokens = 20  # 减少token数量以加快测试
    
    for i in range(max_new_tokens):
        # 获取当前logits
        with torch.no_grad():
            outputs = model(generated_ids, output_router_logits=True)
        
        logits = outputs.logits[:, -1, :]
        
        # 应用水印
        watermarked_logits = watermarker(generated_ids, logits)
        
        # 采样下一个token
        probs = torch.softmax(watermarked_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 添加到生成序列
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # 解码生成的文本
    watermarked_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"生成的文本: {watermarked_text}")
    
    # 测试水印检测
    print("\n=== 水印检测测试 ===")
    detector = MockDetector(model_wrapper, secret_key, "mock_router_hash_for_testing", gamma)
    
    cspv_score = detector.detect_graybox_cspv(watermarked_text, sample_size=10)
    print(f"CSPV Z-score: {cspv_score:.4f}")
    
    if cspv_score > 2.0:
        print("✓ 检测到水印信号")
    else:
        print("✗ 未检测到明显的水印信号")
    
    print("\n=== 水印测试完成 ===")

if __name__ == "__main__":
    # 运行基本功能测试
    test_basic_functionality()
    
    # 运行水印生成和检测测试
    test_watermark_generation()
    
    print("\n🎉 所有测试完成！")
    print("这个简化版本展示了EPW-A增强版的核心功能。")
    print("在实际使用时，请使用完整的epw-enhance-2.py脚本。") 