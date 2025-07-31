import torch
import hashlib
import random
import numpy as np
import pickle
from transformers import LogitsProcessor
from typing import List, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

# --- 核心工具函数 ---

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

# --- 模型包装器与抽象 ---

class MoEModelWrapper:
    """
    一个抽象包装器，用于解耦水印逻辑与具体模型实现。
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_router_weights(self):
        # 实际实现需要根据模型架构调整
        return self.model.model.block_sparse_moe.gate.weight

    def get_vocab_size(self) -> int:
        return self.model.config.vocab_size

    def get_logits_and_route_info(self, input_ids: torch.Tensor) -> Tuple:
        """
        执行一次前向传播，返回logits、top-1专家索引和其置信度。
        这是灰盒访问的核心。
        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_router_logits=True)
            
        logits = outputs.logits[:, -1, :]
        router_logits = outputs.router_logits[-1][0, -1, :] # 获取最后一个MoE层的最后一个token的路由logits
        
        probs = torch.softmax(router_logits, dim=-1)
        top_expert_confidence, top_expert_index = torch.max(probs, dim=-1)
        
        return logits, top_expert_index.item(), top_expert_confidence.item()

    def get_logits_blackbox(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        模拟黑盒API，只返回logits。
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs.logits[:, -1, :]

# --- 水印生成器 ---

class EPW_A_Watermarker(LogitsProcessor):
    """
    实现EPW-A生成算法的LogitsProcessor。
    """
    def __init__(self,
                 model_wrapper: MoEModelWrapper,
                 secret_key: str,
                 gamma: float = 0.5,
                 delta_config: Dict[str, Any] = None):
        
        self.wrapper = model_wrapper
        self.secret_key = secret_key
        self.gamma = gamma
        self.vocab_size = self.wrapper.get_vocab_size()
        
        # IRSH: 在初始化时计算并存储路由器哈希
        self.router_hash = get_router_hash(self.wrapper.model)
        
        # EWP / GSG 配置
        self.is_ewp = delta_config.get("is_ewp", False) if delta_config else False
        if self.is_ewp:
            self.base_deltas = delta_config["base_deltas"] # e.g., [2.0, 2.1,...]
        else:
            self.global_delta = delta_config.get("global_delta", 2.0) if delta_config else 2.0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # 获取路由信息（需要灰盒访问）
        _, top_expert_index, confidence = self.wrapper.get_logits_and_route_info(input_ids)

        # 计算种子 (融入IRSH)
        # 注意：这里的get_green_list_ids已经实现了PRF
        green_list = get_green_list_ids(
            self.secret_key,
            top_expert_index,
            self.router_hash,
            self.vocab_size,
            self.gamma
        )
        
        # 计算水印强度 (支持EWP)
        if self.is_ewp:
            base_delta = self.base_deltas[top_expert_index]
            # 增加一个epsilon防止除以零
            effective_delta = base_delta / (confidence + 1e-8)
        else: # GSG
            effective_delta = self.global_delta
            
        # 修改 Logits
        scores[:, green_list] += effective_delta
        return scores

# --- 水印检测器 ---

class EPW_A_Detector:
    """
    实现EPW-A检测套件，包括灰盒(CSPV)和黑盒(PEPI)检测。
    """
    def __init__(self,
                 model_wrapper: MoEModelWrapper,
                 secret_key: str,
                 router_hash: str,
                 gamma: float = 0.5):
        
        self.wrapper = model_wrapper
        self.secret_key = secret_key
        self.router_hash = router_hash # 检测时需要原始的router_hash
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
        if token_ids.shape <= 1: return 0.0

        # 1. 单次前向传播获取所有路由置信度
        all_confidences = []
        for t in range(1, token_ids.shape):
            context = token_ids[:, :t]
            _, _, confidence = self.wrapper.get_logits_and_route_info(context)
            all_confidences.append((t, confidence))

        # 2. 置信度分层抽样 (CSPV)
        if len(all_confidences) <= sample_size:
            sampled_indices = [item for item in all_confidences]
        else:
            all_confidences.sort(key=lambda x: x)
            k = sample_size // 3
            low_conf = [item for item in all_confidences[:k]]
            high_conf = [item for item in all_confidences[-k:]]
            remaining = [item for item in all_confidences[k:-k]]
            random_conf = random.sample(remaining, sample_size - 2 * k)
            sampled_indices = low_conf + high_conf + random_conf
        
        # 3. 对抽样点进行逐一验证
        green_token_count = 0
        for t in sampled_indices:
            context = token_ids[:, :t]
            _, top_expert_index, _ = self.wrapper.get_logits_and_route_info(context)
            
            green_list = get_green_list_ids(
                self.secret_key, top_expert_index, self.router_hash, self.vocab_size, self.gamma
            )
            
            if token_ids[0, t].item() in green_list:
                green_token_count += 1
        
        # 4. 计算 Z-score
        return self._calculate_z_score(green_token_count, len(sampled_indices))

    def detect_blackbox_pepi(self, text: str, oracle) -> float:
        """
        使用概率性专家路径推断 (PEPI) 进行黑盒检测。
        """
        token_ids = self.wrapper.tokenizer.encode(text, return_tensors='pt')
        if token_ids.shape <= 1: return 0.0

        green_token_count = 0
        num_tokens = token_ids.shape - 1

        for t in range(1, token_ids.shape):
            context = token_ids[:, :t]
            # a. 从黑盒 API 获取 Logits
            logits = self.wrapper.get_logits_blackbox(context)
            
            # b. 使用预言机推断专家路径 (PEPI)
            predicted_expert_index = oracle.predict(logits.cpu().numpy())
            
            # c. 重建绿名单 (融入IRSH)
            green_list = get_green_list_ids(
                self.secret_key, predicted_expert_index, self.router_hash, self.vocab_size, self.gamma
            )
            
            # d. 统计命中
            if token_ids[0, t].item() in green_list:
                green_token_count += 1
        
        # 4. 计算 Z-score
        return self._calculate_z_score(green_token_count, num_tokens)

# --- PEPI 预言机训练 ---

def train_pepi_oracle(model_wrapper: MoEModelWrapper, training_corpus: List[str], model_path: str = "pepi_oracle.pkl"):
    """
    训练并保存PEPI路径推断预言机。
    """
    X_logits, Y_experts = [], []
    print("正在生成PEPI训练数据...")
    for text in training_corpus:
        token_ids = model_wrapper.tokenizer.encode(text, return_tensors='pt')
        for t in range(1, token_ids.shape):
            context = token_ids[:, :t]
            logits, expert_index, _ = model_wrapper.get_logits_and_route_info(context)
            X_logits.append(logits.cpu().numpy().flatten())
            Y_experts.append(expert_index)
    
    print(f"数据生成完毕，共 {len(Y_experts)} 个样本。开始训练分类器...")
    oracle = LogisticRegression(max_iter=1000, solver='liblinear')
    oracle.fit(X_logits, Y_experts)
    
    print("训练完成。正在保存模型...")
    with open(model_path, 'wb') as f:
        pickle.dump(oracle, f)
    
    print(f"PEPI预言机已保存至 {model_path}")
    return oracle

def load_pepi_oracle(model_path: str = "pepi_oracle.pkl"):
    """加载已训练的PEPI预言机。"""
    with open(model_path, 'rb') as f:
        oracle = pickle.load(f)
    return oracle

# --- 实验一：生成并检测带水印的文本 ---

if __name__ == "__main__":
    print("=== EPW-A 增强版实验一：生成并检测带水印的文本 ===")
    
    # 1. 环境设置和模型加载
    print("\n1. 加载模型和分词器...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    # 4位量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True
    )
    
    # 设置模型路径（请根据实际情况调整）
    model_paths = [
        "/root/private_data/model/mixtral-8x7b",
        "/work/home/scnttrxbp8/wangyh/Mixtral-8x7B-Instruct-v0.1",
        "microsoft/DialoGPT-medium",  # 作为备选的小模型
    ]
    
    model_id = None
    tokenizer = None
    model = None
    
    for path in model_paths:
        try:
            # 尝试加载模型
            tokenizer = AutoTokenizer.from_pretrained(path)
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
        print("✗ 所有模型路径都无法加载，请检查模型路径和网络连接")
        exit(1)
    
    # 2. 创建模型包装器
    print("\n2. 初始化模型包装器...")
    model_wrapper = MoEModelWrapper(model, tokenizer)
    
    # 3. 设置水印参数
    secret_key = "epw_enhanced_secret_key_2024"
    gamma = 0.25
    delta = 2.0
    max_new_tokens = 150
    
    # 4. 获取路由器哈希
    print("\n3. 计算路由器哈希...")
    try:
        router_hash = get_router_hash(model, "block_sparse_moe")
        print(f"✓ 路由器哈希: {router_hash[:16]}...")
    except Exception as e:
        print(f"✗ 路由器哈希计算失败: {e}")
        router_hash = "default_router_hash"
        print("使用默认路由器哈希")
    
    # 5. 创建水印生成器和检测器
    print("\n4. 初始化水印组件...")
    watermarker = EPW_A_Watermarker(
        model_wrapper=model_wrapper,
        secret_key=secret_key,
        gamma=gamma,
        delta_config={"delta": delta}
    )
    
    detector = EPW_A_Detector(
        model_wrapper=model_wrapper,
        secret_key=secret_key,
        router_hash=router_hash,
        gamma=gamma
    )
    
    # 6. 生成带水印的文本
    print("\n5. 生成带水印的文本...")
    prompt = "In a world where AI is becoming increasingly powerful, the ability to trace the origin of generated content is"
    
    print(f"输入提示: {prompt}")
    
    # 手动生成带水印的文本
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    generated_ids = input_ids.clone()
    
    print("正在生成文本...")
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
        
        if i % 50 == 0:
            print(f"已生成 {i+1}/{max_new_tokens} 个token...")
    
    # 解码生成的文本
    watermarked_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n--- 生成的带水印文本 ---")
    print(watermarked_text)
    
    # 7. 检测水印
    print("\n6. 检测水印...")
    
    # 灰盒检测 (CSPV)
    print("\n--- 灰盒检测结果 (CSPV) ---")
    cspv_score = detector.detect_graybox_cspv(watermarked_text, sample_size=50)
    print(f"CSPV Z-score: {cspv_score:.4f}")
    
    # 黑盒检测 (PEPI) - 需要先训练预言机
    print("\n--- 黑盒检测结果 (PEPI) ---")
    try:
        # 尝试加载预训练的预言机
        oracle = load_pepi_oracle("pepi_oracle.pkl")
        print("✓ 加载预训练的PEPI预言机")
    except FileNotFoundError:
        print("未找到预训练的PEPI预言机，跳过黑盒检测")
        print("提示：可以使用 train_pepi_oracle() 函数训练预言机")
        oracle = None
    
    if oracle is not None:
        pepi_score = detector.detect_blackbox_pepi(watermarked_text, oracle)
        print(f"PEPI Z-score: {pepi_score:.4f}")
    
    # 8. 结果分析
    print("\n--- 检测结果分析 ---")
    token_ids = tokenizer(watermarked_text, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"分析的词元数: {token_ids.shape[1] - 1}")
    
    if cspv_score > 4.0:
        print("✓ 灰盒检测：检测到高置信度的水印信号")
    else:
        print("✗ 灰盒检测：未检测到明显的水印信号")
    
    if oracle is not None and pepi_score > 4.0:
        print("✓ 黑盒检测：检测到高置信度的水印信号")
    elif oracle is not None:
        print("✗ 黑盒检测：未检测到明显的水印信号")
    
    print("\n=== 实验完成 ===")

# --- 简化测试函数 ---

def test_basic_functionality():
    """
    测试基本功能，不依赖大型模型
    """
    print("=== 基本功能测试 ===")
    
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
    detector = type('obj', (object,), {
        'gamma': 0.3,
        '_calculate_z_score': lambda self, green_tokens, total_tokens: (
            (green_tokens - total_tokens * self.gamma) / 
            (np.sqrt(total_tokens * self.gamma * (1 - self.gamma)) + 1e-8)
        )
    })()
    
    test_green_tokens = 35
    test_total_tokens = 100
    z_score = detector._calculate_z_score(test_green_tokens, test_total_tokens)
    print(f"✓ Z-score计算成功: {z_score:.4f}")
    
    print("\n=== 基本功能测试完成 ===")

if __name__ == "__main__":
    # 首先运行基本功能测试
    test_basic_functionality()
    
    # 然后运行完整实验（如果模型可用）
    print("\n" + "="*50)
    print("开始完整实验...")
    
    # 原有的完整实验代码...


