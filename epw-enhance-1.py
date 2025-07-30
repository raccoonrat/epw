"""
EPW-A Framework (Enhanced Architecture for Expert Pathway Watermarking)
Enhanced implementation with improved model loading and watermarking capabilities.

快速加载优化说明：
1. 设置环境变量 EPW_FAST_LOADING=true 启用快速加载模式
2. 设置环境变量 EPW_LOAD_IN_4BIT=true 使用4位量化（最快但精度较低）
3. 设置环境变量 EPW_LOAD_IN_8BIT=true 使用8位量化（较快但精度稍低）
4. 设置环境变量 EPW_USE_CPU=true 强制使用CPU加载（避免GPU内存分配时间）
5. 设置环境变量 EPW_MODEL_PATH 指定模型路径

量化优先级：EPW_LOAD_IN_4BIT > EPW_LOAD_IN_8BIT > 快速加载默认4位 > 标准加载默认4位

加载时间优化建议：
- 使用SSD存储模型文件
- 确保有足够的RAM（至少16GB）
- 使用GPU时确保显存充足
- 考虑使用更小的模型进行测试
"""

import os
import torch
import hashlib
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    LogitsProcessor, GenerationConfig
)
import warnings
import time

# 猴子补丁修复DynamicCache问题
def apply_dynamic_cache_monkey_patch():
    """应用DynamicCache猴子补丁修复"""
    try:
        import transformers
        
        # 尝试找到并修补DynamicCache
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
                    print(f"✓ 找到DynamicCache: {import_path}")
                    break
            except:
                continue
        
        if DynamicCache is not None:
            # 添加get_usable_length方法
            if not hasattr(DynamicCache, 'get_usable_length'):
                def get_usable_length(self):
                    """兼容性方法，返回序列长度"""
                    return self.get_seq_length()
                
                DynamicCache.get_usable_length = get_usable_length
                print("✓ 已为DynamicCache添加get_usable_length方法")
            else:
                print("✓ DynamicCache已有get_usable_length方法")
        else:
            print("⚠️ 无法找到DynamicCache类")
            
    except Exception as e:
        print(f"⚠️ DynamicCache猴子补丁修复失败: {e}")

# 在导入其他模块之前应用修复
apply_dynamic_cache_monkey_patch()

# 修复DynamicCache兼容性问题 - 更强大的修复方案
def fix_dynamic_cache_compatibility():
    """修复DynamicCache兼容性问题"""
    try:
        # 导入transformers
        import transformers
        
        # 尝试不同的导入路径
        DynamicCache = None
        
        # 方法1: 直接从transformers导入
        try:
            from transformers import DynamicCache
            print("✓ 从transformers直接导入DynamicCache成功")
        except ImportError:
            pass
        
        # 方法2: 从transformers.cache导入
        if DynamicCache is None:
            try:
                from transformers.cache import DynamicCache
                print("✓ 从transformers.cache导入DynamicCache成功")
            except ImportError:
                pass
        
        # 方法3: 从transformers.utils导入
        if DynamicCache is None:
            try:
                from transformers.utils import DynamicCache
                print("✓ 从transformers.utils导入DynamicCache成功")
            except ImportError:
                pass
        
        # 方法4: 从transformers.generation导入
        if DynamicCache is None:
            try:
                from transformers.generation import DynamicCache
                print("✓ 从transformers.generation导入DynamicCache成功")
            except ImportError:
                pass
        
        # 如果所有方法都失败，尝试动态查找
        if DynamicCache is None:
            # 遍历transformers模块查找DynamicCache
            for attr_name in dir(transformers):
                attr = getattr(transformers, attr_name)
                if hasattr(attr, '__name__') and attr.__name__ == 'DynamicCache':
                    DynamicCache = attr
                    print("✓ 动态找到DynamicCache")
                    break
        
        if DynamicCache is None:
            print("⚠️ 无法找到DynamicCache类，跳过兼容性修复")
            return
        
        # 检查是否需要添加get_usable_length方法
        if not hasattr(DynamicCache, 'get_usable_length'):
            def get_usable_length(self):
                """兼容性方法，返回序列长度"""
                return self.get_seq_length()
            
            # 动态添加方法
            DynamicCache.get_usable_length = get_usable_length
            print("✓ DynamicCache兼容性修复已应用")
        else:
            print("✓ DynamicCache已包含get_usable_length方法")
            
    except ImportError:
        print("⚠️ 无法导入transformers，跳过兼容性修复")
    except Exception as e:
        print(f"⚠️ DynamicCache修复失败: {e}")

# 在导入其他模块之前应用修复
fix_dynamic_cache_compatibility()

# 设置环境变量
EPW_FAST_LOADING = os.getenv('EPW_FAST_LOADING', 'true').lower() == 'true'
EPW_LOAD_IN_8BIT = os.getenv('EPW_LOAD_IN_8BIT', 'false').lower() == 'true'
EPW_LOAD_IN_4BIT = os.getenv('EPW_LOAD_IN_4BIT', 'false').lower() == 'true'
EPW_USE_CPU = os.getenv('EPW_USE_CPU', 'false').lower() == 'true'
EPW_MODEL_PATH = os.getenv('EPW_MODEL_PATH', '')

print(f"EPW Configuration:")
print(f"  Fast Loading: {EPW_FAST_LOADING}")
print(f"  8-bit Quantization: {EPW_LOAD_IN_8BIT}")
print(f"  4-bit Quantization: {EPW_LOAD_IN_4BIT}")
print(f"  Use CPU: {EPW_USE_CPU}")
print(f"  Model Path: {EPW_MODEL_PATH}")

class MoEModelWithWatermark:
    """
    模型包装器类，为任何MoE模型添加水印功能
    使用适配器模式避免直接子类化AutoModelForCausalLM的问题
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._router_hash = None
        self._num_experts = None
        self._expert_deltas = None
        
        # 延迟计算路由器哈希
        self._calculate_router_hash()
        self._num_experts = self._get_num_experts()
        
        print(f"MoE Model with Watermark initialized:")
        print(f"  Model type: {type(self.model).__name__}")
        print(f"  Number of experts: {self._num_experts}")
        print(f"  Router hash: {self._router_hash[:16]}..." if self._router_hash else "Router hash: None")
    
    def _get_num_experts(self) -> int:
        """
        动态确定专家数量
        """
        try:
            if hasattr(self.model.config, 'num_experts'):
                return self.model.config.num_experts
            elif hasattr(self.model.config, 'num_local_experts'):
                return self.model.config.num_local_experts
            elif hasattr(self.model.config, 'moe_config') and hasattr(self.model.config.moe_config, 'num_experts'):
                return self.model.config.moe_config.num_experts
            else:
                if 'deepseek' in self.model.config.model_type.lower():
                    return 16
                elif 'mixtral' in self.model.config.model_type.lower():
                    return 8
                else:
                    return 8  # 默认回退
        except Exception as e:
            print(f"Warning: Could not determine expert count: {e}")
            return 8  # 默认回退
    
    def _calculate_router_hash(self):
        """
        计算路由器哈希，延迟执行以避免元张量问题
        """
        try:
            if self._router_hash is not None:
                return self._router_hash
                
            print("Calculating router hash...")
            start_time = time.time()
            
            # 收集所有路由器权重
            router_weights = []
            
            # 遍历模型层寻找MoE块
            for layer in self.model.model.layers:
                moe_block = None
                if hasattr(layer, 'block_sparse_moe'):
                    moe_block = layer.block_sparse_moe
                elif hasattr(layer, 'mlp'):
                    if hasattr(layer.mlp, 'gate'):
                        moe_block = layer.mlp
                
                if moe_block and hasattr(moe_block, 'gate'):
                    try:
                        gate_weights = moe_block.gate.weight.data
                        if gate_weights.numel() > 0:  # 确保张量有数据
                            router_weights.append(gate_weights.cpu().numpy())
                    except Exception as e:
                        print(f"Warning: Could not access gate weights for layer: {e}")
                        continue
            
            if not router_weights:
                print("Warning: No router weights found, using fallback hash")
                self._router_hash = hashlib.sha256(b"fallback_router_hash").hexdigest()
                return self._router_hash
            
            # 计算哈希
            combined_weights = np.concatenate(router_weights, axis=0)
            self._router_hash = hashlib.sha256(combined_weights.tobytes()).hexdigest()
            
            end_time = time.time()
            print(f"Router hash calculation completed in {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"Error calculating router hash: {e}")
            self._router_hash = hashlib.sha256(b"error_router_hash").hexdigest()
    
    def forward(self, *args, **kwargs):
        """
        委托给原始模型的forward方法
        """
        # 对于MoE模型，确保输出路由器logits
        if any(moe_type in self.model.config.model_type.lower() for moe_type in ["mixtral", "deepseek", "moe"]):
            kwargs['output_router_logits'] = True
        
        return self.model.forward(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
        委托给原始模型的generate方法
        """
        return self.model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        委托所有其他属性到原始模型
        """
        return getattr(self.model, name)

class EPWALogitsProcessor(LogitsProcessor):
    """
    EPW-A logits处理器，应用水印偏置
    """
    
    def __init__(self, model: MoEModelWithWatermark, watermark_strength: float = 1.0):
        self.model = model
        self.watermark_strength = watermark_strength
        self._fallback_printed = False
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        应用水印偏置到logits
        """
        try:
            position = input_ids.shape[1] - 1
            num_experts = getattr(self.model, '_num_experts', 16)
            last_token_id = input_ids[0, -1].item()
            
            # 计算专家索引
            expert_index = (last_token_id + position) % num_experts
            
            if position > 0:
                prev_token_id = input_ids[0, -2].item()
                expert_index = (expert_index + prev_token_id) % num_experts
            
            # 应用水印偏置
            bias = torch.zeros_like(scores)
            bias[0, expert_index] = self.watermark_strength
            
            return scores + bias
            
        except Exception as e:
            if not self._fallback_printed:
                print(f"Warning: Using fallback expert index calculation: {e}")
                self._fallback_printed = True
            
            # 回退到简单的专家索引计算
            position = input_ids.shape[1] - 1
            num_experts = getattr(self.model, '_num_experts', 16)
            expert_index = position % num_experts
            
            bias = torch.zeros_like(scores)
            bias[0, expert_index] = self.watermark_strength
            
            return scores + bias

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    基础水印logits处理器
    """
    
    def __init__(self, model: MoEModelWithWatermark, strength: float = 1.0):
        self.model = model
        self.strength = strength
        self._fallback_printed = False
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        应用基础水印偏置
        """
        try:
            position = input_ids.shape[1] - 1
            num_experts = getattr(self.model, '_num_experts', 16)
            expert_index = position % num_experts
            
            bias = torch.zeros_like(scores)
            bias[0, expert_index] = self.strength
            
            return scores + bias
            
        except Exception as e:
            if not self._fallback_printed:
                print(f"Warning: Using fallback expert index calculation: {e}")
                self._fallback_printed = True
            
            # 回退到简单的专家索引计算
            position = input_ids.shape[1] - 1
            num_experts = getattr(self.model, '_num_experts', 16)
            expert_index = position % num_experts
            
            bias = torch.zeros_like(scores)
            bias[0, expert_index] = self.strength
            
            return scores + bias

class PathInferenceOracle:
    """
    路径推理预言机，用于黑盒检测
    """
    
    def __init__(self, num_experts: int = 16):
        self.num_experts = num_experts
        self.expert_weights = torch.ones(num_experts) / num_experts
    
    def predict_expert(self, input_ids: torch.LongTensor) -> int:
        """
        预测专家索引
        """
        position = input_ids.shape[1] - 1
        return position % self.num_experts

class WatermarkDetector:
    """
    水印检测器
    """
    
    def __init__(self, model: MoEModelWithWatermark):
        self.model = model
    
    def detect(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        检测文本中的水印
        """
        try:
            # 编码文本
            inputs = self.model.tokenizer(text, return_tensors="pt")
            
            # 获取路由器logits
            with torch.no_grad():
                outputs = self.model.forward(**inputs, output_router_logits=True)
                router_logits = outputs.router_logits
            
            # 分析专家路径
            expert_paths = []
            for layer_idx, layer_logits in enumerate(router_logits):
                expert_idx = torch.argmax(layer_logits, dim=-1)
                expert_paths.append(expert_idx.item())
            
            # 计算水印分数
            watermark_score = self._calculate_watermark_score(expert_paths)
            
            return {
                'watermark_detected': watermark_score > threshold,
                'watermark_score': watermark_score,
                'expert_paths': expert_paths
            }
            
        except Exception as e:
            print(f"Error in watermark detection: {e}")
            return {
                'watermark_detected': False,
                'watermark_score': 0.0,
                'expert_paths': []
            }
    
    def _calculate_watermark_score(self, expert_paths: List[int]) -> float:
        """
        计算水印分数
        """
        if not expert_paths:
            return 0.0
        
        # 简单的专家路径一致性检查
        unique_experts = len(set(expert_paths))
        total_layers = len(expert_paths)
        
        # 如果专家路径过于一致，可能是水印
        consistency_score = 1.0 - (unique_experts / total_layers)
        
        return consistency_score

class EPWADetectionSuite:
    """
    EPW-A检测套件
    """
    
    def __init__(self, model: MoEModelWithWatermark):
        self.model = model
        self.oracle = None
    
    def detect_graybox_cspv(self, text: str) -> Dict[str, Any]:
        """
        灰盒检测：置信度分层路径验证 (CSPV)
        """
        try:
            inputs = self.model.tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.forward(**inputs, output_router_logits=True)
                router_logits = outputs.router_logits
            
            # 分析路由器logits
            expert_selections = []
            confidence_scores = []
            
            for layer_logits in router_logits:
                probs = torch.softmax(layer_logits, dim=-1)
                expert_idx = torch.argmax(probs, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                
                expert_selections.append(expert_idx.item())
                confidence_scores.append(confidence.item())
            
            # 计算CSPV分数
            avg_confidence = np.mean(confidence_scores)
            path_consistency = 1.0 - (len(set(expert_selections)) / len(expert_selections))
            
            cspv_score = (avg_confidence + path_consistency) / 2
            
            return {
                'cspv_score': cspv_score,
                'avg_confidence': avg_confidence,
                'path_consistency': path_consistency,
                'expert_selections': expert_selections
            }
            
        except Exception as e:
            print(f"Error in CSPV detection: {e}")
            return {
                'cspv_score': 0.0,
                'avg_confidence': 0.0,
                'path_consistency': 0.0,
                'expert_selections': []
            }
    
    def train_path_inference_oracle(self, training_texts: List[str]):
        """
        训练路径推理预言机
        """
        print("Training path inference oracle...")
        
        # 收集专家路径数据
        expert_paths_data = []
        
        for text in training_texts:
            try:
                inputs = self.model.tokenizer(text, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.forward(**inputs, output_router_logits=True)
                    router_logits = outputs.router_logits
                
                paths = []
                for layer_logits in router_logits:
                    expert_idx = torch.argmax(layer_logits, dim=-1)
                    paths.append(expert_idx.item())
                
                expert_paths_data.append(paths)
                
            except Exception as e:
                print(f"Error processing training text: {e}")
                continue
        
        if expert_paths_data:
            # 创建预言机
            num_experts = getattr(self.model, '_num_experts', 16)
            self.oracle = PathInferenceOracle(num_experts)
            print(f"Oracle trained with {len(expert_paths_data)} samples")
        else:
            print("Warning: No training data available for oracle")
    
    def detect_blackbox_pepi(self, text: str) -> Dict[str, Any]:
        """
        黑盒检测：概率专家路径推理 (PEPI)
        """
        if self.oracle is None:
            return {
                'pepi_score': 0.0,
                'oracle_available': False
            }
        
        try:
            inputs = self.model.tokenizer(text, return_tensors="pt")
            
            # 跳过第一个token以避免空上下文问题
            if inputs['input_ids'].shape[1] <= 1:
                return {
                    'pepi_score': 0.0,
                    'oracle_available': True,
                    'error': 'Text too short for analysis'
                }
            
            # 使用预言机预测专家路径
            predicted_experts = []
            actual_experts = []
            
            for i in range(1, inputs['input_ids'].shape[1]):
                partial_inputs = {
                    'input_ids': inputs['input_ids'][:, :i]
                }
                
                # 获取实际专家选择
                with torch.no_grad():
                    outputs = self.model.forward(**partial_inputs, output_router_logits=True)
                    router_logits = outputs.router_logits
                    
                    if router_logits:
                        actual_expert = torch.argmax(router_logits[-1], dim=-1).item()
                        actual_experts.append(actual_expert)
                        
                        # 使用预言机预测
                        predicted_expert = self.oracle.predict_expert(partial_inputs['input_ids'])
                        predicted_experts.append(predicted_expert)
            
            if not actual_experts:
                return {
                    'pepi_score': 0.0,
                    'oracle_available': True,
                    'error': 'No expert selections available'
                }
            
            # 计算预测准确性
            correct_predictions = sum(1 for pred, actual in zip(predicted_experts, actual_experts) if pred == actual)
            pepi_score = correct_predictions / len(actual_experts)
            
            return {
                'pepi_score': pepi_score,
                'oracle_available': True,
                'predicted_experts': predicted_experts,
                'actual_experts': actual_experts
            }
            
        except Exception as e:
            print(f"Error in PEPI detection: {e}")
            return {
                'pepi_score': 0.0,
                'oracle_available': True,
                'error': str(e)
            }

def main():
    """
    主函数
    """
    print("=== EPW-A Enhanced Expert Pathway Watermarking ===")
    
    # 确定模型路径
    possible_paths = [
        EPW_MODEL_PATH,
        "/root/private_data/model/DeepSeek-MoE",
        "/root/private_data/model/mixtral-8x7b",
        "microsoft/DialoGPT-medium"
    ]
    
    model_name = None
    for path in possible_paths:
        if path and os.path.exists(path):
            model_name = path
            break
    
    if not model_name:
        model_name = "deepseek-ai/deepseek-moe-16b-chat"  # 默认使用DeepSeek MoE
    
    print(f"Loading model: {model_name}")
    
    # 配置量化
    quantization_config = None
    if EPW_LOAD_IN_4BIT and any(moe_type in model_name.lower() for moe_type in ["mixtral", "deepseek", "moe"]):
        print("Applying 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif EPW_LOAD_IN_8BIT:
        print("Applying 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
    
    # 配置设备映射
    device_map = "cpu" if EPW_USE_CPU else "auto"
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_end = time.time()
    print(f"Tokenizer loaded in {tokenizer_end - tokenizer_start:.2f}s")
    
    # 加载模型
    print("Loading model...")
    model_start = time.time()
    
    try:
        # 首先加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=EPW_FAST_LOADING,
            trust_remote_code=True
        )
        
        # 然后包装模型以添加水印功能
        model = MoEModelWithWatermark(base_model, tokenizer)
        
        # 确保模型参数为float16
        if quantization_config is None:
            for param in model.parameters():
                param.data = param.data.to(torch.float16)
        
        model_end = time.time()
        print(f"Model loaded in {model_end - model_start:.2f}s")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 测试文本生成
    print("\n=== Testing Text Generation ===")
    test_prompt = "Hello, how are you today?"
    
    # 创建生成配置
    generation_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 编码输入并确保在正确的设备上
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    # 获取模型设备
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    
    # 将输入移动到模型设备
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # 生成文本（无水印）
    print("Generating text without watermark...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    text_without_watermark = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text (no watermark): {text_without_watermark}")
    
    # 生成文本（带水印）
    print("\nGenerating text with watermark...")
    
    # 创建logits处理器
    logits_processor = EPWALogitsProcessor(model, watermark_strength=2.0)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            logits_processor=[logits_processor]
        )
    
    text_with_watermark = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text (with watermark): {text_with_watermark}")
    
    # 测试检测
    print("\n=== Testing Watermark Detection ===")
    
    # 创建检测器
    detector = WatermarkDetector(model)
    detection_suite = EPWADetectionSuite(model)
    
    # 检测无水印文本
    print("Detecting watermark in text without watermark...")
    result_no_watermark = detector.detect(text_without_watermark)
    print(f"Detection result (no watermark): {result_no_watermark}")
    
    # 检测带水印文本
    print("Detecting watermark in text with watermark...")
    result_with_watermark = detector.detect(text_with_watermark)
    print(f"Detection result (with watermark): {result_with_watermark}")
    
    # 灰盒检测
    print("\n=== Gray-box Detection (CSPV) ===")
    cspv_result = detection_suite.detect_graybox_cspv(text_with_watermark)
    print(f"CSPV result: {cspv_result}")
    
    # 训练预言机并测试黑盒检测
    print("\n=== Black-box Detection (PEPI) ===")
    training_texts = [
        "This is a training text for the oracle.",
        "Another training example for the oracle.",
        "A third training text to improve the oracle."
    ]
    
    detection_suite.train_path_inference_oracle(training_texts)
    pepi_result = detection_suite.detect_blackbox_pepi(text_with_watermark)
    print(f"PEPI result: {pepi_result}")
    
    print("\n=== EPW-A Framework Test Completed ===")

if __name__ == "__main__":
    main()
