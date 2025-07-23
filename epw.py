# 1. 环境设置
# !pip install transformers torch accelerate bitsandbytes huggingface_hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LogitsProcessor

# 设置 Token
token = "hf_XXX"

# 使用一个公开的MoE模型，并以4位量化加载以节省资源
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#model_id = "/work/home/scnttrxbp8/wangyh/Mixtral-8x7B-Instruct-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    token=token,
)

# 获取词汇表大小，用于后续的绿/红名单划分
vocab_size = model.config.vocab_size


# 3.1 辅助函数：绿名单生成
import hashlib

def get_green_list(seed, vocab_size, gamma):
    """ deterministically generates a list of 'green' token ids """
    rng = np.random.default_rng(seed)
    green_list_size = int(vocab_size * gamma)
    green_list = rng.choice(vocab_size, size=green_list_size, replace=False)
    return set(green_list)

def get_green_list_ids(seed, vocab_size, gamma=0.5):
    """
    根据给定的种子，确定性地生成绿名单。
    """
    # 使用哈希确保确定性
    hashed_seed_str = str(seed)
    # 创建一个torch的随机数生成器并设置种子
    generator = torch.Generator()
    generator.manual_seed(int(hashed_seed_str))

    # 生成一个随机排列
    vocab_permutation = torch.randperm(vocab_size, generator=generator)

    # 根据gamma比例切分，得到绿名单
    green_list_size = int(vocab_size * gamma)
    green_list_ids = vocab_permutation[:green_list_size]

    return green_list_ids


# 3.2 水印生成器
def generate_and_watermark_manually(prompt, model, tokenizer, max_new_tokens, gamma, delta):
    """
    Generates text token-by-token, applying the GSG watermark at each step.
    This manual loop ensures perfect sync between router logits and vocabulary logits.
    """
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    generated_ids = input_ids
    past_key_values = None

    for _ in range(max_new_tokens):
        # 1. Get model outputs from a single forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                output_router_logits=True,
                use_cache=True,
                past_key_values=past_key_values
            )

        # 2. Extract necessary information
        # Get logits for the very last token
        next_token_logits = outputs.logits[:, -1, :]
        # Get router logits for the last token of the last MoE layer
        last_layer_router_logits = outputs.router_logits[-1]

        if last_layer_router_logits.dim() == 3:
            # Case: [batch_size, sequence_length, num_experts]
            router_logits_for_last_token = last_layer_router_logits[:, -1, :]
        else:
            # Case: [batch_size, num_experts]
            router_logits_for_last_token = last_layer_router_logits
        
        # Store the cache for the next iteration
        past_key_values = outputs.past_key_values

        # 3. Determine the expert choice and create the green list
        #top_expert_idx = torch.argmax(router_logits, dim=-1).item()
        top_expert_idx = torch.argmax(router_logits_for_last_token).item()
        green_list = get_green_list(top_expert_idx, tokenizer.vocab_size, gamma)

        # 4. Apply the watermark bias
        bias = torch.zeros_like(next_token_logits)
        bias[:, list(green_list)] = delta
        watermarked_logits = next_token_logits + bias

        # 5. Sample the next token from the biased distribution
        # You can use top-k, top-p, or temperature here as needed
        next_token = torch.argmax(watermarked_logits, dim=-1).unsqueeze(-1)

        # 6. Append the new token and prepare for the next loop
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Decode the final generated text
    return tokenizer.decode(generated_ids[0])


def generate_with_gsg_watermark(prompt, model, tokenizer, max_new_tokens=100, gamma=0.5, delta=2.0, secret_key=15485863):
    """
    使用GSG水印方案生成文本。
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    generated_ids = input_ids

    print("生成中...")
    for step in range(max_new_tokens):
        # 1. 前向传播以获取logits和路由决策
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                output_router_logits=True, # 关键：获取路由器logits
                return_dict=True
            )

        # 2. 提取顶级专家索引作为水印种子
        # 我们使用最后一个MoE层的最后一个token的路由决策
        # Mixtral的MoE层在奇数层，我们取最后一层(31)
         last_layer_router_logits = outputs.router_logits[-1]

        if last_layer_router_logits.dim() == 3:
            # Case: [batch_size, sequence_length, num_experts]
            router_logits_for_last_token = last_layer_router_logits[:, -1, :]
        else:
            # Case: [batch_size, num_experts]
            router_logits_for_last_token = last_layer_router_logits
        
        # top_expert_index = torch.argmax(router_logits).item()
        top_expert_index = torch.argmax(router_logits_for_last_token).item()

        # 3. 生成水印种子并划分绿/红名单
        # 种子由密钥和专家索引共同决定
        watermark_seed = hash(str(secret_key) + str(top_expert_index))
        green_list_ids = get_green_list_ids(watermark_seed, vocab_size, gamma)

        # 4. 修改Logits
        next_token_logits = outputs.logits[0, -1, :]
        next_token_logits[green_list_ids] += delta

        # 5. 从修改后的分布中采样
        # 应用softmax将logits转换为概率
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(0)

        # 6. 更新序列
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

        # 简单的停止条件
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# 3.3 水印检测器
import numpy as np
from scipy.stats import norm

def detect_gsg_watermark(text, model, tokenizer, gamma=0.5, secret_key=15485863):
    """
    检测给定文本中是否存在GSG水印。
    """
    device = model.device
    tokenized_text = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    if tokenized_text.shape[1] < 2:
        print("文本太短，无法检测。")
        return None, None, None

    green_token_count = 0
    num_tokens_analyzed = tokenized_text.shape[1] - 1

    print("检测中...")
    for t in range(1, tokenized_text.shape[1]):
        # 获取当前词元的前文
        context_ids = tokenized_text[:, :t]

        # 1. 重建专家路径
        with torch.no_grad():
            outputs = model(
                input_ids=context_ids,
                output_router_logits=True,
                return_dict=True
            )

        # Handle both 2D and 3D router_logits tensors
        last_layer_router_logits = outputs.router_logits[-1]
        if last_layer_router_logits.dim() == 3:
            # Case: [batch_size, sequence_length, num_experts]
            router_logits_for_last_token = last_layer_router_logits[0, -1, :]
        else:
            # Case: [batch_size, num_experts] (for sequence_length == 1)
            router_logits_for_last_token = last_layer_router_logits[0, :]

        top_expert_index = torch.argmax(router_logits_for_last_token).item()

        # 2. 重建绿名单
        watermark_seed = hash(str(secret_key) + str(top_expert_index))
        #green_list_ids = get_green_list_ids(watermark_seed, vocab_size, gamma)
        # --- FIX IS HERE ---
        # Move the green list tensor to the same device as the model and tokenized_text
        green_list_ids = get_green_list_ids(watermark_seed, vocab_size, gamma).to(device)

        # 3. 检查实际词元是否在绿名单中
        actual_token_id = tokenized_text[0, t]
        if actual_token_id in green_list_ids:
            green_token_count += 1

    # 4. 计算Z-score
    expected_green_tokens = num_tokens_analyzed * gamma
    std_dev = np.sqrt(num_tokens_analyzed * gamma * (1 - gamma))

    if std_dev == 0:
        return float('inf'), 0.0, green_token_count

    z_score = (green_token_count - expected_green_tokens) / std_dev
    p_value = 1 - norm.cdf(z_score) # 单尾检验

    return z_score, p_value, green_token_count


# 4.1 实验一：生成并检测带水印的文本
prompt = "In a world where AI is becoming increasingly powerful, the ability to trace the origin of generated content is"
max_new_tokens = 150
gamma = 0.25
delta = 2.0

# 生成带水印的文本
# watermarked_text = generate_with_gsg_watermark(prompt, model, tokenizer, max_new_tokens, gamma, delta)
watermarked_text = generate_and_watermark_manually(prompt, model, tokenizer, max_new_tokens, gamma, delta)


print("\n--- 生成的带水印文本 ---")
print(watermarked_text)

# 检测水印
z_score, p_value, count = detect_gsg_watermark(watermarked_text, model, tokenizer, gamma)
#z_score, green_hits, total_tokens = detect_gsg_watermark(watermarked_text, prompt, model, tokenizer, gamma)
#print(watermarked_text)

print("\n--- 检测结果 ---")
token_ids = tokenizer(watermarked_text, return_tensors="pt", add_special_tokens=False).input_ids
print(f"分析的词元数: {token_ids.shape[1] - 1}")
print(f"绿名单命中数: {count}")
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.10f}")

#print("\n--- 检测结果 ---")
#print(f"分析的词元数: {total_tokens}")
#print(f"绿名单命中数: {green_hits}")
#print(f"Z-score: {z_score:.4f}")
#print(f"P-value: {1 - stats.norm.cdf(z_score):.10f}\n") # Using scipy.stats

if z_score > 4.0: # 一个常用的高置信度阈值
    print("\n结论: 检测到高置信度的水印信号。")
else:
    print("\n结论: 未检测到明显的水印信号。")


