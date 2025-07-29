# EPW-A Framework

**增强型专家路径水印（Enhanced Expert Pathway Watermarking）框架**

一个用于MoE（专家混合）大语言模型的先进水印系统，实现了文档中提出的EPW-A完整算法。

## 🚀 特性

### 核心算法
- **IRSH协议**：初始路由器状态哈希，增强安全性
- **GSG模式**：门控SEED绿名单，全局delta策略
- **EWP模式**：专家特异性加权扰动，动态delta策略

### 检测套件
- **CSPV**：置信度分层路径验证，高效灰盒检测
- **PEPI**：概率性专家路径推断，黑盒检测
- **向后兼容**：支持原始EPW检测方法

### 技术优势
- ✅ 与模型架构深度融合
- ✅ 抗PEFT攻击（通过IRSH）
- ✅ 高效检测（通过CSPV）
- ✅ 黑盒支持（通过PEPI）
- ✅ 低模型演进约束

## 📦 安装

```bash
# 克隆项目
git clone <repository-url>
cd epw

# 安装依赖
pip install -r requirements.txt

# 详细安装指南请参考 INSTALL.md
```

## 🎯 快速开始

```python
# 基本使用示例
from epw_enhance_1 import EPWALogitsProcessor, EPWADetectionSuite

# 创建EPW-A处理器
processor = EPWALogitsProcessor(
    vocab_size=model.config.vocab_size,
    gamma=0.5,
    secret_key="your_secret_key",
    router_hash=model.router_hash,
    mode="ewp",  # 或 "gsg"
    delta_config=expert_deltas
)

# 生成水印文本
output = model.generate(
    input_ids,
    logits_processor=[processor],
    max_new_tokens=50
)

# 检测水印
detection_suite = EPWADetectionSuite(tokenizer, model, secret_key)
result = detection_suite.detect_graybox_cspv(text, sample_size=30)
```

## 📊 性能对比

| 指标 | KGW | EPW-EWP | **EPW-A** |
|------|-----|---------|-----------|
| 抗转述鲁棒性 | 中等 | 非常高 | **非常高** |
| 抗PEFT鲁棒性 | 低 | 低 | **高** |
| 检测开销 | 极低 | 高 | **中等** |
| 访问权限 | 公开密钥 | 灰盒 | **黑盒** |

## 🔧 配置选项

### 水印模式
- **GSG**：全局固定delta，适合简单场景
- **EWP**：专家特异性delta，适合复杂场景

### 检测方法
- **Legacy**：原始EPW检测，完整但昂贵
- **CSPV**：分层抽样检测，高效且准确
- **PEPI**：黑盒检测，适用于API场景

## 📚 文档

- [安装指南](INSTALL.md)：详细的安装和配置说明
- [核心思想分析](docs/0729-EPW核心思想分析与修正.md)：理论背景和算法设计
- [API文档](docs/API.md)：详细的API参考（待补充）

## 🧪 测试

```bash
# 运行完整演示
python epw-enhance-1.py

# 这将展示：
# - GSG和EWP水印生成
# - CSPV灰盒检测
# - PEPI黑盒检测
# - 性能对比分析
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

基于原始EPW论文的思想，并进行了重要的理论增强和实践改进。

---

**注意**：首次运行需要下载Mixtral-8x7B模型（约50GB），请确保有稳定的网络连接和足够的存储空间。
