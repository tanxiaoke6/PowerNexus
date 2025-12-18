# ⚡ PowerNexus

> **自主电网巡检与动态调度系统** - Powered by Qwen2.5

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目概述

PowerNexus 是一个基于 **Qwen2.5 大模型** 的智能电网巡检与决策系统，集成了：

- 🔍 **Qwen2.5-VL 视觉检测** - 无人机图像缺陷识别
- 📚 **RAG 知识库检索** - 电力技术标准智能问答
- 🤖 **PPO 强化学习决策** - 电网拓扑优化调度
- 🌐 **Streamlit 可视化界面** - 一站式操作仪表板

### 工作流程

```
感知 (See) → 咨询 (Think) → 决策 (Decide) → 执行 (Act)
    ↓            ↓              ↓             ↓
 Qwen-VL     ChromaDB         PPO          Report
```

---

## 💻 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA 8GB VRAM | NVIDIA 16GB+ VRAM |
| RAM | 16 GB | 32 GB |
| 存储 | 20 GB | 50 GB |
| CUDA | 11.8+ | 12.1+ |

> ⚠️ **显存说明**: 
> - 4-bit 量化模式下，Qwen2.5-VL-7B + Qwen2.5-7B 约需 **~16GB VRAM**
> - CPU 模式可运行但推理速度较慢
> - Mock 模式无需 GPU，用于开发测试

---

## 🚀 快速开始

### 1. 创建环境 (推荐 Conda)

```bash
# 创建 conda 环境
conda create -n powernexus python=3.10 -y
conda activate powernexus

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-org/PowerNexus.git
cd PowerNexus

# 安装依赖
pip install -r requirements.txt
```

### 3. 生成 Mock 数据

```bash
python tools/generate_mock_data.py
```

### 4. 启动 Web UI

```bash
streamlit run src/app.py
```

或使用一键启动脚本：

```bash
python start_powernexus.py
```

访问 http://localhost:8501 即可使用仪表板。

---

## 📁 项目结构

```
PowerNexus/
├── src/
│   ├── app.py                  # Streamlit 仪表板
│   ├── main.py                 # 主工作流 (MainAgent)
│   ├── perception/             # 多模态感知模块
│   │   ├── __init__.py
│   │   └── vision_model.py     # Qwen2.5-VL 封装
│   ├── rag/                    # RAG 知识库模块
│   │   ├── __init__.py
│   │   ├── ingest.py           # 文档摄入
│   │   └── retriever.py        # 知识检索
│   ├── rl_engine/              # 强化学习模块
│   │   ├── __init__.py
│   │   ├── env_wrapper.py      # Grid2Op 环境封装
│   │   └── agent.py            # PPO 智能体
│   └── utils/
│       └── llm_engine.py       # Qwen LLM 引擎
├── tools/
│   └── generate_mock_data.py   # Mock 数据生成器
├── data/
│   ├── images/                 # 巡检图像
│   └── manuals/                # 技术标准 PDF
├── config/                     # 配置文件
├── models/                     # 模型权重
├── tests/                      # 测试用例
├── run_demo.py                 # 命令行演示
├── start_powernexus.py         # 一键启动脚本
├── requirements.txt            # 依赖清单
├── .env.example                # 环境变量示例
└── README.md                   # 本文件
```

---

## 🎮 使用指南

### Streamlit 仪表板

| Tab | 功能 | 操作 |
|-----|------|------|
| 🔍 视觉检测 | Qwen-VL 图像分析 | 上传图像 → 点击分析 |
| 📚 知识检索 | RAG 问答 | 输入问题 → 获取答案 |
| 🤖 RL 优化 | PPO 决策 | 设置负载 → 获取建议 |
| 📊 系统信息 | 状态监控 | 查看 GPU/模块状态 |

### 命令行演示

```bash
# 交互式演示
python run_demo.py

# 直接运行工作流
python -m src.main
```

### Python API

```python
from src.perception import create_detector
from src.rag import init_knowledge_base_with_samples
from src.rl_engine import create_ppo_agent

# 缺陷检测
detector = create_detector(use_mock=True)
result = detector.detect("image.jpg")
print(result.defect_type, result.severity)

# 知识检索
kb = init_knowledge_base_with_samples(use_mock=True)
result = kb.query_and_synthesize("变压器测试要求?")
print(result.synthesized_answer)

# RL 决策
agent = create_ppo_agent(use_mock=True)
action = agent.predict_action(observation)
```

---

## ⚙️ 配置说明

### 环境变量 (.env)

```bash
# Qwen 模型路径 (本地)
QWEN_VL_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
QWEN_LLM_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct

# 量化配置
USE_4BIT_QUANTIZATION=true

# 向量库路径
VECTOR_DB_PATH=./data/vector_db
```

### 模型下载

首次运行时会自动从 HuggingFace 下载模型。也可手动下载：

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

---

## 🧪 测试

```bash
# 运行测试
pytest tests/

# 模块测试
python -m src.perception.vision_model
python -m src.rag.retriever
python -m src.rl_engine.agent
```

---

## 📄 技术栈

| 组件 | 技术 |
|------|------|
| 视觉模型 | Qwen2.5-VL-7B-Instruct |
| 文本模型 | Qwen2.5-7B-Instruct |
| 嵌入模型 | sentence-transformers |
| 向量库 | ChromaDB |
| RL 算法 | PPO (Stable-Baselines3) |
| 电网仿真 | Grid2Op |
| Web 框架 | Streamlit |
| 量化 | BitsAndBytes 4-bit |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📜 许可证

MIT License © 2024 PowerNexus Team

---

<div align="center">
  <b>PowerNexus - 智能电网巡检与决策系统</b><br>
  Powered by Qwen2.5 | Made with ❤️
</div>
