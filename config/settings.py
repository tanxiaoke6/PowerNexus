# -*- coding: utf-8 -*-
"""
PowerNexus - 全局配置文件

本文件包含项目的所有配置参数。
可通过环境变量覆盖默认值。

使用方式:
    from config.settings import config
    print(config.QWEN_VL_MODEL)

作者: PowerNexus Team
日期: 2025-12-18
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANUALS_DIR = DATA_DIR / "manuals"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================================
# 模型配置
# ============================================================================

@dataclass
class QwenVLConfig:
    """Qwen-VL 视觉语言模型配置"""
    # 模型名称/路径
    model_name: str = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    # 运行设备
    device: str = os.getenv("DEVICE", "auto")
    # 是否使用 4-bit 量化
    use_4bit: bool = os.getenv("USE_4BIT", "true").lower() == "true"
    # 是否使用 Flash Attention
    use_flash_attention: bool = True
    # 最大生成 token 数
    max_new_tokens: int = 1024
    # 生成温度
    temperature: float = 0.2
    # 是否信任远程代码
    trust_remote_code: bool = True


@dataclass
class QwenLLMConfig:
    """Qwen 文本语言模型配置"""
    # 模型名称/路径
    model_name: str = os.getenv("QWEN_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    # 运行设备
    device: str = os.getenv("DEVICE", "auto")
    # 是否使用 4-bit 量化
    use_4bit: bool = os.getenv("USE_4BIT", "true").lower() == "true"
    # 最大生成 token 数
    max_new_tokens: int = 2048
    # 生成温度
    temperature: float = 0.7


@dataclass
class RAGConfig:
    """RAG 知识库配置"""
    # 向量库目录
    persist_directory: str = str(VECTOR_DB_DIR)
    # 集合名称
    collection_name: str = "power_grid_standards"
    # 嵌入模型
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # 默认检索数量
    top_k: int = 5
    # 相似度阈值
    score_threshold: float = 0.5


@dataclass
class RLConfig:
    """强化学习配置"""
    # 环境名称
    env_name: str = "l2rpn_case14_sandbox"
    # 学习率
    learning_rate: float = 3e-4
    # 折扣因子
    gamma: float = 0.99
    # 训练步数
    total_timesteps: int = 1_000_000


@dataclass
class AppConfig:
    """应用配置"""
    # 是否使用 Mock 模式
    use_mock: bool = os.getenv("USE_MOCK", "true").lower() == "true"
    # 日志级别
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # 服务端口
    port: int = int(os.getenv("PORT", "8501"))


# ============================================================================
# 全局配置实例
# ============================================================================

@dataclass
class PowerNexusConfig:
    """PowerNexus 全局配置"""
    # 子配置
    qwen_vl: QwenVLConfig = field(default_factory=QwenVLConfig)
    qwen_llm: QwenLLMConfig = field(default_factory=QwenLLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    app: AppConfig = field(default_factory=AppConfig)
    
    # 路径
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    images_dir: Path = IMAGES_DIR
    models_dir: Path = MODELS_DIR


# 全局配置实例
config = PowerNexusConfig()


# ============================================================================
# 便捷函数
# ============================================================================

def get_config() -> PowerNexusConfig:
    """获取全局配置"""
    return config


def is_mock_mode() -> bool:
    """是否为 Mock 模式"""
    return config.app.use_mock


def is_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("PowerNexus 配置:")
    print(f"  项目根目录: {config.project_root}")
    print(f"  Mock 模式: {config.app.use_mock}")
    print(f"  CUDA 可用: {is_cuda_available()}")
    print(f"  Qwen-VL 模型: {config.qwen_vl.model_name}")
    print(f"  Qwen-LLM 模型: {config.qwen_llm.model_name}")
    print(f"  4-bit 量化: {config.qwen_vl.use_4bit}")
