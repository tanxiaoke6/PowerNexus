# -*- coding: utf-8 -*-
"""
PowerNexus - 配置模块

使用方式:
    from config import config, get_config, is_mock_mode
"""

from .settings import (
    config,
    get_config,
    is_mock_mode,
    is_cuda_available,
    PowerNexusConfig,
    QwenVLConfig,
    QwenLLMConfig,
    RAGConfig,
    RLConfig,
    AppConfig,
    PROJECT_ROOT,
    DATA_DIR,
    IMAGES_DIR,
    MODELS_DIR,
)

__all__ = [
    "config",
    "get_config",
    "is_mock_mode",
    "is_cuda_available",
    "PowerNexusConfig",
    "QwenVLConfig",
    "QwenLLMConfig",
    "RAGConfig",
    "RLConfig",
    "AppConfig",
    "PROJECT_ROOT",
    "DATA_DIR",
    "IMAGES_DIR",
    "MODELS_DIR",
]
