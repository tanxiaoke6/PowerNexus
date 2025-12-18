# -*- coding: utf-8 -*-
"""
PowerNexus - 多模态感知模块

本模块负责处理无人机图像，检测电力设备缺陷。
使用 Qwen2.5-VL 视觉语言模型。

主要组件:
- vision_model: DefectDetector 缺陷检测器 (Qwen2.5-VL)
- image_loader: 图像加载与预处理
- equipment_classifier: 设备分类

使用示例:
    >>> from src.perception import DefectDetector, create_detector
    >>> 
    >>> # 创建检测器
    >>> detector = create_detector(use_mock=True)
    >>> 
    >>> # 检测缺陷
    >>> result = detector.detect("path/to/image.jpg")
    >>> print(result.defect_detected)
    >>> print(result.to_json())
"""

# 视觉模型
from .vision_model import (
    # 配置
    QwenVLConfig,
    VisionModelConfig,  # 别名
    # 核心类
    DefectDetector,
    DefectDetectionResult,
    QwenVLPrompts,
    PerceptionPrompts,  # 别名
    # 后端
    Qwen2VLModel,
    VisionModelBackend,  # 别名
    MockQwenVL,
    MockVisionBackend,  # 别名
    # 便捷函数
    create_detector,
    detect_defects,
    # 可用性标志
    TRANSFORMERS_AVAILABLE,
    TORCH_AVAILABLE,
    PIL_AVAILABLE,
    OPENAI_AVAILABLE,
)

# 模块版本
__version__ = "0.2.0"

# 导出列表
__all__ = [
    # 配置
    "QwenVLConfig",
    "VisionModelConfig",
    # 核心类
    "DefectDetector",
    "DefectDetectionResult",
    "QwenVLPrompts",
    "PerceptionPrompts",
    # 后端
    "Qwen2VLModel",
    "VisionModelBackend",
    "MockQwenVL",
    "MockVisionBackend",
    # 便捷函数
    "create_detector",
    "detect_defects",
    # 标志
    "TRANSFORMERS_AVAILABLE",
    "TORCH_AVAILABLE",
    "PIL_AVAILABLE",
    "OPENAI_AVAILABLE",
]
