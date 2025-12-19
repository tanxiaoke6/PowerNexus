# -*- coding: utf-8 -*-
"""
PowerNexus - Qwen2.5-VL 视觉语言模型封装

本模块提供基于 Qwen2.5-VL 的电力设备缺陷检测功能。
支持 CPU 模式备选。

模型: Qwen/Qwen2.5-VL-7B-Instruct (或 Qwen2-VL-7B-Instruct)

特性:
- Qwen2.5-VL 视觉语言模型
- 自动 GPU/CPU 检测
- 电力设备缺陷专用提示
- 结构化 JSON 输出

作者: PowerNexus Team
日期: 2025-12-18
"""

import base64
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 配置日志
logger = logging.getLogger(__name__)

# ============================================================================
# 依赖检测
# ============================================================================

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
QWEN_VL_UTILS_AVAILABLE = False
PIL_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI client 已导入")
except ImportError as e:
    import sys
    logger.warning(f"OpenAI 未安装: {e}")
    logger.warning(f"Python Executable: {sys.executable}")
    logger.warning(f"Sys Path: {sys.path}")
    OpenAI = None

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch 已导入 | CUDA: {torch.cuda.is_available()}")
except ImportError:
    logger.warning("PyTorch 未安装")
    torch = None

try:
    from transformers import (
        AutoProcessor,
        AutoModel,
        AutoModelForVision2Seq,
        Qwen2VLForConditionalGeneration,
    )
    # 尝试导入新版 API
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
        
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers 已导入")
    
    # 尝试导入 transformers  (用户示例中使用)
    try:
        from transformers  import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None
except ImportError:
    logger.warning("Transformers 未安装")
    AutoProcessor = None
    AutoModelForVision2Seq = None
    Qwen2VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
    logger.info("qwen_vl_utils 已导入")
except ImportError:
    logger.warning("qwen_vl_utils 未安装，将使用内置处理")
    process_vision_info = None


try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("Pillow 未安装")
    Image = None


# ============================================================================
# 配置数据类
# ============================================================================

# 导入全局配置
try:
    from config.settings import config as global_config, get_device, get_device_id
    SETTINGS_AVAILABLE = True
except ImportError:
    global_config = None
    get_device = lambda: "cpu"
    get_device_id = lambda: None
    SETTINGS_AVAILABLE = False
    logger.warning("无法导入 config.settings，使用默认配置")

# 获取默认值
def _get_vl_model_name():
    if SETTINGS_AVAILABLE and global_config:
        return global_config.qwen_vl.model_name
    return "/home/tanxk/xiaoke/Qwen2.5-VL-7B-Instruct"

def _get_vl_device():
    if SETTINGS_AVAILABLE:
        return get_device()
    return "cpu"

def _get_vl_device_id():
    if SETTINGS_AVAILABLE:
        return get_device_id()
    return None

def _get_vl_use_flash_attention():
    if SETTINGS_AVAILABLE and global_config:
        return global_config.qwen_vl.use_flash_attention
    return False

@dataclass
class QwenVLConfig:
    """
    Qwen-VL 模型配置
    
    Attributes:
        model_name: 模型名称/路径
        device: 运行设备 ('cuda', 'cpu', 'auto')
        use_flash_attention: 是否使用 Flash Attention
        max_new_tokens: 最大生成 token 数
        temperature: 生成温度
        top_p: Top-p 采样
        trust_remote_code: 是否信任远程代码
    """
    model_name: str = None
    device: str = None
    use_flash_attention: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    trust_remote_code: bool = True
    
    def __post_init__(self):
        # 从 settings 读取默认值
        if self.model_name is None:
            self.model_name = _get_vl_model_name()
        if self.device is None:
            self.device = _get_vl_device()


@dataclass
class DefectDetectionResult:
    """
    缺陷检测结果
    
    Attributes:
        defect_detected: 是否检测到缺陷
        defect_type: 缺陷类型
        severity: 严重程度 (low/medium/high/critical)
        confidence: 置信度 (0-1)
        description: 详细描述
        raw_response: 模型原始响应
        metadata: 附加元数据
    """
    defect_detected: bool = False
    defect_type: str = "none"
    severity: str = "none"
    confidence: float = 0.0
    description: str = ""
    raw_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "defect_detected": self.defect_detected,
            "defect_type": self.defect_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "description": self.description,
        }
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ============================================================================
# 提示模板
# ============================================================================

class QwenVLPrompts:
    """
    Qwen-VL 电力设备检测提示模板
    """
    
    # 系统提示
    SYSTEM_PROMPT = """你是一名专业的电力设备检测专家，专门分析无人机拍摄的电力设备图像。
你需要仔细检测图像中的电力设备是否存在以下缺陷：
- 绝缘子：裂纹、破损、污闪、闪络痕迹
- 变压器：漏油、锈蚀、套管破损
- 导线：断股、磨损、异物悬挂
- 金具：锈蚀、变形、松动
- 杆塔：倾斜、锈蚀、基础损坏

请仔细分析图像并给出专业判断。"""

    # 缺陷检测提示
    DEFECT_DETECTION_PROMPT = """请分析这张电力设备图像，检测是否存在缺陷。

要求：以 JSON 格式返回结果，包含以下字段：
- defect_type: 缺陷类型（如 "insulator_crack", "oil_leakage", "rust", "none" 等）
- severity: 严重程度（"low", "medium", "high", "critical", "none"）
- description: 详细描述（中文）

只返回 JSON，不要其他内容：
```json
{
    "defect_type": "",
    "severity": "",
    "description": ""
}
```"""

    # 详细分析提示
    DETAILED_ANALYSIS_PROMPT = """请对这张电力设备图像进行详细分析：

1. 识别图像中的设备类型
2. 检查设备外观状态
3. 识别任何可见的缺陷或异常
4. 评估问题的严重程度
5. 给出维护建议

以 JSON 格式返回：
```json
{
    "equipment_type": "设备类型",
    "defect_type": "缺陷类型",
    "severity": "严重程度",
    "description": "详细描述",
    "recommendation": "维护建议"
}
```"""


# ============================================================================
# Mock 模型（用于测试）
# ============================================================================

class MockQwenVL:
    """
    Mock Qwen-VL 模型
    
    用于开发测试，无需加载实际模型。
    """
    
    def __init__(self, config: QwenVLConfig):
        self.config = config
        logger.info("Mock Qwen-VL 模型已初始化")
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = None,
    ) -> str:
        """模拟图像分析"""
        filename = Path(image_path).stem.lower() if image_path else "unknown"
        
        # 根据文件名模拟结果
        if "crack" in filename or "裂纹" in filename:
            result = {
                "defect_type": "insulator_crack",
                "severity": "high",
                "description": "检测到绝缘子表面存在明显裂纹，长度约3cm，已贯穿釉面层，存在闪络风险，建议立即更换。"
            }
        elif "rust" in filename or "锈蚀" in filename:
            result = {
                "defect_type": "metal_rust",
                "severity": "medium",
                "description": "检测到金具表面锈蚀，锈蚀面积约占20%，尚未影响结构强度，建议在下次检修时进行防腐处理。"
            }
        elif "leak" in filename or "漏油" in filename:
            result = {
                "defect_type": "oil_leakage",
                "severity": "critical",
                "description": "检测到变压器底部明显油渍，疑似密封失效导致漏油，需立即停机检修，防止绝缘劣化。"
            }
        elif "normal" in filename or "正常" in filename:
            result = {
                "defect_type": "none",
                "severity": "none",
                "description": "设备外观正常，未检测到明显缺陷，建议按周期继续巡检。"
            }
        else:
            # 随机结果
            import random
            random.seed(hash(filename) % 1000)
            
            if random.random() > 0.5:
                defects = [
                    ("insulator_crack", "high", "绝缘子裂纹"),
                    ("rust", "medium", "金属锈蚀"),
                    ("deformation", "medium", "设备变形"),
                ]
                dt, sev, desc = random.choice(defects)
                result = {
                    "defect_type": dt,
                    "severity": sev,
                    "description": f"检测到可能存在{desc}，建议进一步确认。"
                }
            else:
                result = {
                    "defect_type": "none",
                    "severity": "none",
                    "description": "设备状态良好，未发现明显异常。"
                }
        
        return json.dumps(result, ensure_ascii=False)
    
    def close(self):
        """释放资源"""
        pass


# ============================================================================
# Qwen2.5-VL 模型封装
# ============================================================================

class Qwen2VLModel:
    """
    Qwen2.5-VL 视觉语言模型封装
    
    支持 Flash Attention。
    """
    
    def __init__(self, config: QwenVLConfig):
        """
        初始化 Qwen2.5-VL 模型
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model = None
        self.processor = None
        self._device = None
        
        # 检查依赖
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装 transformers 库")
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装 torch 库")
        if not PIL_AVAILABLE:
            raise ImportError("需要安装 Pillow 库")
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """
        加载 Qwen2.5-VL 模型 (Vanilla 模式 - 无优化)
        """
        logger.info(f"加载模型: {self.config.model_name}")
        
        # 获取目标设备串
        device_str = _get_vl_device()
        self._device = device_str
        
        # 处理多卡情况 (device_id: 5,6)
        if self._device == "auto":
            device_id = _get_vl_device_id()
            if device_id and (isinstance(device_id, (str, list))):
                device_id_str = str(device_id) if isinstance(device_id, str) else ",".join(map(str, device_id))
                if ',' in device_id_str or '-' in device_id_str:
                    # 注意：如果 torch 已经初始化，环境设置可能无效
                    os.environ["CUDA_VISIBLE_DEVICES"] = device_id_str
                    logger.info(f"检测到多卡配置: {device_id_str}, 设置 CUDA_VISIBLE_DEVICES")
        
        logger.info(f"目标设备策略: {self._device}")
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # 模型加载参数
        # 基础模型加载参数，仅包含必要的 trust_remote_code        # 模型加载参数
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # 尝试加载模型
        logger.info(f"开始加载 VL 模型 (Direct to {self._device})...")
        
        try:
            # 使用 Qwen2_5_VLForConditionalGeneration (支持 generate 方法)
            if Qwen2_5_VLForConditionalGeneration is not None:
                logger.info("使用 Qwen2_5_VLForConditionalGeneration")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                )
                # 将模型移动到目标设备（除非是 auto 模式）
                if self._device != "auto":
                    self.model.to(self._device)
            elif AutoModelForImageTextToText is not None:
                logger.info("使用 AutoModelForImageTextToText")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                )
            else:
                logger.info("使用 AutoModelForVision2Seq")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                )
                # 将模型移动到目标设备（除非是 auto 模式）
                if self._device != "auto":
                    self.model.to(self._device)
                
            logger.info("VL 模型加载完成")
        except Exception as e:
            logger.error(f"VL 模型加载失败: {e}")
            raise
    
    def _get_best_gpu(self) -> int:
        """选择显存最多的 GPU"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                best_gpu = 0
                max_free = 0
                
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        gpu_idx = int(parts[0].strip())
                        free_mb = int(parts[1].strip())
                        if free_mb > max_free:
                            max_free = free_mb
                            best_gpu = gpu_idx
                
                return best_gpu
        except Exception:
            pass
        return 0
    
    def _prepare_image(self, image_path: str) -> Image.Image:
        """
        准备图像输入
        
        Args:
            image_path: 图像路径
            
        Returns:
            PIL Image 对象
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        return image
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = None,
    ) -> str:
        """
        分析图像并返回结果
        
        Args:
            image_path: 图像路径
            prompt: 自定义提示（可选）
            
        Returns:
            模型响应文本
        """
        if prompt is None:
            prompt = QwenVLPrompts.DEFECT_DETECTION_PROMPT
        
        # 准备图像
        image = self._prepare_image(image_path)
        
        # 构建消息 (Qwen-VL 格式)
        messages = [
            {
                "role": "system",
                "content": QwenVLPrompts.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        # 处理视觉信息
        if QWEN_VL_UTILS_AVAILABLE and process_vision_info:
            # 使用官方工具处理
            image_inputs, video_inputs = process_vision_info(messages)
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            # 简化处理
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            ).to(self.model.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )
        
        # 解码响应
        response = self.processor.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        # 提取生成部分
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def close(self):
        """释放模型资源"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Qwen2.5-VL 模型资源已释放")


# ============================================================================
# Qwen2.5-VL API 模型封装
# ============================================================================

class Qwen2VLAPIModel:
    """
    Qwen2.5-VL API 模型封装
    
    使用 OpenAI 兼容的 API 进行图像分析。
    """
    
    def __init__(self, config: QwenVLConfig = None):
        """
        初始化 Qwen2.5-VL API 模型
        
        Args:
            config: 模型配置（未使用，从全局配置读取API设置）
        """
        self.config = config or QwenVLConfig()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("需要安装 openai 库: pip install openai")
        
        # 从全局配置读取 API 设置
        if global_config:
            api_base_url = global_config.qwen_vl.api_base_url
            api_key = global_config.qwen_vl.api_key
            self.model_name = global_config.qwen_vl.model_name
            self.max_tokens = global_config.qwen_vl.max_tokens
            self.temperature = global_config.qwen_vl.temperature
        else:
            raise ValueError("无法获取全局配置")
        
        if not api_base_url or not api_key:
            raise ValueError("请在 config.yaml 中配置 qwen_vl 的 api_base_url 和 api_key")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )
        
        logger.info(f"Qwen2.5-VL API 模型已初始化 | Model: {self.model_name}")
    
    def _encode_image(self, image_path: str) -> str:
        """将图像编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_mime_type(self, image_path: str) -> str:
        """获取图像 MIME 类型"""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/jpeg")
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = None,
    ) -> str:
        """
        分析图像并返回结果
        
        Args:
            image_path: 图像路径
            prompt: 自定义提示（可选）
            
        Returns:
            模型响应文本
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        if prompt is None:
            prompt = QwenVLPrompts.DEFECT_DETECTION_PROMPT
        
        # 编码图像
        image_base64 = self._encode_image(image_path)
        mime_type = self._get_image_mime_type(image_path)
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": QwenVLPrompts.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # 调用 API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"VL API 调用失败: {e}")
            raise
    
    def close(self):
        """释放资源"""
        pass


# ============================================================================
# 缺陷检测器主类
# ============================================================================

class DefectDetector:
    """
    电力设备缺陷检测器
    
    基于 Qwen2.5-VL 视觉语言模型。
    
    Example:
        >>> detector = DefectDetector(use_mock=True)
        >>> result = detector.detect("path/to/image.jpg")
        >>> print(result.defect_type)
        >>> print(result.to_json())
    """
    
    def __init__(
        self,
        config: Optional[QwenVLConfig] = None,
        use_mock: bool = False,
    ):
        """
        初始化缺陷检测器
        
        Args:
            config: 模型配置
            use_mock: 是否使用 Mock 模式
        """
        global OPENAI_AVAILABLE  # Fix UnboundLocalError
        
        self.config = config or QwenVLConfig()
        self._use_mock = use_mock
        self._use_api = False
        
        # 初始化模型
        if use_mock:
            self._model = MockQwenVL(self.config)
        else:
            # 优先使用 API 模式
            # 检查 API Key 是否存在 (非空字符串)
            use_api = False
            if global_config and global_config.qwen_vl.api_key:
                use_api = True
            
            # 强制尝试 API 模式
            if use_api:
                if not OPENAI_AVAILABLE:
                    logger.warning("虽然配置了 API Key，但 OpenAI 库不可用，尝试导入...")
                    try:
                        import openai
                        OPENAI_AVAILABLE = True
                    except ImportError:
                        logger.error("OpenAI 导入失败，无法使用 API 模式")
                        use_api = False
                
                if use_api:
                    try:
                        self._model = Qwen2VLAPIModel(self.config)
                        self._use_api = True
                        logger.info("使用 Qwen-VL API 模式")
                    except Exception as e:
                        logger.warning(f"加载 Qwen-VL API 失败: {e}，将尝试本地模型")
                        self._model = None
            else:
                self._model = None
            
            # 如果 API 模式不可用，尝试本地模型
            if self._model is None:
                can_load = (
                    TRANSFORMERS_AVAILABLE and 
                    TORCH_AVAILABLE and 
                    PIL_AVAILABLE
                )
                
                if can_load:
                    try:
                        self._model = Qwen2VLModel(self.config)
                        logger.info("使用 Qwen-VL 本地模型")
                    except Exception as e:
                        logger.warning(f"加载 Qwen-VL 本地模型失败: {e}，回退到 Mock 模式")
                        self._model = MockQwenVL(self.config)
                        self._use_mock = True
                else:
                    logger.warning("依赖不完整，使用 Mock 模式")
                    self._model = MockQwenVL(self.config)
                    self._use_mock = True
        
        logger.info(f"DefectDetector 初始化完成 | Mock: {self._use_mock} | API: {self._use_api}")
    
    def _parse_response(self, response: str) -> DefectDetectionResult:
        """
        解析模型响应为结构化结果
        
        Args:
            response: 模型原始响应
            
        Returns:
            DefectDetectionResult 对象
        """
        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                defect_type = str(data.get("defect_type", "unknown"))
                severity = str(data.get("severity", "unknown"))
                
                return DefectDetectionResult(
                    defect_detected=defect_type.lower() not in ["none", "无", "正常", ""],
                    defect_type=defect_type,
                    severity=severity,
                    confidence=0.9 if defect_type != "none" else 0.95,
                    description=str(data.get("description", "")),
                    raw_response=response,
                    metadata={"parse_method": "json"},
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON 解析失败: {e}")
        
        # 文本分析回退
        response_lower = response.lower()
        defect_keywords = ["缺陷", "裂纹", "锈蚀", "漏油", "破损", "异常", "defect", "crack", "rust", "leak"]
        
        defect_detected = any(kw in response_lower for kw in defect_keywords)
        
        return DefectDetectionResult(
            defect_detected=defect_detected,
            defect_type="unknown" if defect_detected else "none",
            severity="unknown" if defect_detected else "none",
            confidence=0.5,
            description=response[:300],
            raw_response=response,
            metadata={"parse_method": "text_fallback"},
        )
    
    def detect(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        detailed: bool = False,
    ) -> DefectDetectionResult:
        """
        检测图像中的设备缺陷
        
        Args:
            image_path: 图像文件路径
            prompt: 自定义提示
            detailed: 是否进行详细分析
            
        Returns:
            DefectDetectionResult 对象
        """
        image_path = str(image_path)
        
        # 检查文件
        if not self._use_mock and not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return DefectDetectionResult(
                defect_detected=False,
                defect_type="error",
                description=f"图像文件不存在: {image_path}",
                metadata={"error": "file_not_found"},
            )
        
        # 选择提示
        if prompt is None:
            prompt = (
                QwenVLPrompts.DETAILED_ANALYSIS_PROMPT 
                if detailed 
                else QwenVLPrompts.DEFECT_DETECTION_PROMPT
            )
        
        try:
            # 调用模型
            response = self._model.analyze_image(image_path, prompt)
            
            # 解析响应
            result = self._parse_response(response)
            result.metadata["image_path"] = image_path
            
            logger.info(
                f"检测完成 | 图像: {Path(image_path).name} | "
                f"缺陷: {result.defect_detected} | "
                f"类型: {result.defect_type} | "
                f"严重程度: {result.severity}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return DefectDetectionResult(
                defect_detected=False,
                defect_type="error",
                description=f"检测过程发生错误: {str(e)}",
                metadata={"error": str(e)},
            )
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
    ) -> str:
        """
        使用自定义提示分析图像
        
        Args:
            image_path: 图像路径
            prompt: 用户提示
            
        Returns:
            模型原始响应
        """
        return self._model.analyze_image(str(image_path), prompt)
    
    def batch_detect(
        self,
        image_paths: List[Union[str, Path]],
        detailed: bool = False,
    ) -> List[DefectDetectionResult]:
        """批量检测多张图像"""
        results = []
        for path in image_paths:
            result = self.detect(path, detailed=detailed)
            results.append(result)
        return results
    
    def close(self):
        """释放资源"""
        if hasattr(self, '_model') and self._model:
            self._model.close()
        logger.info("DefectDetector 已关闭")


# ============================================================================
# 便捷函数
# ============================================================================

def create_detector(
    model_name: str = None,
    use_mock: bool = False,
    **kwargs,
) -> DefectDetector:
    """
    创建缺陷检测器
    
    Args:
        model_name: 模型名称（默认从 settings.py 读取）
        use_mock: 是否使用 Mock 模式
        **kwargs: 其他配置参数
        
    Returns:
        DefectDetector 实例
    """
    config = QwenVLConfig(
        model_name=model_name,
        **kwargs,
    )
    return DefectDetector(config=config, use_mock=use_mock)


def detect_defects(
    image_path: Union[str, Path],
    use_mock: bool = True,
    detailed: bool = False,
) -> Dict[str, Any]:
    """
    快速检测单张图像
    
    Args:
        image_path: 图像路径
        use_mock: 是否使用 Mock 模式
        detailed: 是否详细分析
        
    Returns:
        检测结果字典
    """
    detector = create_detector(use_mock=use_mock)
    result = detector.detect(image_path, detailed=detailed)
    detector.close()
    return result.to_dict()


# ============================================================================
# 模块导出
# ============================================================================

# 兼容性导出 (与旧接口保持一致)
VisionModelConfig = QwenVLConfig
PerceptionPrompts = QwenVLPrompts
MockVisionBackend = MockQwenVL
VisionModelBackend = Qwen2VLModel

# 可用性标志
TRANSFORMERS_AVAILABLE = TRANSFORMERS_AVAILABLE
TORCH_AVAILABLE = TORCH_AVAILABLE
PIL_AVAILABLE = PIL_AVAILABLE
OPENAI_AVAILABLE = False  # 不再使用 OpenAI


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    print("=" * 60)
    print("PowerNexus - Qwen2.5-VL 缺陷检测测试")
    print("=" * 60)
    
    # 创建检测器 (Mock 模式)
    detector = create_detector(use_mock=True)
    
    # 测试图像
    test_images = [
        "insulator_crack_001.jpg",
        "transformer_oil_leak.jpg",
        "equipment_normal.jpg",
    ]
    
    print("\n测试检测结果:")
    print("-" * 40)
    
    for image_name in test_images:
        result = detector.detect(image_name)
        print(f"\n图像: {image_name}")
        print(f"  缺陷: {result.defect_detected}")
        print(f"  类型: {result.defect_type}")
        print(f"  严重程度: {result.severity}")
        print(f"  描述: {result.description[:60]}...")
    
    detector.close()
    print("\n测试完成!")
