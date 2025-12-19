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
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"

# 加载 YAML 配置
_YAML_CONFIG: Dict[str, Any] = {}
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            _YAML_CONFIG = yaml.safe_load(f) or {}
        print(f"已加载配置文件: {CONFIG_FILE}")
    except Exception as e:
        print(f"加载配置文件失败: {e}，将使用默认值")

def _get_yaml_val(section: str, key: str, default: Any) -> Any:
    """从 YAML 配置中获取值，如果不存在则返回默认值"""
    if section in _YAML_CONFIG and key in _YAML_CONFIG[section]:
        return _YAML_CONFIG[section][key]
    return default

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANUALS_DIR = DATA_DIR / "manuals"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================================
# CUDA 配置
# ============================================================================

@dataclass
class CUDAConfig:
    """CUDA 设备配置"""
    # 是否启用 CUDA
    enabled: bool = _get_yaml_val("cuda", "enabled", os.getenv("CUDA_ENABLED", "true").lower() == "true")
    # 指定 GPU 设备 ID (支持 int, str 如 "5,6", 或 list)
    device_id: Any = _get_yaml_val("cuda", "device_id", os.getenv("CUDA_DEVICE_ID"))
    # 是否自动选择最空闲的 GPU
    auto_select: bool = _get_yaml_val("cuda", "auto_select", os.getenv("CUDA_AUTO_SELECT", "true").lower() == "true")
    # 最小可用显存要求 (MB)
    min_free_memory_mb: int = _get_yaml_val("cuda", "min_free_memory_mb", int(os.getenv("CUDA_MIN_FREE_MEMORY", "8000")))
    
    def get_device(self) -> str:
        """获取设备字符串"""
        if not self.enabled:
            return "cpu"
        
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            
            if self.device_id is not None:
                # 检查多卡模式 (str 包含逗号/横杠，或者是 list)
                device_str = str(self.device_id)
                if ',' in device_str or '-' in device_str:
                    return "auto"
                if isinstance(self.device_id, list):
                    return "auto"
                return f"cuda:{self.device_id}"
            
            if self.auto_select:
                best_gpu = self._find_best_gpu()
                if best_gpu is not None:
                    return f"cuda:{best_gpu}"
            
            return "cuda:0"
        except ImportError:
            return "cpu"
    
    def get_device_id(self) -> Union[int, str, None]:
        """获取 GPU 设备 ID"""
        if not self.enabled:
            return None
        
        if self.device_id is not None:
            return self.device_id
            
        device = self.get_device()
        if device.startswith("cuda:"):
            try:
                return int(device.split(":")[1])
            except:
                return device.split(":")[1]
        return None
    
    def _find_best_gpu(self) -> Optional[int]:
        """找到显存最大的 GPU"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                best_gpu = None
                max_free = 0
                
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        gpu_idx = int(parts[0].strip())
                        free_mb = int(parts[1].strip())
                        if free_mb > max_free and free_mb >= self.min_free_memory_mb:
                            max_free = free_mb
                            best_gpu = gpu_idx
                
                return best_gpu
        except Exception:
            pass
        return None


# ============================================================================
# 模型配置
# ============================================================================

@dataclass
class QwenVLConfig:
    """视觉模型API配置"""
    api_base_url: str = _get_yaml_val("qwen_vl", "api_base_url", os.getenv("QWEN_VL_API_BASE_URL", ""))
    api_key: str = _get_yaml_val("qwen_vl", "api_key", os.getenv("QWEN_VL_API_KEY", ""))
    model_name: str = _get_yaml_val("qwen_vl", "model_name", os.getenv("QWEN_VL_MODEL", "qwen-vl-max"))
    max_tokens: int = _get_yaml_val("qwen_vl", "max_tokens", int(os.getenv("VL_MAX_TOKENS", "1024")))
    temperature: float = _get_yaml_val("qwen_vl", "temperature", float(os.getenv("VL_TEMPERATURE", "0.2")))

@dataclass
class QwenLLMConfig:
    """文本模型API配置"""
    api_base_url: str = _get_yaml_val("qwen_llm", "api_base_url", os.getenv("QWEN_LLM_API_BASE_URL", ""))
    api_key: str = _get_yaml_val("qwen_llm", "api_key", os.getenv("QWEN_LLM_API_KEY", ""))
    model_name: str = _get_yaml_val("qwen_llm", "model_name", os.getenv("QWEN_LLM_MODEL", "qwen-max"))
    max_tokens: int = _get_yaml_val("qwen_llm", "max_tokens", int(os.getenv("LLM_MAX_TOKENS", "2048")))
    temperature: float = _get_yaml_val("qwen_llm", "temperature", float(os.getenv("LLM_TEMPERATURE", "0.7")))
    use_for_text: bool = _get_yaml_val("qwen_llm", "use_for_text", os.getenv("LLM_USE_FOR_TEXT", "true").lower() == "true")

@dataclass
class RAGConfig:
    """RAG 知识库配置"""
    persist_directory: str = str(VECTOR_DB_DIR)
    collection_name: str = _get_yaml_val("rag", "collection_name", "power_grid_standards")
    embedding_api_base_url: str = _get_yaml_val("rag", "embedding_api_base_url", os.getenv("EMBEDDING_API_BASE_URL", ""))
    embedding_api_key: str = _get_yaml_val("rag", "embedding_api_key", os.getenv("EMBEDDING_API_KEY", ""))
    embedding_model: str = _get_yaml_val("rag", "embedding_model", os.getenv("EMBEDDING_MODEL", "text-embedding-v3"))
    embedding_batch_size: int = _get_yaml_val("rag", "embedding_batch_size", int(os.getenv("EMBEDDING_BATCH_SIZE", "64")))
    top_k: int = _get_yaml_val("rag", "top_k", 5)
    score_threshold: float = _get_yaml_val("rag", "score_threshold", 0.5)

@dataclass
class RLConfig:
    """强化学习配置"""
    env_name: str = _get_yaml_val("rl", "env_name", "l2rpn_case14_sandbox")
    learning_rate: float = _get_yaml_val("rl", "learning_rate", 3e-4)
    gamma: float = 0.99
    total_timesteps: int = _get_yaml_val("rl", "total_timesteps", 1_000_000)
    ppo_device: str = _get_yaml_val("rl", "ppo_device", os.getenv("PPO_DEVICE", "cpu"))

@dataclass
class AppConfig:
    """应用配置"""
    use_mock: bool = _get_yaml_val("app", "use_mock", os.getenv("USE_MOCK", "false").lower() == "true")
    log_level: str = _get_yaml_val("app", "log_level", os.getenv("LOG_LEVEL", "INFO"))
    port: int = _get_yaml_val("app", "port", int(os.getenv("PORT", "8501")))
    preload_models: bool = _get_yaml_val("app", "preload_models", os.getenv("PRELOAD_MODELS", "false").lower() == "true")


# ============================================================================
# 全局配置实例
# ============================================================================

@dataclass
class PowerNexusConfig:
    """PowerNexus 全局配置"""
    # CUDA 配置
    cuda: CUDAConfig = field(default_factory=CUDAConfig)
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
    
    def __post_init__(self):
        """初始化后设置设备"""
        device = self.cuda.get_device()
        self.qwen_vl.device = device
        self.qwen_llm.device = device


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


def get_device() -> str:
    """获取运行设备"""
    return config.cuda.get_device()


def get_device_id() -> Optional[int]:
    """获取 GPU 设备 ID"""
    return config.cuda.get_device_id()


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("PowerNexus 配置:")
    print(f"  项目根目录: {config.project_root}")
    print(f"  Mock 模式: {config.app.use_mock}")
    print(f"  CUDA 可用: {is_cuda_available()}")
    print(f"  运行设备: {get_device()}")
    print(f"  GPU ID: {get_device_id()}")
    print(f"  Qwen-VL 模型: {config.qwen_vl.model_name}")
    print(f"  Qwen-LLM 模型: {config.qwen_llm.model_name}")
    print(f"  Flash Attention: {config.qwen_vl.use_flash_attention}")
