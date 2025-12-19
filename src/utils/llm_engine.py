# -*- coding: utf-8 -*-
"""
PowerNexus - Qwen LLM 引擎

本模块提供共享的 Qwen 文本语言模型接口。
用于 RAG 答案合成和 RL 决策解释。

模型: Qwen/Qwen2.5-7B-Instruct

特性:
- 4-bit 量化 (BitsAndBytes) 节省显存
- 自动 GPU/CPU 检测
- 流式生成支持
- 多种角色提示模板

作者: PowerNexus Team
日期: 2025-12-18
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

# 配置日志
logger = logging.getLogger(__name__)

# ============================================================================
# 依赖检测
# ============================================================================

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
BNB_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI client 已导入")
except ImportError as e:
    import sys
    logger.warning(f"OpenAI 未安装: {e}")
    logger.warning(f"Python Executable: {sys.executable}")
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
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer,
        pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers 已导入")
except ImportError:
    logger.warning("Transformers 未安装")
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    TextIteratorStreamer = None
    pipeline = None

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
    logger.info("BitsAndBytes 已导入")
except ImportError:
    logger.warning("BitsAndBytes 未安装")
    bnb = None


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
def _get_llm_model_name():
    if SETTINGS_AVAILABLE and global_config:
        return global_config.qwen_llm.model_name
    return "/home/tanxk/xiaoke/Qwen2.5-7B-Instruct"

def _get_llm_device():
    if SETTINGS_AVAILABLE:
        return get_device()
    return "cpu"

def _get_llm_device_id():
    if SETTINGS_AVAILABLE:
        return get_device_id()
    return None

@dataclass
class QwenLLMConfig:
    """
    Qwen LLM 配置
    
    Attributes:
        model_name: 模型名称/路径
        device: 运行设备 ('cuda', 'cpu', 'auto')
        max_new_tokens: 最大生成 token 数
        temperature: 生成温度
        top_p: Top-p 采样
        top_k: Top-k 采样
        repetition_penalty: 重复惩罚
        do_sample: 是否采样
    """
    model_name: str = None
    device: str = None
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    def __post_init__(self):
        # 从 settings 读取默认值
        if self.model_name is None:
            self.model_name = _get_llm_model_name()
        if self.device is None:
            self.device = _get_llm_device()


# ============================================================================
# 角色提示模板
# ============================================================================

class QwenPromptTemplates:
    """
    Qwen 提示模板集合
    """
    
    # 电力工程师角色
    POWER_ENGINEER_SYSTEM = """你是一名资深电力系统工程师，拥有20年电网运维经验。
你的职责是：
1. 分析电力设备状态和缺陷
2. 解读技术标准和规程
3. 提供专业的维护建议
4. 评估电网运行风险

请基于你的专业知识，给出准确、专业的回答。回答应简洁明了，重点突出。"""

    # RAG 问答模板
    RAG_QA_TEMPLATE = """根据以下参考资料回答问题。

## 参考资料
{context}

## 问题
{question}

## 要求
1. 基于参考资料回答，如引用请标注来源
2. 如资料中无相关信息，明确说明
3. 回答要专业、准确、简洁

请回答："""

    # RL 决策解释模板
    RL_INTERPRET_TEMPLATE = """你是电网调度专家。请解释以下 RL 决策的含义和影响。

## 当前电网状态
{grid_state}

## RL 引擎建议
动作: {action_name}
动作编号: {action_id}

## 任务
1. 解释这个动作的具体操作含义
2. 分析执行该动作的潜在影响
3. 评估风险和注意事项
4. 给出执行建议

请给出分析："""

    # 缺陷分析模板
    DEFECT_ANALYSIS_TEMPLATE = """请分析以下电力设备缺陷。

## 缺陷信息
类型: {defect_type}
严重程度: {severity}
位置: {location}
描述: {description}

## 任务
1. 分析缺陷的可能原因
2. 评估风险等级
3. 给出处理建议
4. 说明维护优先级

请给出分析："""


# ============================================================================
# Mock LLM
# ============================================================================

class MockQwenLLM:
    """
    Mock Qwen LLM
    
    用于开发测试，无需加载实际模型。
    """
    
    def __init__(self, config: QwenLLMConfig):
        self.config = config
        logger.info("Mock Qwen LLM 已初始化")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """生成模拟响应"""
        # 简单的模拟响应
        if "RL" in prompt or "动作" in prompt:
            return """基于当前电网状态分析：

1. **动作含义**: RL 引擎建议执行拓扑调整操作，将部分负荷从高负载线路转移至低负载线路。

2. **潜在影响**: 
   - 可有效降低过载线路的热稳定风险
   - 可能导致部分节点电压轻微波动
   - 预计负载率可降低 10-15%

3. **风险评估**: 
   - 操作风险等级: 低
   - 需确认目标线路具备承载余量

4. **执行建议**: 
   - 建议在确认备用线路容量后执行
   - 操作前通知相关值班人员
   - 执行后监控电压和潮流变化"""
        
        elif "缺陷" in prompt or "defect" in prompt.lower():
            return """缺陷分析结果：

1. **可能原因**: 
   - 长期运行导致的材料老化
   - 环境因素影响 (温度、湿度、污染)
   - 制造工艺缺陷

2. **风险等级**: 中等
   - 可能影响设备绝缘性能
   - 存在进一步恶化风险

3. **处理建议**:
   - 近期安排专业检测
   - 评估是否需要更换
   - 加强日常巡检频次

4. **优先级**: 建议在下个检修周期内处理"""
        
        else:
            return """根据参考资料分析：

1. 相关技术标准要求设备应满足规定的性能指标
2. 建议按照标准规程进行检测和维护
3. 如发现异常应及时上报处理

以上是基于技术标准的专业建议。"""
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """流式生成"""
        response = self.generate(prompt, system_prompt, **kwargs)
        for char in response:
            yield char
    
    def close(self):
        """释放资源"""
        pass


# ============================================================================
# Qwen LLM 封装
# ============================================================================

class QwenLLM:
    """
    Qwen2.5 文本语言模型封装
    
    支持 4-bit 量化，用于 RAG 答案合成和 RL 决策解释。
    """
    
    def __init__(self, config: QwenLLMConfig):
        """
        初始化 Qwen LLM
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._device = None
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装 transformers 库")
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装 torch 库")
        
        self.load_model()
    
    def load_model(self):
        """加载 Qwen 模型 (Vanilla 模式 - 无优化)"""
        logger.info(f"加载模型: {self.config.model_name}")
        
        # 获取目标设备串
        device_str = _get_llm_device()
        self._device = device_str
        
        # 处理多卡情况 (device_id: 5,6)
        is_multi_gpu = False
        if self._device == "auto":
            device_id = _get_llm_device_id()
            if device_id and (isinstance(device_id, (str, list))):
                device_id_str = str(device_id) if isinstance(device_id, str) else ",".join(map(str, device_id))
                if ',' in device_id_str or '-' in device_id_str:
                    is_multi_gpu = True
                    # 注意：如果 torch 已经初始化，环境设置可能无效
                    os.environ["CUDA_VISIBLE_DEVICES"] = device_id_str
                    logger.info(f"检测到多卡配置: {device_id_str}, 设置 CUDA_VISIBLE_DEVICES")
        
        logger.info(f"目标设备策略: {self._device}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # 模型加载参数
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # 加载模型
        logger.info(f"开始加载 LLM (Device Strategy: {self._device})...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs,
            )
            
            # 手动移动到设备 (如果不是 auto 模式)
            if self._device != "auto":
                self.model.to(self._device)
                
            logger.info("LLM 模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
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
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 覆盖生成参数
            
        Returns:
            生成的文本
        """
        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 编码
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
        }
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            
        Yields:
            生成的文本片段
        """
        if TextIteratorStreamer is None:
            # 回退到普通生成
            yield self.generate(prompt, system_prompt, **kwargs)
            return
        
        import threading
        
        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 创建流式输出器
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )
        
        # 生成参数
        gen_kwargs = {
            **inputs,
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "streamer": streamer,
            "do_sample": True,
        }
        
        # 后台线程生成
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=gen_kwargs,
        )
        thread.start()
        
        # 流式输出
        for text in streamer:
            yield text
        
        thread.join()
    
    def close(self):
        """释放资源"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Qwen LLM 资源已释放")


# ============================================================================
# Qwen LLM API 模型封装
# ============================================================================

class QwenLLMAPIModel:
    """
    Qwen LLM API 模型封装
    
    使用 OpenAI 兼容的 API 进行文本生成。
    """
    
    def __init__(self, config: QwenLLMConfig = None):
        """
        初始化 Qwen LLM API 模型
        
        Args:
            config: 模型配置（未使用，从全局配置读取API设置）
        """
        self.config = config or QwenLLMConfig()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("需要安装 openai 库: pip install openai")
        
        # 从全局配置读取 API 设置
        if global_config:
            api_base_url = global_config.qwen_llm.api_base_url
            api_key = global_config.qwen_llm.api_key
            self.model_name = global_config.qwen_llm.model_name
            self.max_tokens = global_config.qwen_llm.max_tokens
            self.temperature = global_config.qwen_llm.temperature
        else:
            raise ValueError("无法获取全局配置")
        
        if not api_base_url or not api_key:
            raise ValueError("请在 config.yaml 中配置 qwen_llm 的 api_base_url 和 api_key")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )
        
        logger.info(f"Qwen LLM API 模型已初始化 | Model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 覆盖生成参数
            
        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            
        Yields:
            生成的文本片段
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                stream=True,
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM API 流式调用失败: {e}")
            raise
    
    def close(self):
        """释放资源"""
        pass


# ============================================================================
# LLM 引擎管理
# ============================================================================

class LLMEngine:
    """
    LLM 引擎
    
    提供统一的 LLM 接口，支持 Mock 和真实模型。
    
    Example:
        >>> engine = LLMEngine(use_mock=True)
        >>> response = engine.chat("什么是电力系统？")
        >>> print(response)
    """
    
    # 单例实例
    _instance: Optional["LLMEngine"] = None
    
    def __init__(
        self,
        config: Optional[QwenLLMConfig] = None,
        use_mock: bool = False,
    ):
        """
        初始化 LLM 引擎
        
        Args:
            config: 模型配置
            use_mock: 是否使用 Mock 模式
        """
        self.config = config or QwenLLMConfig()
        self._use_mock = use_mock
        self._use_api = False
        
        # 初始化模型
        if use_mock:
            self._llm = MockQwenLLM(self.config)
        else:
            # 优先使用 API 模式
            if OPENAI_AVAILABLE and global_config and global_config.qwen_llm.api_key:
                try:
                    self._llm = QwenLLMAPIModel(self.config)
                    self._use_api = True
                    logger.info("使用 Qwen LLM API 模式")
                except Exception as e:
                    logger.warning(f"加载 Qwen LLM API 失败: {e}，尝试本地模型")
                    self._llm = None
            else:
                self._llm = None
            
            # 如果 API 模式不可用，尝试本地模型
            if self._llm is None:
                can_load = TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE
                
                if can_load:
                    try:
                        self._llm = QwenLLM(self.config)
                        logger.info("使用 Qwen LLM 本地模型")
                    except Exception as e:
                        logger.warning(f"加载 Qwen LLM 本地模型失败: {e}，回退到 Mock")
                        self._llm = MockQwenLLM(self.config)
                        self._use_mock = True
                else:
                    logger.warning("依赖不完整，使用 Mock 模式")
                    self._llm = MockQwenLLM(self.config)
                    self._use_mock = True
        
        logger.info(f"LLMEngine 初始化完成 | Mock: {self._use_mock} | API: {self._use_api}")
    
    @classmethod
    def get_instance(cls, use_mock: bool = False) -> "LLMEngine":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(use_mock=use_mock)
        return cls._instance
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        对话接口
        
        Args:
            message: 用户消息
            system_prompt: 系统提示
            
        Returns:
            模型响应
        """
        return self._llm.generate(message, system_prompt, **kwargs)
    
    def rag_synthesize(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        RAG 答案合成
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            
        Returns:
            合成的答案
        """
        prompt = QwenPromptTemplates.RAG_QA_TEMPLATE.format(
            context=context,
            question=question,
        )
        
        return self._llm.generate(
            prompt,
            system_prompt=QwenPromptTemplates.POWER_ENGINEER_SYSTEM,
        )
    
    def interpret_rl_action(
        self,
        action_id: int,
        action_name: str,
        grid_state: str,
    ) -> str:
        """
        解释 RL 动作
        
        Args:
            action_id: 动作 ID
            action_name: 动作名称
            grid_state: 电网状态描述
            
        Returns:
            动作解释
        """
        prompt = QwenPromptTemplates.RL_INTERPRET_TEMPLATE.format(
            action_id=action_id,
            action_name=action_name,
            grid_state=grid_state,
        )
        
        return self._llm.generate(
            prompt,
            system_prompt=QwenPromptTemplates.POWER_ENGINEER_SYSTEM,
        )
    
    def analyze_defect(
        self,
        defect_type: str,
        severity: str,
        location: str,
        description: str,
    ) -> str:
        """
        分析缺陷
        
        Args:
            defect_type: 缺陷类型
            severity: 严重程度
            location: 位置
            description: 描述
            
        Returns:
            缺陷分析
        """
        prompt = QwenPromptTemplates.DEFECT_ANALYSIS_TEMPLATE.format(
            defect_type=defect_type,
            severity=severity,
            location=location,
            description=description,
        )
        
        return self._llm.generate(
            prompt,
            system_prompt=QwenPromptTemplates.POWER_ENGINEER_SYSTEM,
        )
    
    def close(self):
        """释放资源"""
        if hasattr(self, '_llm') and self._llm:
            self._llm.close()
        LLMEngine._instance = None
        logger.info("LLMEngine 已关闭")


# ============================================================================
# 便捷函数
# ============================================================================

def create_llm_engine(
    model_name: str = None,
    use_mock: bool = False,
    **kwargs,
) -> LLMEngine:
    """
    创建 LLM 引擎
    
    Args:
        model_name: 模型名称（默认从 settings.py 读取）
        use_mock: 是否使用 Mock
        
    Returns:
        LLMEngine 实例
    """
    config = QwenLLMConfig(
        model_name=model_name,
        **kwargs,
    )
    return LLMEngine(config=config, use_mock=use_mock)


def get_llm_engine(use_mock: bool = True) -> LLMEngine:
    """获取全局 LLM 引擎实例"""
    return LLMEngine.get_instance(use_mock=use_mock)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    print("=" * 60)
    print("PowerNexus - Qwen LLM 引擎测试")
    print("=" * 60)
    
    # 创建引擎 (Mock 模式)
    engine = create_llm_engine(use_mock=True)
    
    # 测试 RAG 合成
    print("\n测试 RAG 答案合成:")
    print("-" * 40)
    
    context = "GB 50150-2016 规定绝缘电阻测量应在环境温度10-40℃进行。"
    question = "绝缘电阻测量的温度要求是什么？"
    
    answer = engine.rag_synthesize(question, context)
    print(f"问题: {question}")
    print(f"答案: {answer}")
    
    # 测试 RL 解释
    print("\n测试 RL 动作解释:")
    print("-" * 40)
    
    interpretation = engine.interpret_rl_action(
        action_id=3,
        action_name="拓扑调整 - 断开线路 L1",
        grid_state="L1 线路负载率 92%，L2 线路负载率 45%",
    )
    print(f"解释:\n{interpretation}")
    
    engine.close()
    print("\n测试完成!")
