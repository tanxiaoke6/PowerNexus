# -*- coding: utf-8 -*-
"""
PowerNexus - 智能体主工作流

本模块实现 "感知 -> 咨询 -> 决策 -> 执行" 的闭环智能体工作流。
整合多模态感知、RAG 知识库和强化学习决策引擎。

工作流程:
1. See (感知): 使用视觉模型检测设备缺陷
2. Think (咨询): 使用 RAG 检索相关技术标准
3. Decide (决策): 使用 RL 智能体评估电网状态和维护可行性
4. Act (执行): 生成维护报告和调度建议

作者: PowerNexus Team
日期: 2025-12-18
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# 导入子模块
# ============================================================================

try:
    from src.perception import DefectDetector, create_detector, DefectDetectionResult
    PERCEPTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"感知模块导入失败: {e}")
    PERCEPTION_AVAILABLE = False

try:
    from src.rag import KnowledgeBase, create_knowledge_base, init_knowledge_base_with_samples
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG 模块导入失败: {e}")
    RAG_AVAILABLE = False

try:
    from src.rl_engine import PPO_Agent, create_ppo_agent, make_power_grid_env
    RL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RL 模块导入失败: {e}")
    RL_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================================
# 工作流状态枚举
# ============================================================================

class WorkflowState(Enum):
    """工作流状态"""
    IDLE = "idle"                    # 空闲
    PERCEIVING = "perceiving"        # 感知中
    CONSULTING = "consulting"        # 咨询中
    DECIDING = "deciding"            # 决策中
    EXECUTING = "executing"          # 执行中
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 失败


class MaintenanceDecision(Enum):
    """维护决策类型"""
    IMMEDIATE = "immediate"          # 立即维护
    SCHEDULED = "scheduled"          # 计划维护
    DEFERRED = "deferred"            # 延期维护
    LOAD_SHIFT_REQUIRED = "load_shift_required"  # 需要负荷转移
    NO_ACTION = "no_action"          # 无需操作


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class GridTelemetry:
    """
    电网遥测数据
    
    Attributes:
        line_loads: 线路负载率 (0-1)
        voltage_levels: 节点电压 (pu)
        generator_output: 发电机出力 (MW)
        total_load: 总负荷 (MW)
        timestamp: 时间戳
    """
    line_loads: List[float] = field(default_factory=list)
    voltage_levels: List[float] = field(default_factory=list)
    generator_output: List[float] = field(default_factory=list)
    total_load: float = 0.0
    timestamp: str = ""
    
    @classmethod
    def create_mock(cls, load_level: str = "normal") -> "GridTelemetry":
        """
        创建模拟电网数据
        
        Args:
            load_level: 负荷水平 ('low', 'normal', 'high', 'critical')
        """
        import random
        
        load_configs = {
            "low": (0.2, 0.4),
            "normal": (0.4, 0.7),
            "high": (0.7, 0.9),
            "critical": (0.85, 1.0),
        }
        
        low, high = load_configs.get(load_level, (0.4, 0.7))
        
        return cls(
            line_loads=[random.uniform(low, high) for _ in range(20)],
            voltage_levels=[random.uniform(0.95, 1.05) for _ in range(14)],
            generator_output=[random.uniform(20, 100) for _ in range(5)],
            total_load=random.uniform(150, 250),
            timestamp=datetime.now().isoformat(),
        )
    
    @property
    def max_load(self) -> float:
        """最大线路负载率"""
        return max(self.line_loads) if self.line_loads else 0.0
    
    @property
    def avg_load(self) -> float:
        """平均线路负载率"""
        return sum(self.line_loads) / len(self.line_loads) if self.line_loads else 0.0
    
    @property
    def is_overloaded(self) -> bool:
        """是否过载"""
        return self.max_load > 0.95
    
    def to_observation(self) -> "np.ndarray":
        """转换为 RL 观测向量"""
        if np is None:
            return self.line_loads + self.voltage_levels + self.generator_output
        
        # 构建观测向量 (与 PowerGridEnv 兼容)
        obs = np.concatenate([
            np.array(self.line_loads),
            np.array(self.generator_output) / 100.0,  # 标准化
            np.array([1.0] * 5),  # 发电机电压 (假设)
            np.array([self.total_load / 10] * 11),  # 负荷
            np.zeros(11),  # 无功负荷
            np.ones(57),  # 拓扑向量
        ]).astype(np.float32)
        
        return obs


@dataclass
class InspectionInput:
    """
    巡检输入
    
    Attributes:
        image_path: 设备图像路径
        equipment_id: 设备编号
        location: 位置信息
        grid_telemetry: 电网遥测数据
    """
    image_path: str
    equipment_id: str = "unknown"
    location: str = "unknown"
    grid_telemetry: Optional[GridTelemetry] = None


@dataclass
class WorkflowResult:
    """
    工作流结果
    
    Attributes:
        state: 最终状态
        defect_result: 缺陷检测结果
        rag_context: RAG 检索上下文
        maintenance_decision: 维护决策
        rl_action: RL 动作
        report: 最终报告
        steps: 执行步骤记录
    """
    state: WorkflowState = WorkflowState.IDLE
    defect_result: Optional[Dict] = None
    rag_context: str = ""
    maintenance_decision: MaintenanceDecision = MaintenanceDecision.NO_ACTION
    rl_action: int = 0
    rl_safe_to_disconnect: bool = False
    report: str = ""
    steps: List[Dict] = field(default_factory=list)
    
    def add_step(self, name: str, result: Any, duration_ms: float = 0):
        """添加执行步骤"""
        self.steps.append({
            "name": name,
            "result": str(result)[:200],
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "state": self.state.value,
            "defect_result": self.defect_result,
            "rag_context_preview": self.rag_context[:300] + "..." if len(self.rag_context) > 300 else self.rag_context,
            "maintenance_decision": self.maintenance_decision.value,
            "rl_action": self.rl_action,
            "rl_safe_to_disconnect": self.rl_safe_to_disconnect,
            "report": self.report,
            "num_steps": len(self.steps),
        }
    
    def to_json(self) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ============================================================================
# 主智能体类
# ============================================================================

class MainAgent:
    """
    PowerNexus 主智能体
    
    整合感知、RAG、RL 模块，实现端到端的电网巡检与决策工作流。
    
    工作流程:
    1. See (感知) - 调用 DefectDetector 分析图像
    2. Think (咨询) - 调用 KnowledgeBase 检索维护标准
    3. Decide (决策) - 调用 PPO_Agent 评估维护可行性
    4. Act (执行) - 生成维护报告
    
    Example:
        >>> agent = MainAgent(use_mock=True)
        >>> input_data = InspectionInput(image_path="device.jpg")
        >>> result = agent.run(input_data)
        >>> print(result.report)
    """
    
    def __init__(
        self,
        use_mock: bool = False,
        perception_config: Optional[Dict] = None,
        rag_config: Optional[Dict] = None,
        rl_config: Optional[Dict] = None,
    ):
        """
        初始化主智能体
        
        Args:
            use_mock: 是否使用 Mock 模式
            perception_config: 感知模块配置
            rag_config: RAG 模块配置
            rl_config: RL 模块配置
        """
        self.use_mock = use_mock
        self._current_state = WorkflowState.IDLE
        
        # 初始化子模块
        self._init_perception(perception_config)
        self._init_rag(rag_config)
        self._init_rl(rl_config)
        
        logger.info(
            f"MainAgent 初始化完成 | "
            f"Mock 模式: {use_mock} | "
            f"模块状态: Perception={self._perception_ready}, "
            f"RAG={self._rag_ready}, RL={self._rl_ready}"
        )
    
    def _init_perception(self, config: Optional[Dict]):
        """初始化感知模块"""
        self._perception_ready = False
        
        if not PERCEPTION_AVAILABLE:
            logger.warning("感知模块不可用")
            self.detector = None
            return
        
        try:
            self.detector = create_detector(use_mock=self.use_mock)
            self._perception_ready = True
            logger.info("感知模块初始化成功")
        except Exception as e:
            logger.error(f"感知模块初始化失败: {e}")
            self.detector = None
    
    def _init_rag(self, config: Optional[Dict]):
        """初始化 RAG 模块"""
        self._rag_ready = False
        
        if not RAG_AVAILABLE:
            logger.warning("RAG 模块不可用")
            self.knowledge_base = None
            return
        
        try:
            # 初始化带示例数据的知识库
            self.knowledge_base = init_knowledge_base_with_samples(use_mock=self.use_mock)
            self._rag_ready = True
            logger.info("RAG 模块初始化成功")
        except Exception as e:
            logger.error(f"RAG 模块初始化失败: {e}")
            self.knowledge_base = None
    
    def _init_rl(self, config: Optional[Dict]):
        """初始化 RL 模块"""
        self._rl_ready = False
        
        if not RL_AVAILABLE:
            logger.warning("RL 模块不可用")
            self.rl_agent = None
            self.rl_env = None
            return
        
        try:
            self.rl_env = make_power_grid_env(use_mock=self.use_mock)
            self.rl_agent = create_ppo_agent(use_mock=self.use_mock)
            self._rl_ready = True
            logger.info("RL 模块初始化成功")
        except Exception as e:
            logger.error(f"RL 模块初始化失败: {e}")
            self.rl_agent = None
            self.rl_env = None
    
    def run(
        self,
        input_data: InspectionInput,
        skip_rl: bool = False,
    ) -> WorkflowResult:
        """
        运行完整工作流
        
        Args:
            input_data: 巡检输入数据
            skip_rl: 是否跳过 RL 决策步骤
            
        Returns:
            WorkflowResult 对象
        """
        result = WorkflowResult()
        
        try:
            # Step 1: 感知 (See)
            result = self._step_perceive(input_data, result)
            
            # 如果没有检测到缺陷，可以提前返回
            if result.defect_result and not result.defect_result.get("defect_detected"):
                result.maintenance_decision = MaintenanceDecision.NO_ACTION
                result.report = self._generate_report(result, input_data)
                result.state = WorkflowState.COMPLETED
                return result
            
            # Step 2: 咨询 (Think)
            result = self._step_consult(input_data, result)
            
            # Step 3: 决策 (Decide)
            if not skip_rl:
                result = self._step_decide(input_data, result)
            
            # Step 4: 执行 (Act)
            result = self._step_act(input_data, result)
            
            result.state = WorkflowState.COMPLETED
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            result.state = WorkflowState.FAILED
            result.report = f"工作流执行失败: {str(e)}"
            result.add_step("error", str(e))
        
        return result
    
    def _step_perceive(
        self,
        input_data: InspectionInput,
        result: WorkflowResult,
    ) -> WorkflowResult:
        """
        Step 1: 感知 - 检测设备缺陷
        """
        self._current_state = WorkflowState.PERCEIVING
        logger.info("Step 1: 感知 - 开始图像分析")
        
        if not self._perception_ready:
            # 使用模拟结果
            result.defect_result = {
                "defect_detected": True,
                "defect_type": "insulator_crack",
                "confidence": 0.92,
                "description": "检测到绝缘子裂纹 (模拟结果)",
            }
            result.add_step("perceive", "使用模拟感知结果")
        else:
            detection = self.detector.detect(input_data.image_path)
            result.defect_result = detection.to_dict()
            result.add_step("perceive", detection.to_dict())
        
        logger.info(
            f"感知完成 | 缺陷: {result.defect_result.get('defect_detected')} | "
            f"类型: {result.defect_result.get('defect_type')}"
        )
        
        return result
    
    def _step_consult(
        self,
        input_data: InspectionInput,
        result: WorkflowResult,
    ) -> WorkflowResult:
        """
        Step 2: 咨询 - 检索相关技术标准
        """
        self._current_state = WorkflowState.CONSULTING
        logger.info("Step 2: 咨询 - 检索技术标准")
        
        defect_type = result.defect_result.get("defect_type", "unknown")
        
        if not self._rag_ready:
            # 使用模拟上下文
            result.rag_context = f"""
### 参考 1 | 来源: DL/T 596-2018
绝缘子缺陷处理规程：
1. 发现裂纹应立即上报
2. 评估裂纹长度和深度
3. 必要时安排更换

### 参考 2 | 来源: GB 50150-2016  
绝缘子更换流程：
1. 确认线路停电
2. 拆除故障绝缘子
3. 安装新绝缘子
4. 进行耐压试验
"""
            result.add_step("consult", "使用模拟 RAG 结果")
        else:
            query = f"{defect_type} 维护处理规程 检修标准"
            rag_result = self.knowledge_base.query(query, top_k=3)
            result.rag_context = rag_result.formatted_context
            result.add_step("consult", f"检索到 {rag_result.num_results} 条相关标准")
        
        logger.info(f"咨询完成 | 上下文长度: {len(result.rag_context)} 字符")
        
        return result
    
    def _step_decide(
        self,
        input_data: InspectionInput,
        result: WorkflowResult,
    ) -> WorkflowResult:
        """
        Step 3: 决策 - RL 智能体评估维护可行性
        """
        self._current_state = WorkflowState.DECIDING
        logger.info("Step 3: 决策 - 评估电网状态")
        
        # 获取电网遥测数据
        telemetry = input_data.grid_telemetry or GridTelemetry.create_mock("normal")
        
        # 分析电网状态
        max_load = telemetry.max_load
        avg_load = telemetry.avg_load
        
        if not self._rl_ready:
            # 基于简单规则模拟决策
            if max_load > 0.9:
                result.rl_safe_to_disconnect = False
                result.rl_action = 0  # 不操作
                decision_reason = "电网负荷过高，不建议立即维护"
            elif max_load > 0.75:
                result.rl_safe_to_disconnect = False
                result.rl_action = 1  # 负荷转移
                decision_reason = "电网负荷较高，建议先进行负荷转移"
            else:
                result.rl_safe_to_disconnect = True
                result.rl_action = 0
                decision_reason = "电网负荷正常，可以安排维护"
            
            result.add_step("decide", f"模拟决策: {decision_reason}")
        else:
            # 使用 RL 智能体
            obs = telemetry.to_observation()
            
            # 确保观测维度正确
            if len(obs) < self.rl_env.observation_space.shape[0]:
                # 填充到正确维度
                obs = np.pad(obs, (0, self.rl_env.observation_space.shape[0] - len(obs)))
            elif len(obs) > self.rl_env.observation_space.shape[0]:
                obs = obs[:self.rl_env.observation_space.shape[0]]
            
            result.rl_action = self.rl_agent.predict_action(obs)
            
            # 解释动作
            if result.rl_action == 0:
                result.rl_safe_to_disconnect = max_load < 0.8
                decision_reason = "RL 建议保持当前拓扑"
            else:
                result.rl_safe_to_disconnect = False
                decision_reason = f"RL 建议执行拓扑调整 (动作 {result.rl_action})"
            
            result.add_step("decide", f"RL 决策: {decision_reason}")
        
        # 确定维护决策
        defect_confidence = result.defect_result.get("confidence", 0)
        
        if result.rl_safe_to_disconnect:
            if defect_confidence > 0.9:
                result.maintenance_decision = MaintenanceDecision.IMMEDIATE
            else:
                result.maintenance_decision = MaintenanceDecision.SCHEDULED
        else:
            result.maintenance_decision = MaintenanceDecision.LOAD_SHIFT_REQUIRED
        
        logger.info(
            f"决策完成 | 最大负载: {max_load:.1%} | "
            f"安全断开: {result.rl_safe_to_disconnect} | "
            f"决策: {result.maintenance_decision.value}"
        )
        
        return result
    
    def _step_act(
        self,
        input_data: InspectionInput,
        result: WorkflowResult,
    ) -> WorkflowResult:
        """
        Step 4: 执行 - 生成维护报告
        """
        self._current_state = WorkflowState.EXECUTING
        logger.info("Step 4: 执行 - 生成报告")
        
        result.report = self._generate_report(result, input_data)
        result.add_step("act", "报告生成完成")
        
        logger.info(f"执行完成 | 报告长度: {len(result.report)} 字符")
        
        return result
    
    def _generate_report(
        self,
        result: WorkflowResult,
        input_data: InspectionInput,
    ) -> str:
        """生成最终维护报告"""
        defect = result.defect_result or {}
        
        # 决策描述
        decision_texts = {
            MaintenanceDecision.IMMEDIATE: "🔴 立即安排维护",
            MaintenanceDecision.SCHEDULED: "🟡 安排计划性维护",
            MaintenanceDecision.DEFERRED: "🟢 可延期维护",
            MaintenanceDecision.LOAD_SHIFT_REQUIRED: "⚠️ 需先进行负荷转移后再维护",
            MaintenanceDecision.NO_ACTION: "✅ 设备正常，无需维护",
        }
        
        decision_text = decision_texts.get(result.maintenance_decision, "未知决策")
        
        report = f"""
{'='*60}
             PowerNexus 电网巡检报告
{'='*60}

📅 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📍 设备编号: {input_data.equipment_id}
📌 位置: {input_data.location}

{'─'*60}
                    缺陷检测结果
{'─'*60}

• 检测到缺陷: {'是' if defect.get('defect_detected') else '否'}
• 缺陷类型: {defect.get('defect_type', 'N/A')}
• 置信度: {defect.get('confidence', 0):.1%}
• 描述: {defect.get('description', 'N/A')}

{'─'*60}
                    电网状态评估  
{'─'*60}

• 可安全断开维护: {'是' if result.rl_safe_to_disconnect else '否'}
• RL 推荐动作: {'保持当前状态' if result.rl_action == 0 else f'拓扑调整 #{result.rl_action}'}

{'─'*60}
                    维护决策
{'─'*60}

{decision_text}

{'─'*60}
                    相关技术标准
{'─'*60}

{result.rag_context[:500] if result.rag_context else '未检索到相关标准'}
{'...(更多内容省略)' if len(result.rag_context) > 500 else ''}

{'='*60}
                    END OF REPORT
{'='*60}
"""
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "current_state": self._current_state.value,
            "perception_ready": self._perception_ready,
            "rag_ready": self._rag_ready,
            "rl_ready": self._rl_ready,
            "use_mock": self.use_mock,
        }
    
    def close(self):
        """释放资源"""
        if self.detector:
            self.detector.close()
        if self.rl_env:
            self.rl_env.close()
        logger.info("MainAgent 已关闭")


# ============================================================================
# 便捷函数
# ============================================================================

def create_main_agent(use_mock: bool = True) -> MainAgent:
    """
    创建主智能体的便捷函数
    
    Args:
        use_mock: 是否使用 Mock 模式
        
    Returns:
        MainAgent 实例
    """
    return MainAgent(use_mock=use_mock)


def run_inspection(
    image_path: str,
    equipment_id: str = "unknown",
    grid_load_level: str = "normal",
    use_mock: bool = True,
) -> WorkflowResult:
    """
    运行单次巡检的便捷函数
    
    Args:
        image_path: 图像路径
        equipment_id: 设备编号
        grid_load_level: 电网负荷水平
        use_mock: 是否使用 Mock 模式
        
    Returns:
        WorkflowResult 对象
    """
    agent = create_main_agent(use_mock=use_mock)
    
    input_data = InspectionInput(
        image_path=image_path,
        equipment_id=equipment_id,
        grid_telemetry=GridTelemetry.create_mock(grid_load_level),
    )
    
    result = agent.run(input_data)
    agent.close()
    
    return result


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("        PowerNexus 智能体工作流演示")
    print("=" * 60)
    
    # 创建智能体
    agent = create_main_agent(use_mock=True)
    print(f"\n智能体状态: {agent.get_status()}")
    
    # 模拟输入
    print("\n" + "-" * 60)
    print("场景: 无人机发现绝缘子裂纹，电网负荷正常")
    print("-" * 60)
    
    input_data = InspectionInput(
        image_path="./drone_images/insulator_crack_001.jpg",
        equipment_id="INS-2024-001",
        location="220kV 线路 L1 #35 塔",
        grid_telemetry=GridTelemetry.create_mock("normal"),
    )
    
    # 运行工作流
    result = agent.run(input_data)
    
    # 输出报告
    print(result.report)
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("结果摘要:")
    print("=" * 60)
    print(result.to_json())
    
    # 关闭智能体
    agent.close()
    
    print("\n演示完成!")
