# -*- coding: utf-8 -*-
"""
PowerNexus - PPO 强化学习智能体

本模块提供基于 Stable-Baselines3 的 PPO 智能体封装，
用于电网拓扑优化决策。支持 Qwen LLM 解释决策。

特性:
- 封装 SB3 PPO 算法
- 支持模型训练、保存、加载
- 提供推理接口用于实时决策
- Qwen LLM 解释 RL 决策
- 支持 Mock 模式用于测试

作者: PowerNexus Team
日期: 2025-12-18
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

# ============================================================================
# 尝试导入 Stable-Baselines3
# ============================================================================

SB3_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
    logger.info("Stable-Baselines3 已成功导入")
except ImportError:
    logger.warning("Stable-Baselines3 未安装，将使用 Mock 模式")
    PPO = None

# 导入 Qwen LLM 引擎
QWEN_LLM_AVAILABLE = False
try:
    from src.utils.llm_engine import LLMEngine, create_llm_engine
    QWEN_LLM_AVAILABLE = True
    logger.info("Qwen LLM 引擎已导入 (RL)")
except ImportError:
    logger.warning("Qwen LLM 引擎未导入，RL 决策解释功能不可用")
    LLMEngine = None

# 导入本地环境封装
from .env_wrapper import PowerGridEnv, make_power_grid_env, PowerGridEnvConfig


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class PPOAgentConfig:
    """
    PPO 智能体配置类
    
    Attributes:
        learning_rate: 学习率
        n_steps: 每次更新的步数
        batch_size: 批次大小
        n_epochs: 每次更新的训练轮数
        gamma: 折扣因子
        gae_lambda: GAE lambda 参数
        clip_range: PPO 裁剪范围
        ent_coef: 熵系数 (鼓励探索)
        vf_coef: 价值函数系数
        max_grad_norm: 梯度裁剪
        policy_kwargs: 策略网络参数
        tensorboard_log: TensorBoard 日志目录
        verbose: 日志详细程度
    """
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Optional[Dict] = None
    tensorboard_log: Optional[str] = None
    verbose: int = 1


@dataclass
class TrainingConfig:
    """
    训练配置类
    
    Attributes:
        total_timesteps: 总训练步数
        eval_freq: 评估频率
        n_eval_episodes: 每次评估的 episode 数
        save_freq: 模型保存频率
        save_path: 模型保存路径
        best_model_save_path: 最佳模型保存路径
        log_path: 日志路径
    """
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    save_freq: int = 50_000
    save_path: str = "./models/rl/checkpoints"
    best_model_save_path: str = "./models/rl/best_model"
    log_path: str = "./logs/rl_training"


# ============================================================================
# Mock PPO 智能体（用于开发测试）
# ============================================================================

class MockPPO:
    """
    Mock PPO 智能体
    
    当 Stable-Baselines3 未安装时提供基本功能，
    用于开发测试和代码验证。
    """
    
    def __init__(
        self,
        policy: str,
        env: Any,
        learning_rate: float = 3e-4,
        **kwargs,
    ):
        """
        初始化 Mock PPO
        
        Args:
            policy: 策略类型 (忽略，仅用于兼容)
            env: 训练环境
            learning_rate: 学习率
            **kwargs: 其他参数 (忽略)
        """
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self._num_timesteps = 0
        
        # 获取动作空间大小
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space
        else:
            # 假设环境有 n_actions 属性
            self.action_space = type('ActionSpace', (), {'n': 58})()
        
        logger.info("Mock PPO 已初始化")
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        **kwargs,
    ) -> "MockPPO":
        """
        模拟训练过程
        
        Args:
            total_timesteps: 训练步数
            callback: 回调函数 (忽略)
            **kwargs: 其他参数
            
        Returns:
            self
        """
        logger.info(f"Mock PPO 开始模拟训练: {total_timesteps} 步")
        
        # 模拟训练进度
        for step in range(0, total_timesteps, 10000):
            self._num_timesteps = step
            logger.debug(f"  训练进度: {step}/{total_timesteps}")
        
        self._num_timesteps = total_timesteps
        logger.info("Mock PPO 训练完成")
        
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        预测动作
        
        Args:
            observation: 观测向量
            deterministic: 是否确定性策略
            
        Returns:
            (action, state) 元组
        """
        # 基于观测的简单启发式策略
        # 如果负载率高，尝试拓扑调整；否则不操作
        rho_mean = np.mean(observation[:20])  # 假设前20维是负载率
        
        if rho_mean > 0.8:
            # 高负载时随机选择拓扑动作
            action = np.array([np.random.randint(1, self.action_space.n)])
        else:
            # 低负载时不操作
            action = np.array([0])
        
        return action, None
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 模拟保存
        with open(f"{path}.mock", 'w') as f:
            f.write(f"MockPPO model saved at timestep {self._num_timesteps}")
        
        logger.info(f"Mock PPO 模型已保存: {path}")
    
    @classmethod
    def load(cls, path: str, env: Optional[Any] = None) -> "MockPPO":
        """
        加载模型
        
        Args:
            path: 模型路径
            env: 环境 (可选)
            
        Returns:
            MockPPO 实例
        """
        logger.info(f"Mock PPO 模型已加载: {path}")
        return cls("MlpPolicy", env)
    
    @property
    def num_timesteps(self) -> int:
        """返回训练步数"""
        return self._num_timesteps


# ============================================================================
# 主智能体类
# ============================================================================

class PPO_Agent:
    """
    PPO 强化学习智能体
    
    封装 Stable-Baselines3 的 PPO 算法，提供：
    - 模型初始化与配置
    - 训练接口
    - 推理接口
    - 模型保存与加载
    
    Example:
        >>> # 创建智能体
        >>> agent = PPO_Agent()
        >>> 
        >>> # 训练模型
        >>> agent.train_model(total_timesteps=100000)
        >>> 
        >>> # 预测动作
        >>> action = agent.predict_action(observation)
        >>> 
        >>> # 保存模型
        >>> agent.save("models/ppo_agent.zip")
    """
    
    def __init__(
        self,
        env: Optional[PowerGridEnv] = None,
        env_config: Optional[PowerGridEnvConfig] = None,
        agent_config: Optional[PPOAgentConfig] = None,
        model_path: Optional[str] = None,
        use_mock: bool = False,
        enable_llm: bool = True,
    ):
        """
        初始化 PPO 智能体
        
        Args:
            env: 训练环境 (如果为 None，会自动创建)
            env_config: 环境配置
            agent_config: 智能体配置
            model_path: 预训练模型路径 (如果提供，会加载模型)
            use_mock: 是否强制使用 Mock 模式
            enable_llm: 是否启用 Qwen LLM 决策解释
        """
        self.agent_config = agent_config or PPOAgentConfig()
        self.env_config = env_config or PowerGridEnvConfig()
        
        # 判断是否使用 Mock
        self._use_mock = use_mock or not SB3_AVAILABLE
        self._llm_enabled = enable_llm and QWEN_LLM_AVAILABLE
        
        # 初始化环境
        if env is not None:
            self.env = env
        else:
            self.env = make_power_grid_env(
                env_name=self.env_config.env_name,
                use_mock=self._use_mock or self.env_config.use_mock,
            )
        
        # 初始化模型
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._init_model()
        
        # 初始化 Qwen LLM 引擎
        self._llm_engine = None
        if self._llm_enabled:
            try:
                self._llm_engine = create_llm_engine(use_mock=use_mock)
                logger.info("Qwen LLM 引擎已初始化 (RL 决策解释)")
            except Exception as e:
                logger.warning(f"Qwen LLM 初始化失败: {e}")
                self._llm_enabled = False
        
        # 动作名称映射
        self._action_names = self._build_action_names()
        
        logger.info(
            f"PPO_Agent 初始化完成 | "
            f"Mock: {self._use_mock} | "
            f"LLM: {self._llm_enabled} | "
            f"环境: {self.env_config.env_name}"
        )
    
    def _build_action_names(self) -> Dict[int, str]:
        """构建动作名称映射"""
        names = {0: "不操作 (No-Op)"}
        n_actions = getattr(self.env.action_space, 'n', 58)
        
        for i in range(1, n_actions):
            if i <= 20:
                names[i] = f"拓扑调整 - 切换线路 {i} 母线连接"
            elif i <= 40:
                names[i] = f"拓扑调整 - 切换发电机 {i-20} 母线"
            else:
                names[i] = f"拓扑调整 - 切换负荷 {i-40} 母线"
        
        return names
    
    def _init_model(self):
        """初始化 PPO 模型"""
        # 选择 PPO 实现
        PPOClass = MockPPO if self._use_mock else PPO
        
        # 策略网络配置
        policy_kwargs = self.agent_config.policy_kwargs or {
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        }
        
        # 创建模型
        self.model = PPOClass(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.agent_config.learning_rate,
            n_steps=self.agent_config.n_steps,
            batch_size=self.agent_config.batch_size,
            n_epochs=self.agent_config.n_epochs,
            gamma=self.agent_config.gamma,
            gae_lambda=self.agent_config.gae_lambda,
            clip_range=self.agent_config.clip_range,
            ent_coef=self.agent_config.ent_coef,
            vf_coef=self.agent_config.vf_coef,
            max_grad_norm=self.agent_config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.agent_config.tensorboard_log,
            verbose=self.agent_config.verbose,
        )
        
        logger.debug("PPO 模型已初始化")
    
    def _load_model(self, model_path: str):
        """
        加载预训练模型
        
        Args:
            model_path: 模型路径
        """
        PPOClass = MockPPO if self._use_mock else PPO
        
        try:
            self.model = PPOClass.load(model_path, env=self.env)
            logger.info(f"模型已从 {model_path} 加载")
        except Exception as e:
            logger.warning(f"加载模型失败: {e}，将创建新模型")
            self._init_model()
    
    def train_model(
        self,
        total_timesteps: Optional[int] = None,
        training_config: Optional[TrainingConfig] = None,
        callback: Optional[Any] = None,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        训练 PPO 模型
        
        Args:
            total_timesteps: 训练步数 (覆盖 training_config)
            training_config: 训练配置
            callback: 自定义回调
            progress_bar: 是否显示进度条
            
        Returns:
            训练结果信息
        """
        config = training_config or TrainingConfig()
        timesteps = total_timesteps or config.total_timesteps
        
        logger.info(f"开始训练 | 总步数: {timesteps}")
        
        # 创建回调列表
        callbacks = []
        
        if SB3_AVAILABLE and not self._use_mock:
            # 检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=config.save_freq,
                save_path=config.save_path,
                name_prefix="ppo_grid",
            )
            callbacks.append(checkpoint_callback)
            
            # 评估回调
            eval_env = make_power_grid_env(
                env_name=self.env_config.env_name,
                use_mock=self._use_mock or self.env_config.use_mock,
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=config.best_model_save_path,
                log_path=config.log_path,
                eval_freq=config.eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)
        
        # 添加自定义回调
        if callback:
            callbacks.append(callback)
        
        # 组合回调
        combined_callback = CallbackList(callbacks) if callbacks and SB3_AVAILABLE else None
        
        # 训练
        try:
            self.model.learn(
                total_timesteps=timesteps,
                callback=combined_callback,
                progress_bar=progress_bar if SB3_AVAILABLE else False,
            )
            
            training_result = {
                "status": "success",
                "total_timesteps": timesteps,
                "model_timesteps": self.model.num_timesteps,
            }
            
            logger.info(f"训练完成 | 总步数: {self.model.num_timesteps}")
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            training_result = {
                "status": "failed",
                "error": str(e),
            }
        
        return training_result
    
    def predict_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        预测最优动作
        
        Args:
            observation: 观测向量 (来自 PowerGridEnv)
            deterministic: 是否使用确定性策略
            
        Returns:
            最优动作索引
        """
        # 确保观测是正确的形状
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        # 预测
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic,
        )
        
        # 提取单个动作
        if isinstance(action, np.ndarray):
            action = int(action[0]) if action.ndim > 0 else int(action)
        
        return action
    
    def get_action_name(self, action: int) -> str:
        """获取动作名称"""
        return self._action_names.get(action, f"未知动作 {action}")
    
    def interpret_action(
        self,
        action: int,
        grid_state: str,
    ) -> str:
        """
        使用 Qwen LLM 解释 RL 动作
        
        Args:
            action: 动作 ID
            grid_state: 电网状态描述
            
        Returns:
            动作解释文本
        """
        action_name = self.get_action_name(action)
        
        if self._llm_enabled and self._llm_engine:
            try:
                interpretation = self._llm_engine.interpret_rl_action(
                    action_id=action,
                    action_name=action_name,
                    grid_state=grid_state,
                )
                logger.info(f"RL 动作解释完成 | 动作: {action}")
                return interpretation
            except Exception as e:
                logger.warning(f"动作解释失败: {e}")
                return f"[解释失败: {e}]"
        else:
            # 简单的规则解释
            if action == 0:
                return f"RL 引擎建议保持当前电网拓扑不变。当前状态: {grid_state}"
            else:
                return (
                    f"RL 引擎建议执行: {action_name}\n"
                    f"当前电网状态: {grid_state}\n"
                    f"[注: LLM 未启用，无法提供详细解释]"
                )
    
    def predict_and_interpret(
        self,
        observation: np.ndarray,
        grid_state_description: str,
        deterministic: bool = True,
    ) -> Tuple[int, str]:
        """
        预测动作并解释
        
        Args:
            observation: 观测向量
            grid_state_description: 电网状态描述
            deterministic: 是否确定性策略
            
        Returns:
            (动作, 解释) 元组
        """
        action = self.predict_action(observation, deterministic)
        interpretation = self.interpret_action(action, grid_state_description)
        return action, interpretation
    
    def evaluate(
        self,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            n_eval_episodes: 评估 episode 数量
            deterministic: 是否使用确定性策略
            
        Returns:
            评估指标字典
        """
        logger.info(f"开始评估 | Episodes: {n_eval_episodes}")
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_eval_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            step_count = 0
            
            while not (done or truncated):
                action = self.predict_action(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                step_count += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            logger.debug(f"  Episode {ep+1}: reward={total_reward:.2f}, steps={step_count}")
        
        # 计算统计指标
        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "n_episodes": n_eval_episodes,
        }
        
        logger.info(
            f"评估完成 | "
            f"平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        
        return results
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        # 确保目录存在
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        logger.info(f"模型已保存: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        self._load_model(path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_type": "PPO",
            "use_mock": self._use_mock,
            "num_timesteps": getattr(self.model, 'num_timesteps', 0),
            "observation_space": str(self.env.observation_space),
            "action_space": str(self.env.action_space),
            "learning_rate": self.agent_config.learning_rate,
            "gamma": self.agent_config.gamma,
        }


# ============================================================================
# 训练回调类
# ============================================================================

class GridTrainingCallback:
    """
    电网训练回调
    
    用于在训练过程中记录额外信息。
    """
    
    def __init__(self, log_interval: int = 1000):
        """
        初始化回调
        
        Args:
            log_interval: 日志记录间隔
        """
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0
        self._current_length = 0
    
    def __call__(self, locals_: Dict, globals_: Dict) -> bool:
        """
        回调函数
        
        Args:
            locals_: 本地变量
            globals_: 全局变量
            
        Returns:
            是否继续训练
        """
        # 记录信息
        if 'rewards' in locals_:
            self._current_reward += locals_['rewards']
        
        self._current_length += 1
        
        # 检查是否 episode 结束
        if 'dones' in locals_ and locals_['dones']:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0
            self._current_length = 0
        
        return True
    
    def get_statistics(self) -> Dict[str, float]:
        """获取训练统计"""
        if not self.episode_rewards:
            return {}
        
        return {
            "mean_reward": np.mean(self.episode_rewards[-100:]),
            "mean_length": np.mean(self.episode_lengths[-100:]),
            "total_episodes": len(self.episode_rewards),
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_ppo_agent(
    env_name: str = "l2rpn_case14_sandbox",
    model_path: Optional[str] = None,
    use_mock: bool = False,
    **agent_kwargs,
) -> PPO_Agent:
    """
    创建 PPO 智能体的便捷函数
    
    Args:
        env_name: 环境名称
        model_path: 预训练模型路径
        use_mock: 是否使用 Mock 模式
        **agent_kwargs: 传递给 PPOAgentConfig 的参数
        
    Returns:
        PPO_Agent 实例
    """
    env_config = PowerGridEnvConfig(env_name=env_name, use_mock=use_mock)
    agent_config = PPOAgentConfig(**agent_kwargs)
    
    return PPO_Agent(
        env_config=env_config,
        agent_config=agent_config,
        model_path=model_path,
        use_mock=use_mock,
    )


def train_and_save(
    save_path: str = "./models/rl/ppo_grid.zip",
    total_timesteps: int = 100_000,
    env_name: str = "l2rpn_case14_sandbox",
    use_mock: bool = False,
) -> PPO_Agent:
    """
    训练并保存模型的便捷函数
    
    Args:
        save_path: 模型保存路径
        total_timesteps: 训练步数
        env_name: 环境名称
        use_mock: 是否使用 Mock 模式
        
    Returns:
        训练好的 PPO_Agent
    """
    agent = create_ppo_agent(env_name=env_name, use_mock=use_mock)
    agent.train_model(total_timesteps=total_timesteps)
    agent.save(save_path)
    return agent


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    print("=" * 60)
    print("PowerNexus - PPO 智能体测试")
    print("=" * 60)
    
    # 创建智能体 (Mock 模式)
    agent = create_ppo_agent(use_mock=True)
    
    print(f"\n模型信息: {agent.get_model_info()}")
    
    # 短期训练测试
    print("\n开始训练测试 (1000 步)...")
    result = agent.train_model(total_timesteps=1000)
    print(f"训练结果: {result}")
    
    # 评估测试
    print("\n开始评估测试...")
    eval_result = agent.evaluate(n_eval_episodes=3)
    print(f"评估结果: {eval_result}")
    
    # 动作预测测试
    print("\n动作预测测试...")
    obs, info = agent.env.reset()
    for i in range(5):
        action = agent.predict_action(obs)
        print(f"  Step {i+1}: 预测动作 = {action}")
        obs, reward, done, truncated, info = agent.env.step(action)
        if done or truncated:
            break
    
    # 保存测试
    test_save_path = "./models/rl/test_ppo.zip"
    print(f"\n保存模型到: {test_save_path}")
    agent.save(test_save_path)
    
    print("\n测试完成!")
