# -*- coding: utf-8 -*-
"""
PowerNexus - 电网环境封装器

本模块提供 Grid2Op 电网仿真环境的 Gymnasium 兼容封装。
支持标准 Gym 接口 (reset, step)，用于强化学习训练。

特性:
- 观测空间：线路负载率、节点电压、发电机出力、拓扑状态
- 动作空间：拓扑切换（母线分裂）
- 支持 Mock 模式用于测试

作者: PowerNexus Team
日期: 2025-12-18
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

# ============================================================================
# 尝试导入 Grid2Op，如果失败则使用 Mock 模式
# ============================================================================

GRID2OP_AVAILABLE = False

try:
    import grid2op
    from grid2op.Action import PlayableAction
    from grid2op.Observation import CompleteObservation
    from grid2op.Environment import Environment as Grid2OpEnv
    from grid2op.Reward import BaseReward
    GRID2OP_AVAILABLE = True
    logger.info("Grid2Op 已成功导入")
except ImportError:
    logger.warning("Grid2Op 未安装，将使用 Mock 环境进行开发测试")
    grid2op = None

# ============================================================================
# 尝试导入 Gymnasium/Gym，如果失败则使用 Mock
# ============================================================================

GYM_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
    logger.info("Gymnasium 已成功导入")
except ImportError:
    try:
        # 兼容旧版本 gym
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
        logger.info("Gym (旧版) 已成功导入")
    except ImportError:
        logger.warning("Gymnasium/Gym 未安装，将使用 Mock 空间类")
        gym = None
        
        # Mock spaces 模块
        class MockSpaces:
            """Mock gymnasium.spaces 模块"""
            
            @staticmethod
            def Box(low, high, shape, dtype=np.float32):
                """Mock Box 空间"""
                return type('Box', (), {
                    'shape': shape,
                    'dtype': dtype,
                    'low': low,
                    'high': high,
                    'sample': lambda self: np.random.uniform(low, high, shape).astype(dtype),
                })()
            
            @staticmethod
            def Discrete(n):
                """Mock Discrete 空间"""
                return type('Discrete', (), {
                    'n': n,
                    'shape': (),
                    'sample': lambda self: np.random.randint(0, n),
                })()
        
        spaces = MockSpaces()


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class PowerGridEnvConfig:
    """
    电网环境配置类
    
    Attributes:
        env_name: Grid2Op 环境名称
        use_mock: 是否使用 Mock 环境
        observation_attributes: 需要包含在观测中的属性列表
        reward_weights: 奖励函数权重配置
        max_steps: 单个 episode 最大步数
        thermal_limit_threshold: 热稳定限制阈值 (0-1)
    """
    env_name: str = "l2rpn_case14_sandbox"
    use_mock: bool = False
    observation_attributes: List[str] = field(default_factory=lambda: [
        "rho",          # 线路负载率
        "gen_p",        # 发电机有功出力
        "gen_v",        # 发电机电压
        "load_p",       # 负荷有功
        "load_q",       # 负荷无功
        "topo_vect",    # 拓扑向量
    ])
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "operational_cost": -0.3,    # 运行成本惩罚
        "stability_margin": 0.4,     # 稳定裕度奖励
        "topology_action": -0.1,     # 拓扑动作惩罚
        "recovery_bonus": 0.5,       # 故障恢复奖励
        "overload_penalty": -1.0,    # 过载惩罚
    })
    max_steps: int = 2016  # 一周的时间步 (15分钟间隔)
    thermal_limit_threshold: float = 0.95


# ============================================================================
# Mock 环境（用于开发测试，无需安装 Grid2Op）
# ============================================================================

class MockGrid2OpEnv:
    """
    Mock Grid2Op 环境
    
    当 Grid2Op 未安装时提供基本的仿真功能，
    用于开发测试和代码验证。
    """
    
    def __init__(self, env_name: str = "l2rpn_case14_sandbox"):
        """
        初始化 Mock 环境
        
        Args:
            env_name: 环境名称（用于日志）
        """
        self.env_name = env_name
        self._current_step = 0
        self._max_steps = 2016
        
        # 模拟电网参数 (IEEE 14 节点系统)
        self.n_line = 20       # 线路数量
        self.n_gen = 5         # 发电机数量
        self.n_load = 11       # 负荷数量
        self.n_sub = 14        # 变电站数量
        self.dim_topo = 57     # 拓扑向量维度
        
        # 初始化状态
        self._init_state()
        
        logger.info(f"Mock 环境已初始化: {env_name}")
    
    def _init_state(self):
        """初始化环境状态"""
        # 随机初始化线路负载率 (0.3 - 0.8)
        self.rho = np.random.uniform(0.3, 0.8, self.n_line).astype(np.float32)
        
        # 发电机有功出力 (MW)
        self.gen_p = np.array([100.0, 50.0, 30.0, 40.0, 20.0], dtype=np.float32)
        
        # 发电机电压 (pu)
        self.gen_v = np.ones(self.n_gen, dtype=np.float32) * 1.0
        
        # 负荷有功/无功 (MW/MVar)
        self.load_p = np.random.uniform(10, 50, self.n_load).astype(np.float32)
        self.load_q = self.load_p * 0.3  # 假设功率因数 0.95
        
        # 拓扑向量 (1 = 母线1, 2 = 母线2)
        self.topo_vect = np.ones(self.dim_topo, dtype=np.int32)
        
        # 电网存活标志
        self._is_done = False
    
    def reset(self) -> "MockObservation":
        """
        重置环境
        
        Returns:
            初始观测
        """
        self._current_step = 0
        self._init_state()
        return MockObservation(self)
    
    def step(self, action: Any) -> Tuple["MockObservation", float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作（在 Mock 中忽略具体内容）
            
        Returns:
            (observation, reward, done, info) 元组
        """
        self._current_step += 1
        
        # 模拟负荷波动
        self.load_p *= np.random.uniform(0.98, 1.02, self.n_load)
        
        # 更新线路负载率
        self.rho = np.clip(
            self.rho + np.random.uniform(-0.05, 0.05, self.n_line),
            0.1, 1.5
        ).astype(np.float32)
        
        # 检查是否过载导致级联故障
        if np.any(self.rho > 1.0):
            self._is_done = True
        
        # 检查是否达到最大步数
        if self._current_step >= self._max_steps:
            self._is_done = True
        
        # 计算奖励
        reward = self._compute_reward()
        
        obs = MockObservation(self)
        info = {"step": self._current_step}
        
        return obs, reward, self._is_done, info
    
    def _compute_reward(self) -> float:
        """
        计算奖励
        
        奖励 = 存活奖励 - 过载惩罚 - 运行成本
        
        Returns:
            奖励值
        """
        if self._is_done and np.any(self.rho > 1.0):
            return -100.0  # 级联故障大惩罚
        
        # 存活奖励
        survival_reward = 1.0
        
        # 过载惩罚 (软约束)
        overload_penalty = np.sum(np.maximum(self.rho - 0.95, 0)) * 10
        
        # 运行成本 (发电成本)
        operation_cost = np.sum(self.gen_p) * 0.001
        
        return survival_reward - overload_penalty - operation_cost
    
    @property
    def action_space(self):
        """返回动作空间"""
        return MockActionSpace(self.dim_topo)
    
    def get_obs(self) -> "MockObservation":
        """获取当前观测"""
        return MockObservation(self)


class MockObservation:
    """Mock 观测类"""
    
    def __init__(self, env: MockGrid2OpEnv):
        """
        从环境状态构建观测
        
        Args:
            env: Mock 环境实例
        """
        self.rho = env.rho.copy()
        self.gen_p = env.gen_p.copy()
        self.gen_v = env.gen_v.copy()
        self.load_p = env.load_p.copy()
        self.load_q = env.load_q.copy()
        self.topo_vect = env.topo_vect.copy()
    
    def to_vect(self) -> np.ndarray:
        """
        将观测转换为向量
        
        Returns:
            观测向量
        """
        return np.concatenate([
            self.rho,
            self.gen_p,
            self.gen_v,
            self.load_p,
            self.load_q,
            self.topo_vect.astype(np.float32),
        ])


class MockActionSpace:
    """Mock 动作空间"""
    
    def __init__(self, dim_topo: int):
        """
        初始化动作空间
        
        Args:
            dim_topo: 拓扑向量维度
        """
        self.n = dim_topo + 1  # +1 为不操作动作
        self.dim_topo = dim_topo
    
    def sample(self) -> int:
        """随机采样动作"""
        return np.random.randint(0, self.n)
    
    @property
    def shape(self):
        return (self.n,)


# ============================================================================
# 主环境封装类
# ============================================================================

# 根据 gym 是否可用选择基类
if GYM_AVAILABLE:
    _BaseEnvClass = gym.Env
else:
    # 创建一个简单的基类
    class _BaseEnvClass:
        """Mock base environment class"""
        def __init__(self):
            pass
        def reset(self, seed=None):
            pass


class PowerGridEnv(_BaseEnvClass):
    """
    电网强化学习环境
    
    封装 Grid2Op 环境，提供标准 Gymnasium 接口。
    支持以下功能:
    - 标准 reset/step 接口
    - 自定义观测空间（负载率、电压、拓扑等）
    - 自定义动作空间（拓扑切换）
    - 自定义奖励函数
    
    Example:
        >>> env = PowerGridEnv()
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, done, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config: Optional[PowerGridEnvConfig] = None,
        env_name: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        初始化电网环境
        
        Args:
            config: 环境配置对象
            env_name: 环境名称 (会覆盖 config 中的设置)
            use_mock: 是否强制使用 Mock 环境
        """
        super().__init__()
        
        # 加载配置
        self.config = config or PowerGridEnvConfig()
        if env_name:
            self.config.env_name = env_name
        if use_mock:
            self.config.use_mock = True
        
        # 判断是否使用 Mock 环境
        self._use_mock = self.config.use_mock or not GRID2OP_AVAILABLE
        
        # 初始化底层环境
        self._init_backend_env()
        
        # 定义观测空间和动作空间
        self._setup_spaces()
        
        # 步数计数器
        self._current_step = 0
        
        logger.info(
            f"PowerGridEnv 初始化完成 | "
            f"环境: {self.config.env_name} | "
            f"Mock 模式: {self._use_mock}"
        )
    
    def _init_backend_env(self):
        """初始化后端环境 (Grid2Op 或 Mock)"""
        if self._use_mock:
            self._backend_env = MockGrid2OpEnv(self.config.env_name)
        else:
            # 使用 Grid2Op 创建环境
            self._backend_env = grid2op.make(
                self.config.env_name,
                test=True,  # 使用测试模式避免下载大数据集
            )
        
        # 缓存环境参数
        self.n_line = self._backend_env.n_line
        self.n_gen = self._backend_env.n_gen
        self.n_load = self._backend_env.n_load
        self.n_sub = getattr(self._backend_env, 'n_sub', 14)
        self.dim_topo = getattr(self._backend_env, 'dim_topo', 57)
    
    def _setup_spaces(self):
        """设置观测空间和动作空间"""
        # 计算观测向量维度
        obs_dim = (
            self.n_line +     # rho (线路负载率)
            self.n_gen +      # gen_p (发电机有功)
            self.n_gen +      # gen_v (发电机电压)
            self.n_load +     # load_p (负荷有功)
            self.n_load +     # load_q (负荷无功)
            self.dim_topo     # topo_vect (拓扑状态)
        )
        
        # 观测空间: 连续向量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # 动作空间: 离散拓扑动作
        # 动作 0 = 不操作
        # 动作 i (i > 0) = 切换拓扑元素 i-1 的母线连接
        n_actions = self.dim_topo + 1
        self.action_space = spaces.Discrete(n_actions)
        
        logger.debug(
            f"空间设置完成 | "
            f"观测维度: {obs_dim} | "
            f"动作数量: {n_actions}"
        )
    
    def _extract_observation(self, backend_obs) -> np.ndarray:
        """
        从后端观测提取特征向量
        
        Args:
            backend_obs: 后端环境观测对象
            
        Returns:
            标准化的观测向量
        """
        if self._use_mock:
            return backend_obs.to_vect()
        else:
            # 从 Grid2Op 观测中提取特征
            features = [
                backend_obs.rho,
                backend_obs.gen_p / 100.0,  # 标准化
                backend_obs.gen_v,
                backend_obs.load_p / 100.0,  # 标准化
                backend_obs.load_q / 100.0,  # 标准化
                backend_obs.topo_vect.astype(np.float32),
            ]
            return np.concatenate(features).astype(np.float32)
    
    def _convert_action(self, action: int):
        """
        将离散动作转换为后端环境动作
        
        Args:
            action: 离散动作索引
            
        Returns:
            后端环境动作对象
        """
        if self._use_mock:
            return action
        
        # Grid2Op 动作转换
        act = self._backend_env.action_space({})
        
        if action == 0:
            # 不操作
            pass
        else:
            # 拓扑切换动作
            # 切换指定元素的母线连接 (1 <-> 2)
            topo_idx = action - 1
            if topo_idx < self.dim_topo:
                # 获取当前拓扑状态
                current_obs = self._backend_env.get_obs()
                current_topo = current_obs.topo_vect[topo_idx]
                # 切换到另一条母线
                new_bus = 2 if current_topo == 1 else 1
                act.set_bus = [(topo_idx, new_bus)]
        
        return act
    
    def _compute_custom_reward(
        self,
        obs: np.ndarray,
        action: int,
        done: bool,
        backend_reward: float,
    ) -> float:
        """
        计算自定义奖励
        
        Args:
            obs: 观测向量
            action: 执行的动作
            done: 是否终止
            backend_reward: 后端环境奖励
            
        Returns:
            自定义奖励值
        """
        weights = self.config.reward_weights
        
        # 提取线路负载率
        rho = obs[:self.n_line]
        
        # 1. 过载惩罚
        overload = np.sum(np.maximum(rho - self.config.thermal_limit_threshold, 0))
        overload_penalty = overload * weights["overload_penalty"]
        
        # 2. 稳定裕度奖励 (负载率低于阈值)
        margin = np.mean(np.maximum(self.config.thermal_limit_threshold - rho, 0))
        stability_reward = margin * weights["stability_margin"]
        
        # 3. 拓扑动作惩罚 (鼓励少操作)
        action_penalty = (1 if action > 0 else 0) * weights["topology_action"]
        
        # 4. 运行成本 (发电量)
        gen_p_start = self.n_line
        gen_p_end = gen_p_start + self.n_gen
        gen_p = obs[gen_p_start:gen_p_end]
        operation_cost = np.sum(gen_p) * weights["operational_cost"] * 0.01
        
        # 5. 级联故障大惩罚
        if done and np.any(rho > 1.0):
            cascade_penalty = -100.0
        else:
            cascade_penalty = 0.0
        
        # 总奖励
        total_reward = (
            stability_reward +
            overload_penalty +
            action_penalty +
            operation_cost +
            cascade_penalty +
            1.0  # 存活奖励
        )
        
        return float(total_reward)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            (observation, info) 元组
        """
        super().reset(seed=seed)
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 重置后端环境
        backend_obs = self._backend_env.reset()
        
        # 提取观测
        obs = self._extract_observation(backend_obs)
        
        # 重置计数器
        self._current_step = 0
        
        info = {
            "step": self._current_step,
            "env_name": self.config.env_name,
            "use_mock": self._use_mock,
        }
        
        return obs, info
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 离散动作索引
            
        Returns:
            (observation, reward, terminated, truncated, info) 元组
        """
        # 转换动作
        backend_action = self._convert_action(action)
        
        # 执行动作
        backend_obs, backend_reward, done, info = self._backend_env.step(backend_action)
        
        # 提取观测
        obs = self._extract_observation(backend_obs)
        
        # 计算自定义奖励
        reward = self._compute_custom_reward(obs, action, done, backend_reward)
        
        # 更新步数
        self._current_step += 1
        
        # 判断是否截断 (达到最大步数但未终止)
        truncated = self._current_step >= self.config.max_steps and not done
        
        # 更新 info
        info.update({
            "step": self._current_step,
            "backend_reward": backend_reward,
            "custom_reward": reward,
            "max_rho": float(obs[:self.n_line].max()),
        })
        
        return obs, reward, done, truncated, info
    
    def render(self, mode: str = "human"):
        """
        渲染环境状态
        
        Args:
            mode: 渲染模式 ('human' 或 'ansi')
        """
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "human":
            print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """ANSI 文本渲染"""
        obs = self._backend_env.get_obs() if self._use_mock else self._backend_env.get_obs()
        rho = obs.rho if hasattr(obs, 'rho') else np.zeros(self.n_line)
        
        lines = [
            f"=== PowerGridEnv Step {self._current_step} ===",
            f"环境: {self.config.env_name}",
            f"最大负载率: {rho.max():.2%}",
            f"平均负载率: {rho.mean():.2%}",
            f"过载线路: {np.sum(rho > 0.95)}",
        ]
        return "\n".join(lines)
    
    def close(self):
        """关闭环境"""
        if hasattr(self._backend_env, 'close'):
            self._backend_env.close()
        logger.info("PowerGridEnv 已关闭")
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        获取观测空间信息
        
        Returns:
            观测空间结构信息
        """
        return {
            "total_dim": self.observation_space.shape[0],
            "components": {
                "rho": {"start": 0, "end": self.n_line, "desc": "线路负载率"},
                "gen_p": {"start": self.n_line, "end": self.n_line + self.n_gen, "desc": "发电机有功"},
                "gen_v": {"start": self.n_line + self.n_gen, "end": self.n_line + 2*self.n_gen, "desc": "发电机电压"},
                "load_p": {"start": self.n_line + 2*self.n_gen, "end": self.n_line + 2*self.n_gen + self.n_load, "desc": "负荷有功"},
                "load_q": {"start": self.n_line + 2*self.n_gen + self.n_load, "end": self.n_line + 2*self.n_gen + 2*self.n_load, "desc": "负荷无功"},
                "topo_vect": {"start": self.n_line + 2*self.n_gen + 2*self.n_load, "end": None, "desc": "拓扑状态"},
            }
        }


# ============================================================================
# 便捷函数
# ============================================================================

def make_power_grid_env(
    env_name: str = "l2rpn_case14_sandbox",
    use_mock: bool = False,
    **kwargs,
) -> PowerGridEnv:
    """
    创建电网环境的便捷函数
    
    Args:
        env_name: 环境名称
        use_mock: 是否使用 Mock 环境
        **kwargs: 传递给 PowerGridEnvConfig 的其他参数
        
    Returns:
        PowerGridEnv 实例
        
    Example:
        >>> env = make_power_grid_env("l2rpn_case14_sandbox")
        >>> obs, info = env.reset()
    """
    config = PowerGridEnvConfig(env_name=env_name, use_mock=use_mock, **kwargs)
    return PowerGridEnv(config=config)


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("PowerNexus - 电网环境测试")
    print("=" * 60)
    
    # 创建环境 (使用 Mock 模式测试)
    env = make_power_grid_env(use_mock=True)
    
    print(f"\n观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"观测信息: {env.get_observation_info()}")
    
    # 测试 reset
    obs, info = env.reset(seed=42)
    print(f"\n初始观测形状: {obs.shape}")
    print(f"初始信息: {info}")
    
    # 测试 step
    print("\n运行 10 步测试...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: action={action}, reward={reward:.3f}, max_rho={info['max_rho']:.2%}")
        
        if terminated or truncated:
            print(f"  Episode 结束: terminated={terminated}, truncated={truncated}")
            break
    
    print(f"\n总奖励: {total_reward:.3f}")
    
    # 渲染
    print("\n环境状态:")
    env.render()
    
    # 关闭环境
    env.close()
    
    print("\n测试完成!")
