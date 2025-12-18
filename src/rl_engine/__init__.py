# -*- coding: utf-8 -*-
"""
PowerNexus - 强化学习决策引擎

本模块负责基于 Grid2Op 环境进行电网拓扑优化决策。

主要组件:
- env_wrapper: PowerGridEnv 环境封装 (Grid2Op Gym 接口)
- agent: PPO_Agent 强化学习智能体
- reward_function: 奖励函数设计
- topology_actions: 拓扑动作空间
- training: 训练相关模块

使用示例:
    >>> from src.rl_engine import PowerGridEnv, PPO_Agent, make_power_grid_env
    >>> 
    >>> # 创建环境
    >>> env = make_power_grid_env(use_mock=True)
    >>> 
    >>> # 创建智能体
    >>> agent = PPO_Agent(env=env)
    >>> 
    >>> # 训练
    >>> agent.train_model(total_timesteps=10000)
    >>> 
    >>> # 预测
    >>> obs, info = env.reset()
    >>> action = agent.predict_action(obs)
"""

# 环境封装
from .env_wrapper import (
    PowerGridEnv,
    PowerGridEnvConfig,
    make_power_grid_env,
    MockGrid2OpEnv,
    GRID2OP_AVAILABLE,
)

# PPO 智能体
from .agent import (
    PPO_Agent,
    PPOAgentConfig,
    TrainingConfig,
    create_ppo_agent,
    train_and_save,
    GridTrainingCallback,
    SB3_AVAILABLE,
)

# 模块版本
__version__ = "0.1.0"

# 导出列表
__all__ = [
    # 环境
    "PowerGridEnv",
    "PowerGridEnvConfig",
    "make_power_grid_env",
    "MockGrid2OpEnv",
    "GRID2OP_AVAILABLE",
    # 智能体
    "PPO_Agent",
    "PPOAgentConfig",
    "TrainingConfig",
    "create_ppo_agent",
    "train_and_save",
    "GridTrainingCallback",
    "SB3_AVAILABLE",
]
