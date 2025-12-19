# -*- coding: utf-8 -*-
"""
PowerNexus - PPO 训练脚本

独立的 PPO 训练脚本，用于训练电网拓扑优化智能体。

使用方法:
    # 快速测试 (Mock 模式)
    python tools/train_ppo.py --timesteps 1000 --use-mock
    
    # 正式训练 (需要 Grid2Op)
    python tools/train_ppo.py --timesteps 100000 --save-path models/rl/ppo_grid.zip
    
    # 继续训练已有模型
    python tools/train_ppo.py --timesteps 50000 --load-path models/rl/ppo_grid.zip

作者: PowerNexus Team
日期: 2025-12-20
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# 禁用输出缓冲，确保实时输出
os.environ['PYTHONUNBUFFERED'] = '1'

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PowerNexus PPO 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试 (Mock 模式，1000 步)
  python tools/train_ppo.py --timesteps 1000 --use-mock

  # 正式训练 (10万步)
  python tools/train_ppo.py --timesteps 100000

  # 继续训练已有模型
  python tools/train_ppo.py --timesteps 50000 --load-path models/rl/ppo_grid.zip
        """
    )
    
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100_000,
        help="训练总步数 (默认: 100000)"
    )
    
    parser.add_argument(
        "--save-path", "-s",
        type=str,
        default="./models/rl/ppo_grid.zip",
        help="模型保存路径 (默认: ./models/rl/ppo_grid.zip)"
    )
    
    parser.add_argument(
        "--load-path", "-l",
        type=str,
        default=None,
        help="加载已有模型继续训练 (可选)"
    )
    
    parser.add_argument(
        "--env-name", "-e",
        type=str,
        default="l2rpn_case14_sandbox",
        help="Grid2Op 环境名称 (默认: l2rpn_case14_sandbox)"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=3e-4,
        help="学习率 (默认: 3e-4)"
    )
    
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="使用 Mock 模式 (无需 Grid2Op)"
    )
    
    parser.add_argument(
        "--eval-episodes", 
        type=int,
        default=0,  # 默认跳过评估，因为 Grid2Op 评估可能很慢
        help="训练后评估的 episode 数 (默认: 0, 跳过评估)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="TensorBoard 日志目录 (可选)"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("PowerNexus - PPO 训练脚本")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 显示配置
    print("训练配置:")
    print(f"  - 训练步数: {args.timesteps:,}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 环境: {args.env_name}")
    print(f"  - 模式: {'Mock' if args.use_mock else '真实 Grid2Op'}")
    print(f"  - 保存路径: {args.save_path}")
    if args.load_path:
        print(f"  - 加载模型: {args.load_path}")
    if args.tensorboard:
        print(f"  - TensorBoard: {args.tensorboard}")
    print()
    
    # 导入 RL 模块
    try:
        from src.rl_engine import (
            PPO_Agent, 
            PPOAgentConfig, 
            PowerGridEnvConfig,
            TrainingConfig,
            create_ppo_agent,
            SB3_AVAILABLE,
        )
        logger.info("RL 模块导入成功")
    except ImportError as e:
        logger.error(f"导入 RL 模块失败: {e}")
        print("\n请确保已安装依赖: pip install stable-baselines3 grid2op")
        sys.exit(1)
    
    # 检查 SB3 可用性
    if not SB3_AVAILABLE and not args.use_mock:
        logger.warning("Stable-Baselines3 未安装，自动切换到 Mock 模式")
        args.use_mock = True
    
    # 创建配置
    env_config = PowerGridEnvConfig(
        env_name=args.env_name,
        use_mock=args.use_mock,
    )
    
    agent_config = PPOAgentConfig(
        learning_rate=args.learning_rate,
        tensorboard_log=args.tensorboard,
    )
    
    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        save_path=str(Path(args.save_path).parent / "checkpoints"),
        best_model_save_path=str(Path(args.save_path).parent / "best_model"),
    )
    
    agent = None
    training_success = False
    
    # 创建智能体
    print("创建 PPO 智能体...")
    try:
        agent = PPO_Agent(
            env_config=env_config,
            agent_config=agent_config,
            model_path=args.load_path,
            use_mock=args.use_mock,
            enable_llm=False,  # 训练时不需要 LLM
        )
        
        model_info = agent.get_model_info()
        print(f"  - 观测空间: {model_info['observation_space']}")
        print(f"  - 动作空间: {model_info['action_space']}")
        print(f"  - Mock 模式: {model_info['use_mock']}")
        print()
        
    except Exception as e:
        logger.error(f"创建智能体失败: {e}")
        sys.exit(1)
    
    # 开始训练
    print("=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    try:
        result = agent.train_model(
            total_timesteps=args.timesteps,
            training_config=training_config,
            progress_bar=True,
        )
        
        if result["status"] == "success":
            print()
            print("✅ 训练完成!")
            print(f"  - 总步数: {result.get('model_timesteps', args.timesteps):,}")
            training_success = True
        else:
            print()
            print(f"❌ 训练失败: {result.get('error', '未知错误')}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
        print("正在保存当前模型...")
        training_success = True  # 仍然尝试保存
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存模型
    if agent is not None:
        print()
        print(f"保存模型到: {args.save_path}")
        try:
            agent.save(args.save_path)
            print("✅ 模型保存成功")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    # 评估模型 (仅在训练成功时)
    if training_success and args.eval_episodes > 0 and agent is not None:
        print()
        print("=" * 60)
        print(f"评估模型 ({args.eval_episodes} episodes)...")
        print("=" * 60)
        
        try:
            eval_result = agent.evaluate(n_eval_episodes=args.eval_episodes)
            
            print()
            print("评估结果:")
            print(f"  - 平均奖励: {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
            print(f"  - 最小奖励: {eval_result['min_reward']:.2f}")
            print(f"  - 最大奖励: {eval_result['max_reward']:.2f}")
            print(f"  - 平均步数: {eval_result['mean_length']:.1f}")
            
        except Exception as e:
            logger.warning(f"评估失败 (可忽略): {e}")
            print("⚠️ 评估跳过 (模型已保存)")
    
    print()
    print("=" * 60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 显式成功退出
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
