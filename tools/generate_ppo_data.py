# -*- coding: utf-8 -*-
"""
PowerNexus - PPO 训练数据生成脚本

专门为 PPO 训练生成电网状态数据的便捷脚本。
基于 simulate_grid_state.py，提供更简单的接口。

使用方法:
    # 生成默认训练数据 (1000个样本，混合场景)
    python tools/generate_ppo_data.py
    
    # 生成更多样本
    python tools/generate_ppo_data.py --samples 5000
    
    # 生成特定场景数据
    python tools/generate_ppo_data.py --scenario overload --samples 500
    
    # 生成完整训练/验证/测试集
    python tools/generate_ppo_data.py --full-dataset

作者: PowerNexus Team
日期: 2025-12-20
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.simulate_grid_state import GridStateGenerator, GridConfig, GenerationConfig


def generate_training_data(
    n_samples: int = 1000,
    scenario: str = "mixed",
    output_dir: str = "data",
    seed: int = None,
):
    """
    生成 PPO 训练数据
    
    Args:
        n_samples: 样本数量
        scenario: 场景类型 (normal, high_load, overload, mixed)
        output_dir: 输出目录
        seed: 随机种子
    """
    print("=" * 60)
    print("PowerNexus - PPO 训练数据生成器")
    print("=" * 60)
    print()
    
    # 创建生成器
    generator = GridStateGenerator(seed=seed)
    
    # 生成数据
    print(f"生成 {n_samples} 个电网状态 (场景: {scenario})...")
    
    if scenario == "weighted":
        # 按权重生成不同场景，更适合训练
        scenario_weights = {
            "normal": 0.4,      # 40% 正常运行
            "high_load": 0.35,  # 35% 高负荷
            "overload": 0.20,   # 20% 过载
            "mixed": 0.05,      # 5% 混合
        }
        states = generator.generate_mixed_scenarios(n_samples, scenario_weights)
    else:
        states = generator.generate_batch(n_samples, scenario)
    
    # 打印统计信息
    generator.print_statistics(states)
    
    # 保存数据
    output_path = Path(output_dir) / "grid_states.npz"
    generator.save_to_npz(states, output_path)
    
    print(f"\n✅ 数据已保存到: {output_path}")
    print(f"   样本数量: {len(states)}")
    print(f"   观测维度: {states[0].to_observation().shape[0]}")
    
    return output_path


def generate_full_dataset(
    train_samples: int = 5000,
    val_samples: int = 1000,
    test_samples: int = 1000,
    output_dir: str = "data/ppo_dataset",
    seed: int = 42,
):
    """
    生成完整的训练/验证/测试数据集
    
    Args:
        train_samples: 训练集样本数
        val_samples: 验证集样本数
        test_samples: 测试集样本数
        output_dir: 输出目录
        seed: 随机种子
    """
    print("=" * 60)
    print("PowerNexus - 完整 PPO 数据集生成器")
    print("=" * 60)
    print()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成器配置
    generator = GridStateGenerator(seed=seed)
    
    # 场景权重配置
    train_weights = {
        "normal": 0.35,
        "high_load": 0.35,
        "overload": 0.25,
        "mixed": 0.05,
    }
    
    val_test_weights = {
        "normal": 0.4,
        "high_load": 0.3,
        "overload": 0.2,
        "mixed": 0.1,
    }
    
    # 生成训练集
    print(f"\n📊 生成训练集 ({train_samples} 样本)...")
    train_states = generator.generate_mixed_scenarios(train_samples, train_weights)
    generator.save_to_npz(train_states, output_path / "train.npz")
    print(f"   ✅ 保存到: {output_path / 'train.npz'}")
    
    # 生成验证集
    print(f"\n📊 生成验证集 ({val_samples} 样本)...")
    val_states = generator.generate_mixed_scenarios(val_samples, val_test_weights)
    generator.save_to_npz(val_states, output_path / "val.npz")
    print(f"   ✅ 保存到: {output_path / 'val.npz'}")
    
    # 生成测试集
    print(f"\n📊 生成测试集 ({test_samples} 样本)...")
    test_states = generator.generate_mixed_scenarios(test_samples, val_test_weights)
    generator.save_to_npz(test_states, output_path / "test.npz")
    print(f"   ✅ 保存到: {output_path / 'test.npz'}")
    
    # 总结
    print("\n" + "=" * 60)
    print("数据集生成完成!")
    print("=" * 60)
    print(f"""
数据集路径: {output_path}
├── train.npz  ({train_samples} 样本)
├── val.npz    ({val_samples} 样本)  
└── test.npz   ({test_samples} 样本)

总计: {train_samples + val_samples + test_samples} 样本

使用方法:
  1. PPO 训练:
     python tools/train_ppo.py --timesteps 100000
  
  2. 加载数据进行自定义训练:
     import numpy as np
     data = np.load('data/ppo_dataset/train.npz')
     observations = np.concatenate([
         data['rho'], data['gen_p'], data['gen_v'],
         data['load_p'], data['load_q'], data['topo_vect']
     ], axis=1)
""")
    
    return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PowerNexus PPO 训练数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成默认训练数据 (1000 样本)
  python tools/generate_ppo_data.py

  # 生成 5000 样本，过载场景
  python tools/generate_ppo_data.py --samples 5000 --scenario overload

  # 生成完整数据集 (训练/验证/测试)
  python tools/generate_ppo_data.py --full-dataset

  # 生成大规模训练集
  python tools/generate_ppo_data.py --full-dataset --train-samples 10000
        """
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=1000,
        help="样本数量 (默认: 1000)"
    )
    
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="weighted",
        choices=["normal", "high_load", "overload", "mixed", "weighted"],
        help="场景类型 (默认: weighted，按比例混合各场景)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data",
        help="输出目录 (默认: data)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="生成完整的训练/验证/测试数据集"
    )
    
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="训练集样本数 (仅 --full-dataset 模式，默认: 5000)"
    )
    
    parser.add_argument(
        "--val-samples",
        type=int,
        default=1000,
        help="验证集样本数 (仅 --full-dataset 模式，默认: 1000)"
    )
    
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        help="测试集样本数 (仅 --full-dataset 模式，默认: 1000)"
    )
    
    args = parser.parse_args()
    
    if args.full_dataset:
        generate_full_dataset(
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            output_dir=args.output if args.output != "data" else "data/ppo_dataset",
            seed=args.seed,
        )
    else:
        generate_training_data(
            n_samples=args.samples,
            scenario=args.scenario,
            output_dir=args.output,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
