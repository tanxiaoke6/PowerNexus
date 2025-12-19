# -*- coding: utf-8 -*-
"""
PowerNexus - 电网状态模拟数据生成器

本脚本用于生成多样化的电网状态快照（Snapshots），
可用于 RL 环境测试和模型评估。

目标系统: IEEE 14 节点系统 (L2RPN Case 14)
维度定义:
- 线路数量 (n_line): 20
- 发电机数量 (n_gen): 5
- 负荷数量 (n_load): 11
- 拓扑维度 (dim_topo): 57

使用方式:
    python tools/simulate_grid_state.py --n_samples 1000 --output data/grid_states.npz
    python tools/simulate_grid_state.py --scenario overload --n_samples 100

作者: PowerNexus Team
日期: 2025-12-19
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置参数
# ============================================================================

@dataclass
class GridConfig:
    """电网配置参数 (IEEE 14 节点系统)"""
    n_line: int = 20       # 线路数量
    n_gen: int = 5         # 发电机数量
    n_load: int = 11       # 负荷数量
    n_sub: int = 14        # 变电站数量
    dim_topo: int = 57     # 拓扑向量维度


@dataclass
class GenerationConfig:
    """数据生成配置"""
    # 线路负载率范围 (不同场景)
    rho_ranges: Dict[str, Tuple[float, float]] = None
    
    # 发电机有功出力范围 (MW)
    gen_p_range: Tuple[float, float] = (10.0, 150.0)
    
    # 发电机电压范围 (p.u.)
    gen_v_range: Tuple[float, float] = (0.95, 1.05)
    
    # 负荷有功范围 (MW)
    load_p_range: Tuple[float, float] = (5.0, 60.0)
    
    # 负荷功率因数范围 (用于计算无功)
    power_factor_range: Tuple[float, float] = (0.85, 0.98)
    
    def __post_init__(self):
        if self.rho_ranges is None:
            self.rho_ranges = {
                "normal": (0.2, 0.7),       # 正常运行
                "high_load": (0.7, 0.95),   # 高负荷
                "overload": (0.95, 1.2),    # 过载
                "mixed": (0.1, 1.1),        # 混合场景
            }


# ============================================================================
# 电网状态数据类
# ============================================================================

@dataclass
class GridState:
    """
    电网状态快照
    
    包含完整的观测空间数据。
    """
    rho: np.ndarray        # 线路负载率 (n_line,)
    gen_p: np.ndarray      # 发电机有功 (n_gen,)
    gen_v: np.ndarray      # 发电机电压 (n_gen,)
    load_p: np.ndarray     # 负荷有功 (n_load,)
    load_q: np.ndarray     # 负荷无功 (n_load,)
    topo_vect: np.ndarray  # 拓扑向量 (dim_topo,)
    scenario: str = ""     # 场景标签
    
    def to_observation(self) -> np.ndarray:
        """
        转换为观测向量
        
        Returns:
            拼接后的观测向量
        """
        return np.concatenate([
            self.rho.astype(np.float32),
            self.gen_p.astype(np.float32),
            self.gen_v.astype(np.float32),
            self.load_p.astype(np.float32),
            self.load_q.astype(np.float32),
            self.topo_vect.astype(np.float32),
        ])
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "rho": self.rho.tolist(),
            "gen_p": self.gen_p.tolist(),
            "gen_v": self.gen_v.tolist(),
            "load_p": self.load_p.tolist(),
            "load_q": self.load_q.tolist(),
            "topo_vect": self.topo_vect.tolist(),
            "scenario": self.scenario,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GridState":
        """从字典创建"""
        return cls(
            rho=np.array(data["rho"], dtype=np.float32),
            gen_p=np.array(data["gen_p"], dtype=np.float32),
            gen_v=np.array(data["gen_v"], dtype=np.float32),
            load_p=np.array(data["load_p"], dtype=np.float32),
            load_q=np.array(data["load_q"], dtype=np.float32),
            topo_vect=np.array(data["topo_vect"], dtype=np.int32),
            scenario=data.get("scenario", ""),
        )


# ============================================================================
# 电网状态生成器
# ============================================================================

class GridStateGenerator:
    """
    电网状态生成器
    
    生成多样化的电网状态快照，支持多种场景。
    
    Example:
        >>> generator = GridStateGenerator()
        >>> states = generator.generate_batch(n_samples=100, scenario="mixed")
        >>> generator.save_to_npz(states, "data/grid_states.npz")
    """
    
    def __init__(
        self,
        grid_config: Optional[GridConfig] = None,
        gen_config: Optional[GenerationConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        初始化生成器
        
        Args:
            grid_config: 电网配置
            gen_config: 生成配置
            seed: 随机种子
        """
        self.grid_config = grid_config or GridConfig()
        self.gen_config = gen_config or GenerationConfig()
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(
            f"电网状态生成器初始化 | "
            f"n_line={self.grid_config.n_line}, "
            f"n_gen={self.grid_config.n_gen}, "
            f"n_load={self.grid_config.n_load}"
        )
    
    def generate_rho(
        self,
        scenario: str = "normal",
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        生成线路负载率
        
        Args:
            scenario: 场景类型 ("normal", "high_load", "overload", "mixed")
            n_samples: 样本数量
            
        Returns:
            线路负载率数组 (n_samples, n_line)
        """
        n_line = self.grid_config.n_line
        
        if scenario == "mixed":
            # 混合场景：每条线路随机选择一个场景
            rho = np.zeros((n_samples, n_line), dtype=np.float32)
            for i in range(n_samples):
                for j in range(n_line):
                    # 60% 正常，30% 高负荷，10% 过载
                    rand = np.random.random()
                    if rand < 0.6:
                        low, high = self.gen_config.rho_ranges["normal"]
                    elif rand < 0.9:
                        low, high = self.gen_config.rho_ranges["high_load"]
                    else:
                        low, high = self.gen_config.rho_ranges["overload"]
                    rho[i, j] = np.random.uniform(low, high)
        else:
            # 单一场景
            low, high = self.gen_config.rho_ranges.get(
                scenario, 
                self.gen_config.rho_ranges["normal"]
            )
            rho = np.random.uniform(low, high, (n_samples, n_line)).astype(np.float32)
        
        return rho
    
    def generate_gen_p(self, n_samples: int = 1) -> np.ndarray:
        """
        生成发电机有功出力
        
        考虑不同发电机的容量差异。
        
        Args:
            n_samples: 样本数量
            
        Returns:
            发电机有功数组 (n_samples, n_gen)
        """
        n_gen = self.grid_config.n_gen
        low, high = self.gen_config.gen_p_range
        
        # 定义各发电机的典型出力比例 (模拟不同容量)
        # 假设：G1 最大，G2-G5 依次递减
        capacity_ratios = np.array([1.0, 0.6, 0.4, 0.5, 0.3])
        
        gen_p = np.zeros((n_samples, n_gen), dtype=np.float32)
        for i in range(n_gen):
            gen_p[:, i] = np.random.uniform(
                low * capacity_ratios[i],
                high * capacity_ratios[i],
                n_samples
            )
        
        return gen_p
    
    def generate_gen_v(self, n_samples: int = 1) -> np.ndarray:
        """
        生成发电机电压
        
        发电机电压通常维持在 1.0 p.u. 左右。
        
        Args:
            n_samples: 样本数量
            
        Returns:
            发电机电压数组 (n_samples, n_gen)
        """
        n_gen = self.grid_config.n_gen
        low, high = self.gen_config.gen_v_range
        
        return np.random.uniform(low, high, (n_samples, n_gen)).astype(np.float32)
    
    def generate_load(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成负荷有功和无功
        
        无功由有功和功率因数计算得到。
        
        Args:
            n_samples: 样本数量
            
        Returns:
            (load_p, load_q) 元组
        """
        n_load = self.grid_config.n_load
        low, high = self.gen_config.load_p_range
        pf_low, pf_high = self.gen_config.power_factor_range
        
        # 生成有功负荷
        load_p = np.random.uniform(low, high, (n_samples, n_load)).astype(np.float32)
        
        # 生成功率因数
        power_factor = np.random.uniform(pf_low, pf_high, (n_samples, n_load))
        
        # 根据功率因数计算无功: Q = P * tan(arccos(pf))
        # tan(arccos(pf)) = sqrt(1 - pf^2) / pf
        tan_phi = np.sqrt(1 - power_factor**2) / power_factor
        load_q = (load_p * tan_phi).astype(np.float32)
        
        return load_p, load_q
    
    def generate_topo_vect(
        self,
        n_samples: int = 1,
        split_probability: float = 0.05,
    ) -> np.ndarray:
        """
        生成拓扑向量
        
        大多数元素连接到母线 1，少部分可能连接到母线 2（母线分裂）。
        
        Args:
            n_samples: 样本数量
            split_probability: 母线分裂概率
            
        Returns:
            拓扑向量数组 (n_samples, dim_topo)
        """
        dim_topo = self.grid_config.dim_topo
        
        # 默认所有元素连接到母线 1
        topo_vect = np.ones((n_samples, dim_topo), dtype=np.int32)
        
        # 随机将部分元素切换到母线 2
        mask = np.random.random((n_samples, dim_topo)) < split_probability
        topo_vect[mask] = 2
        
        return topo_vect
    
    def generate_single(self, scenario: str = "normal") -> GridState:
        """
        生成单个电网状态
        
        Args:
            scenario: 场景类型
            
        Returns:
            GridState 对象
        """
        rho = self.generate_rho(scenario, 1)[0]
        gen_p = self.generate_gen_p(1)[0]
        gen_v = self.generate_gen_v(1)[0]
        load_p, load_q = self.generate_load(1)
        topo_vect = self.generate_topo_vect(1)[0]
        
        return GridState(
            rho=rho,
            gen_p=gen_p,
            gen_v=gen_v,
            load_p=load_p[0],
            load_q=load_q[0],
            topo_vect=topo_vect,
            scenario=scenario,
        )
    
    def generate_batch(
        self,
        n_samples: int,
        scenario: str = "mixed",
    ) -> List[GridState]:
        """
        批量生成电网状态
        
        Args:
            n_samples: 样本数量
            scenario: 场景类型
            
        Returns:
            GridState 对象列表
        """
        logger.info(f"开始生成 {n_samples} 个电网状态 (场景: {scenario})")
        
        # 批量生成各个组件
        rho = self.generate_rho(scenario, n_samples)
        gen_p = self.generate_gen_p(n_samples)
        gen_v = self.generate_gen_v(n_samples)
        load_p, load_q = self.generate_load(n_samples)
        topo_vect = self.generate_topo_vect(n_samples)
        
        # 组装为 GridState 列表
        states = []
        for i in range(n_samples):
            state = GridState(
                rho=rho[i],
                gen_p=gen_p[i],
                gen_v=gen_v[i],
                load_p=load_p[i],
                load_q=load_q[i],
                topo_vect=topo_vect[i],
                scenario=scenario,
            )
            states.append(state)
        
        logger.info(f"生成完成: {len(states)} 个电网状态")
        return states
    
    def generate_mixed_scenarios(
        self,
        n_samples: int,
        scenario_weights: Optional[Dict[str, float]] = None,
    ) -> List[GridState]:
        """
        生成混合场景的电网状态
        
        按权重比例生成不同场景的数据。
        
        Args:
            n_samples: 总样本数量
            scenario_weights: 场景权重 {"normal": 0.6, "high_load": 0.3, ...}
            
        Returns:
            GridState 对象列表
        """
        if scenario_weights is None:
            scenario_weights = {
                "normal": 0.5,
                "high_load": 0.3,
                "overload": 0.15,
                "mixed": 0.05,
            }
        
        # 归一化权重
        total_weight = sum(scenario_weights.values())
        scenario_weights = {k: v / total_weight for k, v in scenario_weights.items()}
        
        states = []
        for scenario, weight in scenario_weights.items():
            n_scenario = int(n_samples * weight)
            if n_scenario > 0:
                states.extend(self.generate_batch(n_scenario, scenario))
        
        # 补齐剩余样本
        remaining = n_samples - len(states)
        if remaining > 0:
            states.extend(self.generate_batch(remaining, "normal"))
        
        # 打乱顺序
        np.random.shuffle(states)
        
        return states
    
    # ========================================================================
    # 保存和加载
    # ========================================================================
    
    def save_to_npz(
        self,
        states: List[GridState],
        filepath: Union[str, Path],
    ):
        """
        保存状态到 NPZ 文件
        
        Args:
            states: 状态列表
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 将所有状态转换为数组
        n_samples = len(states)
        
        rho = np.stack([s.rho for s in states])
        gen_p = np.stack([s.gen_p for s in states])
        gen_v = np.stack([s.gen_v for s in states])
        load_p = np.stack([s.load_p for s in states])
        load_q = np.stack([s.load_q for s in states])
        topo_vect = np.stack([s.topo_vect for s in states])
        scenarios = np.array([s.scenario for s in states])
        
        # 保存为 NPZ
        np.savez(
            filepath,
            rho=rho,
            gen_p=gen_p,
            gen_v=gen_v,
            load_p=load_p,
            load_q=load_q,
            topo_vect=topo_vect,
            scenarios=scenarios,
            n_samples=n_samples,
            # 元数据
            n_line=self.grid_config.n_line,
            n_gen=self.grid_config.n_gen,
            n_load=self.grid_config.n_load,
            dim_topo=self.grid_config.dim_topo,
        )
        
        logger.info(f"状态数据已保存到: {filepath} ({n_samples} 个样本)")
    
    def save_to_json(
        self,
        states: List[GridState],
        filepath: Union[str, Path],
    ):
        """
        保存状态到 JSON 文件
        
        Args:
            states: 状态列表
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "n_samples": len(states),
            "grid_config": asdict(self.grid_config),
            "states": [s.to_dict() for s in states],
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"状态数据已保存到: {filepath} ({len(states)} 个样本)")
    
    @staticmethod
    def load_from_npz(filepath: Union[str, Path]) -> List[GridState]:
        """
        从 NPZ 文件加载状态
        
        Args:
            filepath: 文件路径
            
        Returns:
            GridState 对象列表
        """
        data = np.load(filepath, allow_pickle=True)
        
        n_samples = int(data['n_samples'])
        states = []
        
        for i in range(n_samples):
            state = GridState(
                rho=data['rho'][i],
                gen_p=data['gen_p'][i],
                gen_v=data['gen_v'][i],
                load_p=data['load_p'][i],
                load_q=data['load_q'][i],
                topo_vect=data['topo_vect'][i],
                scenario=str(data['scenarios'][i]),
            )
            states.append(state)
        
        logger.info(f"从 {filepath} 加载了 {len(states)} 个状态")
        return states
    
    @staticmethod
    def load_from_json(filepath: Union[str, Path]) -> List[GridState]:
        """
        从 JSON 文件加载状态
        
        Args:
            filepath: 文件路径
            
        Returns:
            GridState 对象列表
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        states = [GridState.from_dict(s) for s in data['states']]
        
        logger.info(f"从 {filepath} 加载了 {len(states)} 个状态")
        return states
    
    # ========================================================================
    # 统计信息
    # ========================================================================
    
    def print_statistics(self, states: List[GridState]):
        """
        打印状态数据的统计摘要
        
        Args:
            states: 状态列表
        """
        n_samples = len(states)
        
        # 收集所有数据
        rho = np.stack([s.rho for s in states])
        gen_p = np.stack([s.gen_p for s in states])
        gen_v = np.stack([s.gen_v for s in states])
        load_p = np.stack([s.load_p for s in states])
        load_q = np.stack([s.load_q for s in states])
        topo_vect = np.stack([s.topo_vect for s in states])
        
        # 场景分布
        scenario_counts = {}
        for s in states:
            scenario_counts[s.scenario] = scenario_counts.get(s.scenario, 0) + 1
        
        print("\n" + "=" * 60)
        print("电网状态数据统计摘要")
        print("=" * 60)
        
        print(f"\n总样本数: {n_samples}")
        print(f"\n场景分布:")
        for scenario, count in sorted(scenario_counts.items()):
            print(f"  - {scenario}: {count} ({count/n_samples*100:.1f}%)")
        
        print(f"\n线路负载率 (rho):")
        print(f"  最小值: {rho.min():.4f}")
        print(f"  最大值: {rho.max():.4f}")
        print(f"  均值: {rho.mean():.4f}")
        print(f"  标准差: {rho.std():.4f}")
        print(f"  过载线路比例 (>0.95): {(rho > 0.95).sum() / rho.size * 100:.2f}%")
        
        print(f"\n发电机有功 (gen_p) [MW]:")
        print(f"  最小值: {gen_p.min():.2f}")
        print(f"  最大值: {gen_p.max():.2f}")
        print(f"  均值: {gen_p.mean():.2f}")
        
        print(f"\n发电机电压 (gen_v) [p.u.]:")
        print(f"  最小值: {gen_v.min():.4f}")
        print(f"  最大值: {gen_v.max():.4f}")
        print(f"  均值: {gen_v.mean():.4f}")
        
        print(f"\n负荷有功 (load_p) [MW]:")
        print(f"  最小值: {load_p.min():.2f}")
        print(f"  最大值: {load_p.max():.2f}")
        print(f"  均值: {load_p.mean():.2f}")
        
        print(f"\n负荷无功 (load_q) [MVar]:")
        print(f"  最小值: {load_q.min():.2f}")
        print(f"  最大值: {load_q.max():.2f}")
        print(f"  均值: {load_q.mean():.2f}")
        
        print(f"\n拓扑向量 (topo_vect):")
        print(f"  母线1连接比例: {(topo_vect == 1).sum() / topo_vect.size * 100:.2f}%")
        print(f"  母线2连接比例: {(topo_vect == 2).sum() / topo_vect.size * 100:.2f}%")
        
        # 观测向量维度
        obs_dim = states[0].to_observation().shape[0]
        print(f"\n观测向量维度: {obs_dim}")
        print(f"  = n_line({self.grid_config.n_line}) + "
              f"n_gen({self.grid_config.n_gen}) + "
              f"n_gen({self.grid_config.n_gen}) + "
              f"n_load({self.grid_config.n_load}) + "
              f"n_load({self.grid_config.n_load}) + "
              f"dim_topo({self.grid_config.dim_topo})")
        
        print("\n" + "=" * 60)


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PowerNexus 电网状态模拟数据生成器"
    )
    
    parser.add_argument(
        "--n_samples", "-n",
        type=int,
        default=1000,
        help="生成的样本数量 (默认: 1000)"
    )
    
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="mixed",
        choices=["normal", "high_load", "overload", "mixed", "weighted"],
        help="场景类型 (默认: mixed)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/grid_states.npz",
        help="输出文件路径 (默认: data/grid_states.npz)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="npz",
        choices=["npz", "json"],
        help="输出格式 (默认: npz)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子 (用于复现)"
    )
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = GridStateGenerator(seed=args.seed)
    
    # 生成数据
    if args.scenario == "weighted":
        states = generator.generate_mixed_scenarios(args.n_samples)
    else:
        states = generator.generate_batch(args.n_samples, args.scenario)
    
    # 打印统计信息
    generator.print_statistics(states)
    
    # 保存数据
    if args.format == "npz":
        output_path = args.output if args.output.endswith('.npz') else args.output + '.npz'
        generator.save_to_npz(states, output_path)
    else:
        output_path = args.output if args.output.endswith('.json') else args.output + '.json'
        generator.save_to_json(states, output_path)
    
    print(f"\n数据已保存到: {output_path}")


if __name__ == "__main__":
    main()
