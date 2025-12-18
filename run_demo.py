#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PowerNexus - 完整演示脚本

模拟场景: 无人机发现绝缘子裂纹 → RAG 检索维护手册 → RL 评估后因负荷过高建议延期维护

运行方式:
    python run_demo.py

作者: PowerNexus Team
日期: 2025-12-18
"""

import logging
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ██████╗  ██████╗ ██╗    ██╗███████╗██████╗ ███╗   ██╗███████╗   ║
║     ██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔══██╗████╗  ██║██╔════╝   ║
║     ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝██╔██╗ ██║█████╗     ║
║     ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗██║╚██╗██║██╔══╝     ║
║     ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║██║ ╚████║███████╗   ║
║     ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ║
║                                                                      ║
║          自主电网巡检与动态调度系统 - PowerNexus Demo                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_scenario_high_load():
    """
    演示场景: 高负荷情况下的缺陷处理
    
    场景描述:
    - 无人机巡检发现 #35 塔绝缘子存在裂纹
    - 当前电网负荷处于高峰期 (负载率 85-95%)
    - RL 智能体评估后建议先进行负荷转移再维护
    """
    print("\n" + "="*70)
    print("         📋 演示场景: 高负荷期间的绝缘子裂纹处理")
    print("="*70)
    
    print("""
    📝 场景描述:
    ┌─────────────────────────────────────────────────────────────────┐
    │  时间: 2024-12-18 14:30 (用电高峰)                              │
    │  地点: 220kV 传输线路 L1 #35 塔                                  │
    │  事件: 无人机巡检发现绝缘子疑似裂纹                              │
    │  电网状态: 负荷高峰期，多条线路负载率超过 85%                    │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    input("按 Enter 键开始演示...")
    
    # 导入模块
    from src.main import MainAgent, InspectionInput, GridTelemetry
    
    print("\n" + "-"*70)
    print("Step 0: 初始化 PowerNexus 智能体系统")
    print("-"*70)
    
    # 创建智能体 (Mock 模式)
    agent = MainAgent(use_mock=True)
    print(f"✓ 智能体初始化完成")
    print(f"  - 感知模块: {'就绪' if agent._perception_ready else '模拟模式'}")
    print(f"  - RAG 模块: {'就绪' if agent._rag_ready else '模拟模式'}")
    print(f"  - RL 模块: {'就绪' if agent._rl_ready else '模拟模式'}")
    
    # 模拟高负荷电网数据
    print("\n" + "-"*70)
    print("Step 1: 📡 获取电网遥测数据")
    print("-"*70)
    
    grid_telemetry = GridTelemetry.create_mock("high")  # 高负荷
    print(f"  最大线路负载率: {grid_telemetry.max_load:.1%}")
    print(f"  平均线路负载率: {grid_telemetry.avg_load:.1%}")
    print(f"  总负荷: {grid_telemetry.total_load:.1f} MW")
    print(f"  过载警告: {'⚠️ 是' if grid_telemetry.is_overloaded else '✓ 否'}")
    
    # 构建输入
    input_data = InspectionInput(
        image_path="./drone_images/insulator_crack_035.jpg",
        equipment_id="INS-L1-035-A",
        location="220kV L1 线路 #35 塔 A相",
        grid_telemetry=grid_telemetry,
    )
    
    print("\n" + "-"*70)
    print("Step 2-5: 🔄 运行智能体工作流")
    print("-"*70)
    print("  See    → 感知检测...")
    print("  Think  → 咨询 RAG...")
    print("  Decide → RL 决策...")
    print("  Act    → 生成报告...")
    
    # 运行工作流
    result = agent.run(input_data)
    
    # 输出结果
    print("\n" + "="*70)
    print("                    📄 巡检报告")
    print("="*70)
    print(result.report)
    
    # 输出关键决策
    print("\n" + "="*70)
    print("                    🔑 关键决策点")
    print("="*70)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │ 缺陷检测:                                                       │
    │   • 类型: {result.defect_result.get('defect_type', 'N/A'):20s}                            │
    │   • 置信度: {result.defect_result.get('confidence', 0):.1%}                                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ RL 决策:                                                        │
    │   • 可安全断开: {'是' if result.rl_safe_to_disconnect else '否':10s}                                  │
    │   • 推荐动作: {'保持状态' if result.rl_action == 0 else f'拓扑调整 #{result.rl_action}':20s}                            │
    ├─────────────────────────────────────────────────────────────────┤
    │ 最终决策: {result.maintenance_decision.value:20s}                            │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 行动建议
    print("\n" + "="*70)
    print("                    📋 行动建议")
    print("="*70)
    
    if result.maintenance_decision.value == "load_shift_required":
        print("""
    由于当前电网负荷较高，系统建议:
    
    1. ⚡ 负荷转移
       - 将 L1 线路部分负荷转移至 L2 线路
       - 协调发电机组调整出力分配
       - 目标: 将 L1 负载率降至 70% 以下
    
    2. 🔧 安排检修窗口
       - 预计可用窗口: 22:00-06:00 (低谷期)
       - 检修时长: 约 4 小时
       - 所需人员: 带电作业班组 4 人
    
    3. 📱 通知相关人员
       - 调度中心值班员
       - 输电检修班组
       - 安全监察人员
        """)
    else:
        print(f"\n    当前决策: {result.maintenance_decision.value}")
    
    # 清理
    agent.close()
    
    print("\n" + "="*70)
    print("                    ✅ 演示完成")
    print("="*70)
    
    return result


def demo_scenario_normal_load():
    """
    演示场景: 正常负荷情况下的缺陷处理
    """
    print("\n" + "="*70)
    print("         📋 演示场景: 正常负荷时的设备维护")
    print("="*70)
    
    from src.main import MainAgent, InspectionInput, GridTelemetry
    
    # 创建智能体
    agent = MainAgent(use_mock=True)
    
    # 正常负荷
    grid_telemetry = GridTelemetry.create_mock("normal")
    
    input_data = InspectionInput(
        image_path="./drone_images/rust_041.jpg",
        equipment_id="MTL-L2-041-B",
        location="110kV L2 线路 #41 塔 金具",
        grid_telemetry=grid_telemetry,
    )
    
    print(f"\n电网状态: 正常 (最大负载: {grid_telemetry.max_load:.1%})")
    
    result = agent.run(input_data)
    
    print(result.report)
    
    agent.close()
    
    return result


def demo_workflow_steps():
    """
    逐步演示工作流各阶段
    """
    print("\n" + "="*70)
    print("         📋 工作流各阶段详细演示")
    print("="*70)
    
    from src.main import MainAgent, InspectionInput, GridTelemetry, WorkflowResult
    
    agent = MainAgent(use_mock=True)
    
    # 准备输入
    grid_telemetry = GridTelemetry.create_mock("high")
    input_data = InspectionInput(
        image_path="./drone_images/oil_leak_012.jpg",
        equipment_id="TRF-SS3-001",
        location="35kV 变电站 #3 主变压器",
        grid_telemetry=grid_telemetry,
    )
    
    result = WorkflowResult()
    
    # 逐步执行
    print("\n" + "─"*50)
    print("🔍 Step 1: See (感知)")
    print("─"*50)
    result = agent._step_perceive(input_data, result)
    print(f"  检测结果: {result.defect_result}")
    input("\n按 Enter 继续...")
    
    print("\n" + "─"*50)
    print("📚 Step 2: Think (咨询)")
    print("─"*50)
    result = agent._step_consult(input_data, result)
    print(f"  RAG 上下文预览:\n{result.rag_context[:300]}...")
    input("\n按 Enter 继续...")
    
    print("\n" + "─"*50)
    print("🤖 Step 3: Decide (决策)")
    print("─"*50)
    result = agent._step_decide(input_data, result)
    print(f"  维护决策: {result.maintenance_decision.value}")
    print(f"  可安全断开: {result.rl_safe_to_disconnect}")
    input("\n按 Enter 继续...")
    
    print("\n" + "─"*50)
    print("📝 Step 4: Act (执行)")
    print("─"*50)
    result = agent._step_act(input_data, result)
    print(result.report)
    
    agent.close()
    
    return result


def main():
    """主函数"""
    print_banner()
    
    print("""
    请选择演示场景:
    
    [1] 高负荷场景 - 绝缘子裂纹 (推荐)
        无人机发现裂纹 → RAG 查标准 → RL 建议延期维护
    
    [2] 正常负荷场景 - 设备锈蚀
        正常负荷下的维护决策流程演示
    
    [3] 逐步演示 - 详细展示各工作流阶段
        交互式逐步执行，观察每个阶段的输出
    
    [4] 运行全部场景
    
    [0] 退出
    """)
    
    while True:
        choice = input("\n请输入选项 (0-4): ").strip()
        
        if choice == "1":
            demo_scenario_high_load()
        elif choice == "2":
            demo_scenario_normal_load()
        elif choice == "3":
            demo_workflow_steps()
        elif choice == "4":
            demo_scenario_high_load()
            print("\n" + "="*70 + "\n")
            demo_scenario_normal_load()
        elif choice == "0":
            print("\n感谢使用 PowerNexus! 👋")
            break
        else:
            print("无效选项，请重新输入")
        
        another = input("\n是否继续演示? (y/n): ").strip().lower()
        if another != 'y':
            print("\n感谢使用 PowerNexus! 👋")
            break


if __name__ == "__main__":
    main()
