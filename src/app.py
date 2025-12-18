# -*- coding: utf-8 -*-
"""
PowerNexus - Streamlit 仪表板

基于 Qwen2.5 模型的电网巡检智能决策系统。

功能:
- Qwen-VL 视觉分析
- RAG 知识检索
- RL 优化决策
- 系统状态监控

运行方式:
    streamlit run src/app.py

作者: PowerNexus Team
日期: 2025-12-18
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import json
from datetime import datetime

# ============================================================================
# 系统检测
# ============================================================================

# 检测 CUDA
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_VERSION = "未安装"

# 检测各模块
PERCEPTION_AVAILABLE = False
RAG_AVAILABLE = False
RL_AVAILABLE = False
LLM_AVAILABLE = False

try:
    from src.perception import DefectDetector, create_detector
    PERCEPTION_AVAILABLE = True
except ImportError as e:
    st.warning(f"感知模块加载失败: {e}")

try:
    from src.rag import KnowledgeBase, init_knowledge_base_with_samples
    RAG_AVAILABLE = True
except ImportError as e:
    st.warning(f"RAG 模块加载失败: {e}")

try:
    from src.rl_engine import PPO_Agent, create_ppo_agent
    RL_AVAILABLE = True
except ImportError as e:
    st.warning(f"RL 模块加载失败: {e}")

try:
    from src.utils.llm_engine import LLMEngine, create_llm_engine
    LLM_AVAILABLE = True
except ImportError as e:
    pass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="PowerNexus - 电网智能巡检系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5 0%, #00ACC1 50%, #00897B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 侧边栏
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=PowerNexus", width=200)
    st.markdown("---")
    
    st.header("⚙️ 模型设置")
    
    # 量化选项
    use_4bit = st.checkbox(
        "启用 4-bit 量化",
        value=True,
        help="使用 BitsAndBytes 4-bit 量化减少显存占用"
    )
    
    use_mock = st.checkbox(
        "Mock 模式 (测试)",
        value=True,
        help="使用模拟模型进行测试，无需加载实际模型"
    )
    
    st.markdown("---")
    
    st.header("📊 系统状态")
    
    # 系统状态显示
    col1, col2 = st.columns(2)
    with col1:
        if CUDA_AVAILABLE:
            st.success("GPU ✓")
        else:
            st.warning("CPU")
    with col2:
        st.info(f"PyTorch {TORCH_VERSION[:5] if len(TORCH_VERSION) > 5 else TORCH_VERSION}")
    
    # 模块状态
    st.markdown("**模块状态:**")
    modules = [
        ("感知", PERCEPTION_AVAILABLE),
        ("RAG", RAG_AVAILABLE),
        ("RL", RL_AVAILABLE),
        ("LLM", LLM_AVAILABLE),
    ]
    
    for name, available in modules:
        if available:
            st.markdown(f"- ✅ {name}")
        else:
            st.markdown(f"- ❌ {name}")
    
    st.markdown("---")
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# 主页面
# ============================================================================

# 标题
st.markdown('<h1 class="main-header">⚡ PowerNexus</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Qwen2.5 | 电网智能巡检与决策系统</p>', unsafe_allow_html=True)

# CUDA 警告
if not CUDA_AVAILABLE:
    st.warning("⚠️ **GPU 不可用** - 系统运行在 CPU 模式，推理速度可能较慢。建议安装 CUDA 版本的 PyTorch 以获得更好性能。")

# 主要功能区
tab1, tab2, tab3, tab4 = st.tabs(["🔍 视觉检测", "📚 知识检索", "🤖 RL 优化", "📊 系统信息"])


# ============================================================================
# Tab 1: 视觉检测
# ============================================================================

with tab1:
    st.header("🔍 Qwen-VL 视觉分析")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("上传图像")
        
        # 图像上传
        uploaded_file = st.file_uploader(
            "选择电力设备图像",
            type=["jpg", "jpeg", "png"],
            help="支持 JPG、PNG 格式图像"
        )
        
        # 示例图像
        st.markdown("**或使用示例图像:**")
        example_images = {
            "绝缘子裂纹": "data/images/insulator_defect.jpg",
            "正常设备": "data/images/insulator_normal.jpg",
            "金具锈蚀": "data/images/metal_rust.jpg",
        }
        
        selected_example = st.selectbox("选择示例", list(example_images.keys()))
        use_example = st.button("使用示例图像")
        
        # 显示图像
        image_to_analyze = None
        image_path = None
        
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file) if PIL_AVAILABLE else None
            # 保存临时文件
            temp_path = Path("data/images/temp_upload.jpg")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            if image_to_analyze:
                image_to_analyze.save(temp_path)
                image_path = str(temp_path)
        elif use_example:
            image_path = example_images[selected_example]
            if Path(image_path).exists() and PIL_AVAILABLE:
                image_to_analyze = Image.open(image_path)
        
        if image_to_analyze:
            st.image(image_to_analyze, caption="待分析图像", use_container_width=True)
    
    with col2:
        st.subheader("分析结果")
        
        if st.button("🔬 执行 Qwen-VL 分析", type="primary", use_container_width=True):
            if not PERCEPTION_AVAILABLE:
                st.error("感知模块未加载")
            elif image_path is None:
                st.warning("请先上传或选择图像")
            else:
                with st.spinner("Qwen-VL 正在分析图像..."):
                    try:
                        # 创建检测器
                        detector = create_detector(use_mock=use_mock)
                        
                        # 执行检测
                        result = detector.detect(image_path)
                        
                        # 显示结果
                        if result.defect_detected:
                            st.error("⚠️ 检测到缺陷!")
                        else:
                            st.success("✅ 设备状态正常")
                        
                        # 详细信息
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("缺陷类型", result.defect_type)
                        with col_b:
                            st.metric("严重程度", result.severity)
                        
                        st.metric("置信度", f"{result.confidence:.1%}")
                        
                        st.markdown("**描述:**")
                        st.info(result.description)
                        
                        # JSON 输出
                        with st.expander("查看 JSON 输出"):
                            st.json(result.to_dict())
                        
                        detector.close()
                        
                    except Exception as e:
                        st.error(f"分析失败: {e}")


# ============================================================================
# Tab 2: 知识检索
# ============================================================================

with tab2:
    st.header("📚 RAG 知识检索")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("查询问题")
        
        # 预设问题
        preset_questions = [
            "变压器绝缘电阻测量的环境要求是什么？",
            "绝缘子裂纹应该如何处理？",
            "高温环境下设备运行有什么注意事项？",
            "负荷转移操作的流程是什么？",
        ]
        
        question_type = st.radio("问题来源", ["预设问题", "自定义问题"])
        
        if question_type == "预设问题":
            question = st.selectbox("选择问题", preset_questions)
        else:
            question = st.text_area("输入问题", placeholder="请输入您的技术问题...")
        
        top_k = st.slider("检索结果数量", 1, 10, 3)
    
    with col2:
        st.subheader("检索结果")
        
        if st.button("🔍 执行 RAG 检索", type="primary", use_container_width=True):
            if not RAG_AVAILABLE:
                st.error("RAG 模块未加载")
            elif not question:
                st.warning("请输入问题")
            else:
                with st.spinner("正在检索知识库..."):
                    try:
                        # 初始化知识库
                        kb = init_knowledge_base_with_samples(use_mock=use_mock)
                        
                        # 检索
                        result = kb.query(question, top_k=top_k)
                        
                        st.success(f"✅ 找到 {result.num_results} 条相关结果")
                        
                        # 显示上下文
                        st.markdown("**检索到的参考资料:**")
                        st.text_area("上下文", result.formatted_context, height=200)
                        
                        # Qwen 合成答案
                        if hasattr(kb, 'query_and_synthesize') and LLM_AVAILABLE:
                            st.markdown("**Qwen LLM 生成的答案:**")
                            synth_result = kb.query_and_synthesize(question, top_k=top_k)
                            st.info(synth_result.synthesized_answer)
                        
                        # 完整提示
                        with st.expander("查看完整 LLM 提示"):
                            st.code(result.prompt, language="markdown")
                        
                    except Exception as e:
                        st.error(f"检索失败: {e}")


# ============================================================================
# Tab 3: RL 优化
# ============================================================================

with tab3:
    st.header("🤖 RL 优化决策")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("电网状态输入")
        
        # 负载设置
        st.markdown("**线路负载率设置:**")
        load_level = st.select_slider(
            "负荷水平",
            options=["低负荷", "正常", "较高", "高峰", "临界"],
            value="正常"
        )
        
        load_map = {
            "低负荷": 0.3,
            "正常": 0.5,
            "较高": 0.7,
            "高峰": 0.85,
            "临界": 0.95,
        }
        
        max_load = load_map[load_level]
        
        # 显示模拟负载
        st.markdown("**模拟线路负载:**")
        import random
        random.seed(42)
        
        line_loads = [max_load + random.uniform(-0.1, 0.1) for _ in range(5)]
        line_loads = [max(0, min(1, l)) for l in line_loads]
        
        for i, load in enumerate(line_loads):
            color = "red" if load > 0.9 else "orange" if load > 0.75 else "green"
            st.progress(load, text=f"线路 L{i+1}: {load:.1%}")
        
        # 状态描述
        grid_state = f"最大负载率 {max_load:.0%}，部分线路接近容量上限"
        st.text_area("电网状态描述", grid_state, height=80)
    
    with col2:
        st.subheader("RL 决策结果")
        
        if st.button("🎯 执行 RL 优化", type="primary", use_container_width=True):
            if not RL_AVAILABLE:
                st.error("RL 模块未加载")
            else:
                with st.spinner("PPO 智能体正在分析..."):
                    try:
                        import numpy as np
                        
                        # 创建智能体
                        agent = create_ppo_agent(use_mock=use_mock)
                        
                        # 构造观测
                        obs = np.array(line_loads + [0.5] * 100, dtype=np.float32)
                        obs = obs[:agent.env.observation_space.shape[0]]
                        
                        # 预测动作
                        action = agent.predict_action(obs)
                        action_name = agent.get_action_name(action)
                        
                        # 显示决策
                        if action == 0:
                            st.success("✅ RL 建议: 保持当前拓扑")
                        else:
                            st.warning(f"⚠️ RL 建议: {action_name}")
                        
                        st.metric("推荐动作", action_name)
                        st.metric("动作编号", action)
                        
                        # 解释
                        if hasattr(agent, 'interpret_action'):
                            st.markdown("**Qwen LLM 决策解释:**")
                            explanation = agent.interpret_action(action, grid_state)
                            st.info(explanation)
                        
                        # 安全评估
                        safe_to_disconnect = max_load < 0.8
                        st.markdown("**维护可行性评估:**")
                        if safe_to_disconnect:
                            st.success("✅ 当前可安全进行维护操作")
                        else:
                            st.error("❌ 负荷过高，建议先进行负荷转移")
                        
                    except Exception as e:
                        st.error(f"RL 决策失败: {e}")


# ============================================================================
# Tab 4: 系统信息
# ============================================================================

with tab4:
    st.header("📊 系统信息")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("环境信息")
        
        info_data = {
            "项目": "PowerNexus",
            "版本": "1.0.0",
            "Python": sys.version.split()[0],
            "PyTorch": TORCH_VERSION,
            "CUDA": "可用" if CUDA_AVAILABLE else "不可用",
            "模式": "Mock" if use_mock else "生产",
        }
        
        for key, value in info_data.items():
            st.markdown(f"**{key}:** {value}")
        
        st.markdown("---")
        
        st.subheader("模型配置")
        
        model_info = {
            "视觉模型": "Qwen/Qwen2.5-VL-7B-Instruct",
            "文本模型": "Qwen/Qwen2.5-7B-Instruct",
            "嵌入模型": "sentence-transformers/all-MiniLM-L6-v2",
            "RL 算法": "PPO (Stable-Baselines3)",
            "量化": "4-bit (BitsAndBytes)" if use_4bit else "FP16",
        }
        
        for key, value in model_info.items():
            st.markdown(f"**{key}:** `{value}`")
    
    with col2:
        st.subheader("系统架构")
        
        st.markdown("""
        ```
        ┌─────────────────────────────────────┐
        │           PowerNexus 架构           │
        ├─────────────────────────────────────┤
        │                                     │
        │   ┌─────────┐   ┌─────────┐        │
        │   │ Qwen-VL │   │  Qwen   │        │
        │   │  视觉   │   │  文本   │        │
        │   └────┬────┘   └────┬────┘        │
        │        │             │             │
        │   ┌────▼─────────────▼────┐        │
        │   │      LLM 引擎         │        │
        │   └───────────┬───────────┘        │
        │               │                    │
        │   ┌───────────▼───────────┐        │
        │   │      智能体工作流      │        │
        │   │  See→Think→Decide→Act │        │
        │   └───────────┬───────────┘        │
        │               │                    │
        │ ┌─────────────▼─────────────┐      │
        │ │                           │      │
        │ │  ┌─────┐ ┌─────┐ ┌─────┐ │      │
        │ │  │感知 │ │ RAG │ │ RL  │ │      │
        │ │  └─────┘ └─────┘ └─────┘ │      │
        │ │                           │      │
        │ └───────────────────────────┘      │
        │                                     │
        └─────────────────────────────────────┘
        ```
        """)
        
        st.markdown("---")
        
        # 快速操作
        st.subheader("快速操作")
        
        if st.button("🔄 刷新系统状态"):
            st.rerun()
        
        if st.button("🗑️ 清除缓存"):
            st.cache_data.clear()
            st.success("缓存已清除")


# ============================================================================
# 页脚
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>PowerNexus © 2024 | Powered by Qwen2.5 | 
    <a href="https://github.com/QwenLM/Qwen2.5" target="_blank">Qwen2.5</a></p>
</div>
""", unsafe_allow_html=True)
