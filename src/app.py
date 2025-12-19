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
import logging

try:
    from config.settings import config as global_config
except ImportError:
    global_config = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 自动文档摄入
# ============================================================================

@st.cache_resource(show_spinner="正在检查并摄入知识库文档...")
def auto_ingest_documents(manuals_dir: str = "data/manuals", use_mock: bool = True):
    """
    自动摄入 data/manuals 目录下的文档到知识库
    
    使用 Streamlit cache_resource 确保只在启动时执行一次
    
    Args:
        manuals_dir: 文档目录路径
        use_mock: 是否使用 Mock 嵌入模型
        
    Returns:
        dict: 摄入统计信息
    """
    manuals_path = PROJECT_ROOT / manuals_dir
    
    if not manuals_path.exists():
        return {"status": "skip", "message": f"目录不存在: {manuals_dir}"}
    
    # 检查是否有文档需要摄入
    supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
    files = [f for f in manuals_path.rglob('*') 
             if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not files:
        return {"status": "skip", "message": "目录为空，无需摄入"}
    
    try:
        from src.rag.ingest import DocumentIngestor, IngestConfig
        
        # 创建摄入器
        config = IngestConfig()
        ingestor = DocumentIngestor(config=config, use_mock=use_mock)
        
        # 获取当前知识库文档数
        initial_count = ingestor.get_stats().get('total_documents', 0)
        
        # 如果知识库已有文档，跳过摄入
        if initial_count > len(files):
            return {
                "status": "exists", 
                "message": f"知识库已有 {initial_count} 个文档块",
                "count": initial_count
            }
        
        # 摄入目录
        total_chunks = ingestor.ingest_directory(str(manuals_path))
        
        return {
            "status": "success",
            "message": f"成功摄入 {len(files)} 个文件，共 {total_chunks} 个块",
            "files": len(files),
            "chunks": total_chunks
        }
        
    except Exception as e:
        return {"status": "error", "message": f"摄入失败: {str(e)}"}


# 执行自动摄入 (仅在 RAG 模块可用时)
def init_knowledge_base():
    """初始化知识库并自动摄入文档"""
    try:
        # 获取 Mock 模式设置
        use_mock = st.session_state.get('use_mock', True)
        result = auto_ingest_documents(use_mock=use_mock)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
    
    # 优先使用配置中的 Mock 设置
    default_mock = True
    if global_config:
        default_mock = global_config.app.use_mock

    use_mock = st.checkbox(
        "Mock 模式 (测试)",
        value=default_mock,
        help="使用模拟模型进行测试，无需加载实际模型"
    )
    
    # 优先使用配置中的文字回答设置
    default_use_llm = True
    if global_config:
        default_use_llm = getattr(global_config.qwen_llm, 'use_for_text', True)

    use_llm_for_text = st.checkbox(
        "文字回答使用 LLM",
        value=default_use_llm,
        help="如果关闭，文字回答也将使用 Qwen-VL 模型"
    )
    
    # 显示当前使用的模型
    st.markdown("---")
    st.markdown("**🤖 模型信息:**")
    if global_config:
        st.caption(f"VL: {global_config.qwen_vl.model_name}")
        st.caption(f"LLM: {global_config.qwen_llm.model_name}")
        st.caption(f"Embedding: {global_config.rag.embedding_model}")
    else:
        st.caption("配置加载失败")
    
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
    
    # 知识库自动摄入状态
    st.markdown("---")
    st.markdown("**📚 知识库:**")
    
    if RAG_AVAILABLE:
        # 执行自动摄入
        ingest_result = auto_ingest_documents(use_mock=use_mock)
        
        if ingest_result["status"] == "success":
            st.success(f"✅ {ingest_result.get('chunks', 0)} 块")
        elif ingest_result["status"] == "exists":
            st.info(f"📄 {ingest_result.get('count', 0)} 块")
        elif ingest_result["status"] == "skip":
            st.caption(ingest_result["message"])
        else:
            st.warning(f"⚠️ {ingest_result['message'][:30]}...")
        
        # 手动刷新按钮
        if st.button("🔄 重新摄入", help="清除缓存并重新摄入文档"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.caption("RAG 模块未加载")
    
    st.markdown("---")
    st.markdown("**✍️ 作者:** TanXiaoke")
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
        
        if st.button("使用示例图像"):
            st.session_state['selected_image_path'] = example_images[selected_example]
            st.session_state['image_source'] = 'example'
        
        # 确定要分析的图像
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
                st.session_state['selected_image_path'] = image_path
                st.session_state['image_source'] = 'upload'
        elif 'selected_image_path' in st.session_state:
            image_path = st.session_state['selected_image_path']
            if Path(image_path).exists() and PIL_AVAILABLE:
                image_to_analyze = Image.open(image_path)
        
        if image_to_analyze:
            st.image(image_to_analyze, caption="待分析图像", use_container_width=True)
        elif 'image_source' not in st.session_state:
            st.info("请上传图像或选择示例图像")
    
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
                        
                        # 显示实际使用的模式
                        if detector._use_mock:
                            st.caption("🔧 当前使用: Mock 模型")
                        else:
                            st.caption("🚀 当前使用: Qwen2.5-VL 真实模型")
                        
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
                        
                        # 🛠️ RAG + LLM 技术标准指导
                        if result.defect_detected and RAG_AVAILABLE:
                            st.markdown("---")
                            st.subheader("🛠️ 技术标准指导 (RAG + LLM)")
                            
                            with st.spinner("正在检索技术标准并生成建议..."):
                                try:
                                    # 延迟导入以避免冲突
                                    from src.rag.retriever import KnowledgeBase
                                    kb = KnowledgeBase(use_mock=use_mock)
                                    
                                    # 使用缺陷类型和描述作为查询
                                    query_text = f"针对{result.defect_type}（{result.description}），相关的电力技术标准和处理建议是什么？"
                                    
                                    rag_result = kb.query_and_synthesize(query_text, top_k=3)
                                    
                                    if rag_result.has_results:
                                        st.success("✅ 查找到相关标准建议：")
                                        st.markdown(rag_result.synthesized_answer)
                                        
                                        with st.expander("查看参考标准原文"):
                                            st.write(rag_result.formatted_context)
                                    else:
                                        st.warning("未在知识库中找到直接匹配的标准。")
                                        
                                except Exception as rag_err:
                                    st.error(f"标准检索识别失败: {rag_err}")
                        
                        # JSON 输出
                        with st.expander("查看 JSON 输出"):
                            st.json(result.to_dict())
                        
                        detector.close()
                        
                    except Exception as e:
                        st.error(f"分析失败: {e}")
                        import traceback
                        with st.expander("错误详情"):
                            st.code(traceback.format_exc())


# ============================================================================
# Tab 2: 知识检索
# ============================================================================

with tab2:
    st.header("📚 RAG 知识检索")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("查询问题")
        
        # 知识库数据预览
        if RAG_AVAILABLE:
            with st.expander("📋 查看知识库数据", expanded=False):
                try:
                    from src.rag.retriever import KnowledgeBase
                    kb_preview = KnowledgeBase(use_mock=use_mock)
                    kb_stats = kb_preview.get_stats()
                    doc_count = kb_stats.get('total_documents', 0)
                    
                    st.markdown(f"**文档块数量:** {doc_count}")
                    st.markdown(f"**集合名称:** {kb_stats.get('collection_name', 'N/A')}")
                    
                    # 显示样本数据
                    if doc_count > 0 and hasattr(kb_preview, '_vector_store'):
                        try:
                            samples = kb_preview._vector_store._collection.peek(5)
                            if samples and 'documents' in samples:
                                st.markdown("**样本文本块:**")
                                for i, doc in enumerate(samples['documents'][:3]):
                                    preview = doc[:150] + "..." if len(doc) > 150 else doc
                                    st.text_area(f"块 {i+1}", preview, height=80, key=f"kb_sample_{i}")
                        except:
                            pass
                except Exception as e:
                    st.caption(f"无法加载知识库: {e}")
            
            st.markdown("---")
            
            # 更新知识库功能
            if st.button("🔄 更新知识库", help="重新摄入 /data/manuals 目录下的文档", use_container_width=True):
                data_dir = os.path.join(global_config.project_root, "data", "manuals")
                if not os.path.exists(data_dir):
                    st.error(f"目录不存在: {data_dir}")
                else:
                    with st.spinner("正在更新知识库，这可能需要一些时间..."):
                        try:
                            # 延迟导入以避免循环依赖
                            from src.rag.ingest import DocumentIngestor
                            
                            ingestor = DocumentIngestor(use_mock=use_mock)
                            
                            # 先清空已有知识库以进行全面更新
                            st.info("正在清空旧数据...")
                            ingestor.vector_store.clear()
                            
                            count = ingestor.ingest_directory(data_dir)
                            
                            if count > 0:
                                st.success(f"✅ 更新成功！共摄入 {count} 个文档块。")
                                # 强制刷新页面以更新统计信息
                                st.rerun()
                            else:
                                st.warning("未找到新文档或摄入失败。")
                                
                        except Exception as e:
                            st.error(f"更新失败: {e}")
                            logger.error(f"知识库更新失败: {e}")
        
        st.markdown("---")
        
        # 预设问题
        preset_questions = [
            "变压器绝缘电阻测量的环境要求是什么？",
            "绝缘子裂纹应该如何处理？",
            "高温环境下设备运行有什么注意事项？",
            "负荷转移操作的流程是什么？",
            "停送电操作顺序是什么？",
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
                        # 使用摄入了真实数据的知识库
                        from src.rag.retriever import KnowledgeBase
                        kb = KnowledgeBase(use_mock=use_mock)
                        
                        # 显示知识库状态
                        kb_stats = kb.get_stats()
                        doc_count = kb_stats.get('total_documents', 0)
                        if kb._use_mock:
                            st.caption(f"🔧 Mock 模式 | 文档: {doc_count} 块")
                        else:
                            st.caption(f"🚀 真实嵌入模型 | 文档: {doc_count} 块")
                        
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
                        import traceback
                        with st.expander("错误详情"):
                            st.code(traceback.format_exc())


# ============================================================================
# Tab 3: RL 优化
# ============================================================================

with tab3:
    st.header("🤖 RL 优化决策")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("电网状态输入")
        
        # 数据源选择
        data_source = st.radio(
            "数据来源",
            ["从文件加载 (grid_states.npz)", "手动设置负荷水平"],
            horizontal=True
        )
        
        import numpy as np
        
        # 初始化变量
        line_loads = None
        max_load = 0.5
        current_state_index = 0
        
        if data_source == "从文件加载 (grid_states.npz)":
            # 显示 IEEE 系统信息
            st.markdown("**📊 IEEE 14-Bus 系统**")
            st.caption("5 发电机 | 11 负荷 | 20 线路 | 57 拓扑元素")
            
            # 从 grid_states.npz 加载数据
            data_path = "data/grid_states.npz"
            
            try:
                if Path(data_path).exists():
                    data = np.load(data_path, allow_pickle=True)
                    n_samples = int(data['n_samples'])
                    
                    st.success(f"✅ 已加载 {n_samples} 个电网状态")
                    
                    # 选择状态索引
                    current_state_index = st.slider(
                        "选择状态快照",
                        0, n_samples - 1, 0,
                        help="从预生成的电网状态中选择一个"
                    )
                    
                    # 读取当前状态的数据
                    line_loads = data['rho'][current_state_index]
                    gen_p = data['gen_p'][current_state_index]
                    gen_v = data['gen_v'][current_state_index]
                    load_p = data['load_p'][current_state_index]
                    load_q = data['load_q'][current_state_index]
                    topo_vect = data['topo_vect'][current_state_index]
                    scenario = str(data['scenarios'][current_state_index])
                    
                    max_load = float(line_loads.max())
                    
                    # 显示场景信息
                    st.info(f"📊 场景: **{scenario}** | 状态索引: {current_state_index}")
                    
                else:
                    st.warning(f"⚠️ 数据文件不存在: {data_path}")
                    st.markdown("请先运行以下命令生成数据:")
                    st.code("python tools/simulate_grid_state.py -n 100 -s mixed -o data/grid_states.npz")
                    line_loads = None
                    
            except Exception as e:
                st.error(f"加载数据失败: {e}")
                line_loads = None
        
        else:
            # 手动设置每条线路负载率
            st.markdown("**📊 IEEE 14-Bus 系统**")
            st.caption("5 发电机 | 11 负荷 | 20 线路 | 57 拓扑元素")
            
            st.markdown("---")
            st.markdown("**线路负载率手动设置:**")
            
            # 预设模式选择
            preset_mode = st.selectbox(
                "预设负荷模式",
                ["自定义", "正常运行", "高峰负荷", "局部过载", "紧急状态"]
            )
            
            # 根据预设初始化
            if preset_mode == "正常运行":
                default_loads = [0.4 + i * 0.02 for i in range(20)]
            elif preset_mode == "高峰负荷":
                default_loads = [0.7 + (i % 5) * 0.05 for i in range(20)]
            elif preset_mode == "局部过载":
                default_loads = [0.5] * 15 + [0.95, 0.98, 1.02, 0.88, 0.92]
            elif preset_mode == "紧急状态":
                default_loads = [0.8 + (i % 4) * 0.1 for i in range(20)]
            else:
                # 自定义模式，使用 session_state 存储的值或默认值
                if 'manual_line_loads' not in st.session_state:
                    st.session_state['manual_line_loads'] = [0.5] * 20
                default_loads = st.session_state['manual_line_loads']
            
            # 创建线路负载率滑块
            line_loads = []
            
            # 分两列显示 10 条线路
            col_l1, col_l2 = st.columns(2)
            
            with col_l1:
                st.markdown("**L1-L10:**")
                for i in range(10):
                    val = st.slider(
                        f"L{i+1}", 0.0, 1.2, 
                        float(default_loads[i]) if i < len(default_loads) else 0.5,
                        0.05,
                        key=f"line_{i}",
                        help=f"线路 {i+1} 负载率"
                    )
                    line_loads.append(val)
            
            with col_l2:
                st.markdown("**L11-L20:**")
                for i in range(10, 20):
                    val = st.slider(
                        f"L{i+1}", 0.0, 1.2,
                        float(default_loads[i]) if i < len(default_loads) else 0.5,
                        0.05,
                        key=f"line_{i}",
                        help=f"线路 {i+1} 负载率"
                    )
                    line_loads.append(val)
            
            line_loads = np.array(line_loads)
            st.session_state['manual_line_loads'] = line_loads.tolist()
            max_load = float(line_loads.max())
        
        # 显示线路负载率
        if line_loads is not None:
            st.markdown("---")
            st.markdown(f"**📈 线路负载率 ({len(line_loads)} 条线路):**")
            
            # 统计信息
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("最大负载率", f"{line_loads.max():.1%}")
            with col_stat2:
                st.metric("平均负载率", f"{line_loads.mean():.1%}")
            with col_stat3:
                overload_count = int((line_loads > 0.95).sum())
                st.metric("过载线路", f"{overload_count} 条", 
                         delta=f"{overload_count}" if overload_count > 0 else None,
                         delta_color="inverse")
            
            # 显示前 10 条线路的进度条
            st.markdown("**线路状态 (前10条):**")
            for i, load in enumerate(line_loads[:10]):
                # 根据负载率选择颜色提示
                if load > 0.95:
                    status = "🔴"
                elif load > 0.8:
                    status = "🟠"
                elif load > 0.6:
                    status = "🟡"
                else:
                    status = "🟢"
                
                # 确保 progress 值在 0-1 之间
                progress_val = min(1.0, max(0.0, float(load)))
                st.progress(progress_val, text=f"{status} L{i+1}: {load:.1%}")
            
            # 如果有更多线路，显示可展开区域
            if len(line_loads) > 10:
                with st.expander(f"查看更多线路 (L11 - L{len(line_loads)})"):
                    for i, load in enumerate(line_loads[10:], start=10):
                        if load > 0.95:
                            status = "🔴"
                        elif load > 0.8:
                            status = "🟠"
                        elif load > 0.6:
                            status = "🟡"
                        else:
                            status = "🟢"
                        progress_val = min(1.0, max(0.0, float(load)))
                        st.progress(progress_val, text=f"{status} L{i+1}: {load:.1%}")
        
        # 状态描述
        if line_loads is not None:
            overload_lines = [i+1 for i, l in enumerate(line_loads) if l > 0.95]
            if overload_lines:
                grid_state = f"最大负载率 {max_load:.0%}，过载线路: {overload_lines}"
            else:
                grid_state = f"最大负载率 {max_load:.0%}，所有线路运行正常"
        else:
            grid_state = "无数据"
        st.text_area("电网状态描述", grid_state, height=80)
    
    with col2:
        st.subheader("RL 决策结果")
        
        # 模型加载部分
        st.markdown("**📦 模型选择**")
        
        # 检查本地模型
        default_model_path = "models/rl/ppo_grid.zip"
        model_exists = Path(default_model_path).exists()
        
        model_source = st.radio(
            "模型来源",
            ["使用默认/Mock 模型", "加载本地训练模型", "上传模型文件"],
            horizontal=True,
            help="选择要使用的 PPO 模型"
        )
        
        model_path = None
        
        if model_source == "加载本地训练模型":
            if model_exists:
                st.success(f"✅ 找到本地模型: `{default_model_path}`")
                model_path = default_model_path
                
                # 显示模型信息
                model_stat = Path(default_model_path).stat()
                model_size_kb = model_stat.st_size / 1024
                from datetime import datetime as dt
                model_time = dt.fromtimestamp(model_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"📊 大小: {model_size_kb:.1f} KB | 修改时间: {model_time}")
            else:
                st.warning(f"⚠️ 本地模型不存在: `{default_model_path}`")
                st.markdown("请先运行训练：")
                st.code("python tools/train_ppo.py --timesteps 2048", language="bash")
                
                # 允许指定其他路径
                custom_path = st.text_input("或输入自定义模型路径", placeholder="models/rl/your_model.zip")
                if custom_path and Path(custom_path).exists():
                    model_path = custom_path
                    st.success(f"✅ 找到模型: `{custom_path}`")
                    
        elif model_source == "上传模型文件":
            uploaded_model = st.file_uploader(
                "上传 PPO 模型 (.zip)",
                type=["zip"],
                help="上传 Stable-Baselines3 训练的 PPO 模型"
            )
            if uploaded_model:
                # 保存上传的模型
                upload_path = Path("models/rl/uploaded_model.zip")
                upload_path.parent.mkdir(parents=True, exist_ok=True)
                with open(upload_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                model_path = str(upload_path)
                st.success(f"✅ 模型已上传: `{model_path}`")
        
        st.markdown("---")
        
        if st.button("🎯 执行 RL 优化", type="primary", use_container_width=True):
            if not RL_AVAILABLE:
                st.error("RL 模块未加载")
            elif line_loads is None:
                st.warning("请先加载电网状态数据")
            else:
                with st.spinner("PPO 智能体正在分析..."):
                    try:
                        # 根据选择创建智能体
                        if model_path and Path(model_path).exists():
                            st.caption(f"🔧 加载模型: `{model_path}`")
                            from src.rl_engine import PPO_Agent, PowerGridEnvConfig
                            env_config = PowerGridEnvConfig(use_mock=use_mock)
                            agent = PPO_Agent(
                                env_config=env_config,
                                model_path=model_path,
                                use_mock=use_mock,
                                enable_llm=not use_mock,
                            )
                        else:
                            st.caption("🔧 使用默认模型")
                            agent = create_ppo_agent(use_mock=use_mock)
                        
                        # 构造完整观测向量
                        if data_source == "从文件加载 (grid_states.npz)" and 'gen_p' in dir():
                            # 使用从文件加载的完整数据
                            obs = np.concatenate([
                                line_loads.astype(np.float32),      # rho (20)
                                gen_p.astype(np.float32),           # gen_p (5)
                                gen_v.astype(np.float32),           # gen_v (5)
                                load_p.astype(np.float32),          # load_p (11)
                                load_q.astype(np.float32),          # load_q (11)
                                topo_vect.astype(np.float32),       # topo_vect (57)
                            ])
                            st.caption(f"📐 观测向量维度: {obs.shape[0]}")
                        else:
                            # 手动模式: 构造模拟观测
                            obs = np.concatenate([
                                line_loads.astype(np.float32),
                                np.array([50.0, 30.0, 20.0, 25.0, 15.0], dtype=np.float32),  # gen_p
                                np.ones(5, dtype=np.float32),                                 # gen_v
                                np.random.uniform(10, 40, 11).astype(np.float32),            # load_p
                                np.random.uniform(3, 15, 11).astype(np.float32),             # load_q
                                np.ones(57, dtype=np.float32),                                # topo_vect
                            ])
                        
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
                        
                        # 显示详细观测信息
                        with st.expander("查看详细观测数据"):
                            st.json({
                                "rho_max": float(line_loads.max()),
                                "rho_mean": float(line_loads.mean()),
                                "overload_lines": int((line_loads > 0.95).sum()),
                                "obs_dim": len(obs),
                            })
                        
                    except Exception as e:
                        st.error(f"RL 决策失败: {e}")
                        import traceback
                        with st.expander("查看错误详情"):
                            st.code(traceback.format_exc())
        
        # PPO 训练部分
        st.markdown("---")
        st.subheader("🎓 PPO 模型训练")
        
        with st.expander("训练参数设置", expanded=False):
            train_col1, train_col2 = st.columns(2)
            
            with train_col1:
                train_timesteps = st.number_input(
                    "训练步数",
                    min_value=100,
                    max_value=1_000_000,
                    value=10_000,
                    step=1000,
                    help="更多步数 = 更好的模型，但训练时间更长"
                )
                
                learning_rate = st.select_slider(
                    "学习率",
                    options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                    value=3e-4,
                    format_func=lambda x: f"{x:.0e}"
                )
            
            with train_col2:
                save_path = st.text_input(
                    "模型保存路径",
                    value="models/rl/ppo_grid.zip",
                    help="训练完成后模型保存位置"
                )
                
                eval_episodes = st.slider(
                    "评估 Episode 数",
                    min_value=0,
                    max_value=20,
                    value=5,
                    help="训练后评估模型性能"
                )
        
        if st.button("🚀 开始训练 PPO", type="secondary", use_container_width=True):
            if not RL_AVAILABLE:
                st.error("RL 模块未加载")
            else:
                # 导入训练配置
                from src.rl_engine import PPOAgentConfig, TrainingConfig, PPO_Agent, PowerGridEnvConfig
                
                # 创建进度显示
                progress_bar = st.progress(0, text="准备训练...")
                status_text = st.empty()
                
                try:
                    # 配置
                    from pathlib import Path
                    env_config = PowerGridEnvConfig(use_mock=use_mock)
                    agent_config = PPOAgentConfig(learning_rate=learning_rate)
                    
                    # 设置保存路径
                    model_dir = Path(save_path).parent
                    training_config = TrainingConfig(
                        total_timesteps=train_timesteps,
                        save_path=str(model_dir / "checkpoints"),
                        best_model_save_path=str(model_dir / "best_model"),
                        log_path=str(model_dir / "logs"),
                    )
                    
                    status_text.text("创建 PPO 智能体...")
                    progress_bar.progress(10, text="创建智能体...")
                    
                    agent = PPO_Agent(
                        env_config=env_config,
                        agent_config=agent_config,
                        use_mock=use_mock,
                        enable_llm=False,
                    )
                    
                    status_text.text(f"开始训练 ({train_timesteps:,} 步)...")
                    progress_bar.progress(20, text="训练中...")
                    
                    # 训练
                    with st.spinner(f"PPO 训练中 ({train_timesteps:,} 步)..."):
                        result = agent.train_model(
                            total_timesteps=train_timesteps,
                            training_config=training_config,
                            progress_bar=False,
                        )
                    
                    progress_bar.progress(80, text="保存模型...")
                    
                    if result["status"] == "success":
                        # 保存模型
                        agent.save(save_path)
                        progress_bar.progress(90, text="评估模型...")
                        
                        # 评估
                        if eval_episodes > 0:
                            eval_result = agent.evaluate(n_eval_episodes=eval_episodes)
                            
                            progress_bar.progress(100, text="完成!")
                            st.success(f"✅ 训练完成! 模型已保存到 `{save_path}`")
                            
                            # 显示评估结果
                            eval_col1, eval_col2, eval_col3 = st.columns(3)
                            with eval_col1:
                                st.metric("平均奖励", f"{eval_result['mean_reward']:.2f}")
                            with eval_col2:
                                st.metric("奖励标准差", f"±{eval_result['std_reward']:.2f}")
                            with eval_col3:
                                st.metric("平均步数", f"{eval_result['mean_length']:.1f}")
                        else:
                            progress_bar.progress(100, text="完成!")
                            st.success(f"✅ 训练完成! 模型已保存到 `{save_path}`")
                    else:
                        st.error(f"训练失败: {result.get('error', '未知错误')}")
                        
                except Exception as e:
                    st.error(f"训练失败: {e}")
                    import traceback
                    with st.expander("查看错误详情"):
                        st.code(traceback.format_exc())
        
        # 提示使用命令行训练
        st.caption("💡 提示: 对于长时间训练，建议使用命令行: `python tools/train_ppo.py --timesteps 100000`")


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
            "精度": "FP16",
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
    <p>PowerNexus © 2025 | Powered by Qwen2.5 | 
    <a href="https://github.com/QwenLM/Qwen2.5" target="_blank">Qwen2.5</a></p>
</div>
""", unsafe_allow_html=True)
