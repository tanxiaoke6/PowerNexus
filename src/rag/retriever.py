# -*- coding: utf-8 -*-
"""
PowerNexus - 知识库检索模块

本模块提供电力技术标准知识库的语义检索功能。
支持查询相关技术文档并使用 Qwen LLM 合成答案。

特性:
- 语义检索电力技术标准
- Qwen LLM 答案合成
- 格式化 LLM 提示模板
- 支持多种检索策略
- 完整的 Mock 模式支持

作者: PowerNexus Team
日期: 2025-12-18
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from string import Template

# 配置日志
logger = logging.getLogger(__name__)

# 导入本地模块
from .ingest import (
    EmbeddingModel,
    VectorStore,
    IngestConfig,
    DocumentIngestor,
    create_sample_documents,
    CHROMADB_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)

# 导入 Qwen LLM 引擎
QWEN_LLM_AVAILABLE = False
try:
    from src.utils.llm_engine import LLMEngine, create_llm_engine, get_llm_engine
    QWEN_LLM_AVAILABLE = True
    logger.info("Qwen LLM 引擎已导入")
except ImportError:
    logger.warning("Qwen LLM 引擎未导入，答案合成功能不可用")
    LLMEngine = None

try:
    import numpy as np
except ImportError:
    import numpy as np


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class RetrieverConfig:
    """
    检索器配置类
    
    Attributes:
        collection_name: ChromaDB 集合名称
        persist_directory: 向量库持久化目录
        embedding_model: 嵌入模型名称
        top_k: 默认返回结果数量
        score_threshold: 相似度阈值 (0-1, 越低越严格)
        use_reranking: 是否使用重排序
    """
    collection_name: str = "power_grid_standards"
    persist_directory: str = "./data/vector_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    score_threshold: float = 0.5
    use_reranking: bool = False


# ============================================================================
# 提示模板
# ============================================================================

class PromptTemplates:
    """
    LLM 提示模板集合
    
    提供各种场景下的提示模板。
    """
    
    # 基础 RAG 问答模板
    RAG_QA_TEMPLATE = """你是一名电力系统专家。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请明确说明。

## 参考资料
{context}

## 用户问题
{question}

## 回答要求
1. 基于参考资料进行回答
2. 如引用具体标准，请标注来源
3. 回答要专业、准确、简洁

请回答："""

    # 技术标准查询模板
    STANDARD_QUERY_TEMPLATE = """请根据以下电力技术标准文档回答问题。

### 相关标准条款
{context}

### 查询问题
{question}

### 注意事项
- 引用标准时需注明标准编号
- 如涉及数值要求，需给出具体数值
- 如标准中未明确规定，请说明

回答："""

    # 故障诊断模板
    FAULT_DIAGNOSIS_TEMPLATE = """你是一名电力设备故障诊断专家。

## 设备信息
{equipment_info}

## 故障现象
{fault_description}

## 相关技术标准
{context}

## 诊断任务
1. 分析可能的故障原因
2. 参考相关标准给出处理建议
3. 列出需要进行的检测项目

请给出诊断结论："""

    # 调度决策支持模板
    DISPATCH_SUPPORT_TEMPLATE = """你是一名电网调度决策支持系统。

## 当前电网状态
{grid_state}

## 异常告警
{alerts}

## 相关技术规范
{context}

## 决策任务
根据技术规范，分析当前情况并给出调度建议。

调度建议："""

    @classmethod
    def format_context(
        cls,
        documents: List[Dict[str, Any]],
        include_source: bool = True,
        max_length: Optional[int] = None,
    ) -> str:
        """
        格式化检索结果为上下文字符串
        
        Args:
            documents: 检索到的文档列表
            include_source: 是否包含来源信息
            max_length: 最大长度限制
            
        Returns:
            格式化的上下文字符串
        """
        if not documents:
            return "未找到相关参考资料。"
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if include_source:
                source = metadata.get("source", metadata.get("filename", "未知来源"))
                chunk_info = f"(块 {metadata.get('chunk_index', '?')}/{metadata.get('total_chunks', '?')})"
                header = f"### 参考 {i} | 来源: {source} {chunk_info}"
            else:
                header = f"### 参考 {i}"
            
            context_parts.append(f"{header}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # 限制长度
        if max_length and len(context) > max_length:
            context = context[:max_length] + "\n\n[... 内容已截断 ...]"
        
        return context
    
    @classmethod
    def build_prompt(
        cls,
        template: str,
        context: str,
        question: str,
        **kwargs,
    ) -> str:
        """
        构建完整提示
        
        Args:
            template: 模板字符串
            context: 上下文内容
            question: 用户问题
            **kwargs: 其他模板变量
            
        Returns:
            完整的提示字符串
        """
        return template.format(
            context=context,
            question=question,
            **kwargs,
        )


# ============================================================================
# 检索结果类
# ============================================================================

@dataclass
class RetrievalResult:
    """
    检索结果封装类
    
    Attributes:
        query: 原始查询
        documents: 检索到的文档列表
        formatted_context: 格式化的上下文
        prompt: 完整的 LLM 提示
        synthesized_answer: Qwen LLM 合成的答案
        metadata: 附加元数据
    """
    query: str
    documents: List[Dict[str, Any]]
    formatted_context: str
    prompt: str
    synthesized_answer: str = ""  # Qwen LLM 生成的答案
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_results(self) -> int:
        """返回结果数量"""
        return len(self.documents)
    
    @property
    def has_results(self) -> bool:
        """是否有检索结果"""
        return len(self.documents) > 0
    
    @property
    def has_answer(self) -> bool:
        """是否有合成答案"""
        return bool(self.synthesized_answer)
    
    def get_top_document(self) -> Optional[Dict[str, Any]]:
        """获取最相关的文档"""
        return self.documents[0] if self.documents else None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "num_results": self.num_results,
            "documents": self.documents,
            "formatted_context": self.formatted_context,
            "prompt": self.prompt,
            "synthesized_answer": self.synthesized_answer,
            "metadata": self.metadata,
        }


# ============================================================================
# 知识库类
# ============================================================================

class KnowledgeBase:
    """
    电力技术标准知识库
    
    提供语义检索和 Qwen LLM 答案合成功能。
    
    Example:
        >>> kb = KnowledgeBase()
        >>> result = kb.query_and_synthesize("变压器绝缘电阻测量要求是什么？")
        >>> print(result.synthesized_answer)  # Qwen 生成的答案
    """
    
    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        use_mock: bool = False,
        enable_llm: bool = True,
    ):
        """
        初始化知识库
        
        Args:
            config: 检索器配置
            use_mock: 是否使用 Mock 模式
            enable_llm: 是否启用 Qwen LLM 答案合成
        """
        self.config = config or RetrieverConfig()
        self._use_mock = use_mock
        self._llm_enabled = enable_llm and QWEN_LLM_AVAILABLE
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel(
            model_name=self.config.embedding_model,
            use_mock=use_mock,
        )
        
        # 初始化向量存储
        self.vector_store = VectorStore(
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
            use_mock=use_mock,
        )
        
        # 初始化 Qwen LLM 引擎
        self._llm_engine = None
        if self._llm_enabled:
            try:
                self._llm_engine = create_llm_engine(use_mock=use_mock)
                logger.info("Qwen LLM 引擎已初始化")
            except Exception as e:
                logger.warning(f"Qwen LLM 初始化失败: {e}")
                self._llm_enabled = False
        
        logger.info(
            f"知识库初始化完成 | "
            f"集合: {self.config.collection_name} | "
            f"Mock: {self._use_mock} | "
            f"LLM: {self._llm_enabled}"
        )
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        template: Optional[str] = None,
        score_threshold: Optional[float] = None,
        include_source: bool = True,
    ) -> RetrievalResult:
        """
        查询知识库
        
        Args:
            question: 用户问题
            top_k: 返回结果数量
            template: 提示模板 (默认使用 RAG_QA_TEMPLATE)
            score_threshold: 相似度阈值
            include_source: 是否在上下文中包含来源
            
        Returns:
            RetrievalResult 对象
        """
        top_k = top_k or self.config.top_k
        template = template or PromptTemplates.RAG_QA_TEMPLATE
        score_threshold = score_threshold or self.config.score_threshold
        
        logger.debug(f"查询: {question[:50]}... (top_k={top_k})")
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed(question)[0]
        
        # 检索相似文档
        documents = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        # 过滤低分文档
        if score_threshold is not None:
            documents = [
                doc for doc in documents
                if doc.get("distance", 1.0) <= (1 - score_threshold)
            ]
        
        # 格式化上下文
        formatted_context = PromptTemplates.format_context(
            documents,
            include_source=include_source,
        )
        
        # 构建提示
        prompt = PromptTemplates.build_prompt(
            template=template,
            context=formatted_context,
            question=question,
        )
        
        # 构建结果
        result = RetrievalResult(
            query=question,
            documents=documents,
            formatted_context=formatted_context,
            prompt=prompt,
            metadata={
                "top_k": top_k,
                "score_threshold": score_threshold,
                "embedding_model": self.config.embedding_model,
            },
        )
        
        logger.info(f"检索完成 | 问题: {question[:30]}... | 结果数: {result.num_results}")
        
        return result
    
    def query_and_synthesize(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_source: bool = True,
    ) -> RetrievalResult:
        """
        检索并使用 Qwen LLM 合成答案
        
        这是推荐的主要查询方法，会自动使用 Qwen LLM 基于检索结果生成答案。
        
        Args:
            question: 用户问题
            top_k: 返回结果数量
            include_source: 是否在上下文中包含来源
            
        Returns:
            包含 synthesized_answer 的 RetrievalResult
        """
        # 先执行检索
        result = self.query(
            question=question,
            top_k=top_k,
            include_source=include_source,
        )
        
        # 使用 Qwen LLM 合成答案
        if self._llm_enabled and self._llm_engine:
            try:
                answer = self._llm_engine.rag_synthesize(
                    question=question,
                    context=result.formatted_context,
                )
                result.synthesized_answer = answer
                result.metadata["llm_synthesized"] = True
                logger.info(f"Qwen LLM 答案合成完成 | 问题: {question[:30]}...")
            except Exception as e:
                logger.warning(f"答案合成失败: {e}")
                result.synthesized_answer = f"[答案合成失败: {e}]"
                result.metadata["llm_error"] = str(e)
        else:
            result.synthesized_answer = "[LLM 未启用，请参考上下文自行分析]"
            result.metadata["llm_synthesized"] = False
        
        return result
    
    def query_for_standard(
        self,
        question: str,
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        查询技术标准（使用专用模板）
        
        Args:
            question: 技术问题
            top_k: 返回结果数量
            
        Returns:
            RetrievalResult 对象
        """
        return self.query(
            question=question,
            top_k=top_k,
            template=PromptTemplates.STANDARD_QUERY_TEMPLATE,
        )
    
    def query_for_diagnosis(
        self,
        equipment_info: str,
        fault_description: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        查询故障诊断相关标准
        
        Args:
            equipment_info: 设备信息
            fault_description: 故障描述
            top_k: 返回结果数量
            
        Returns:
            RetrievalResult 对象
        """
        # 组合查询
        combined_query = f"{equipment_info} {fault_description}"
        
        # 检索
        query_embedding = self.embedding_model.embed(combined_query)[0]
        documents = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        # 格式化上下文
        formatted_context = PromptTemplates.format_context(documents)
        
        # 构建诊断提示
        prompt = PromptTemplates.FAULT_DIAGNOSIS_TEMPLATE.format(
            equipment_info=equipment_info,
            fault_description=fault_description,
            context=formatted_context,
        )
        
        return RetrievalResult(
            query=combined_query,
            documents=documents,
            formatted_context=formatted_context,
            prompt=prompt,
            metadata={"query_type": "diagnosis"},
        )
    
    def query_for_dispatch(
        self,
        grid_state: str,
        alerts: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        查询调度决策相关标准
        
        Args:
            grid_state: 电网状态描述
            alerts: 告警信息
            top_k: 返回结果数量
            
        Returns:
            RetrievalResult 对象
        """
        # 组合查询
        combined_query = f"电网调度 {grid_state} {alerts}"
        
        # 检索
        query_embedding = self.embedding_model.embed(combined_query)[0]
        documents = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        # 格式化上下文
        formatted_context = PromptTemplates.format_context(documents)
        
        # 构建调度提示
        prompt = PromptTemplates.DISPATCH_SUPPORT_TEMPLATE.format(
            grid_state=grid_state,
            alerts=alerts,
            context=formatted_context,
        )
        
        return RetrievalResult(
            query=combined_query,
            documents=documents,
            formatted_context=formatted_context,
            prompt=prompt,
            metadata={"query_type": "dispatch"},
        )
    
    def get_similar_documents(
        self,
        text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        获取与给定文本相似的文档（不构建提示）
        
        Args:
            text: 输入文本
            top_k: 返回数量
            
        Returns:
            相似文档列表
        """
        embedding = self.embedding_model.embed(text)[0]
        return self.vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> int:
        """
        向知识库添加文档
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            添加的文档数量
        """
        if not texts:
            return 0
        
        # 生成嵌入
        embeddings = self.embedding_model.embed(texts)
        
        # 添加到存储
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info(f"添加 {len(texts)} 个文档到知识库")
        return len(texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "collection_name": self.config.collection_name,
            "total_documents": self.vector_store.count(),
            "embedding_model": self.config.embedding_model,
            "top_k_default": self.config.top_k,
            "use_mock": self._use_mock,
        }
    
    def clear(self):
        """清空知识库"""
        self.vector_store.clear()
        logger.info("知识库已清空")


# ============================================================================
# 便捷函数
# ============================================================================

def create_knowledge_base(
    collection_name: str = "power_grid_standards",
    persist_directory: str = "./data/vector_db",
    use_mock: bool = False,
    **kwargs,
) -> KnowledgeBase:
    """
    创建知识库的便捷函数
    
    Args:
        collection_name: 集合名称
        persist_directory: 持久化目录
        use_mock: 是否使用 Mock 模式
        **kwargs: 其他配置参数
        
    Returns:
        KnowledgeBase 实例
    """
    config = RetrieverConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs,
    )
    return KnowledgeBase(config=config, use_mock=use_mock)


def init_knowledge_base_with_samples(use_mock: bool = False) -> KnowledgeBase:
    """
    初始化带示例数据的知识库
    
    Args:
        use_mock: 是否使用 Mock 模式
        
    Returns:
        初始化好的 KnowledgeBase
    """
    # 创建知识库
    kb = create_knowledge_base(use_mock=use_mock)
    
    # 加载示例文档
    sample_docs = create_sample_documents()
    
    # 添加到知识库
    metadatas = [
        {"source": "power_grid_standards", "doc_index": i}
        for i in range(len(sample_docs))
    ]
    kb.add_documents(sample_docs, metadatas)
    
    logger.info(f"知识库初始化完成，加载了 {len(sample_docs)} 个示例文档")
    
    return kb


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
    print("PowerNexus - 知识库检索模块测试")
    print("=" * 60)
    
    # 创建并初始化知识库
    kb = init_knowledge_base_with_samples(use_mock=True)
    
    print(f"\n知识库统计: {kb.get_stats()}")
    
    # 测试查询 1: 基础问答
    print("\n" + "=" * 40)
    print("测试 1: 基础技术问答")
    print("=" * 40)
    
    result = kb.query("变压器绝缘电阻测量的环境要求是什么？")
    print(f"查询: {result.query}")
    print(f"结果数: {result.num_results}")
    print(f"\n格式化上下文:\n{result.formatted_context[:500]}...")
    print(f"\n完整提示长度: {len(result.prompt)} 字符")
    
    # 测试查询 2: 技术标准
    print("\n" + "=" * 40)
    print("测试 2: 技术标准查询")
    print("=" * 40)
    
    result = kb.query_for_standard("绝缘子的试验项目有哪些？")
    print(f"查询: {result.query}")
    print(f"结果数: {result.num_results}")
    
    # 测试查询 3: 故障诊断
    print("\n" + "=" * 40)
    print("测试 3: 故障诊断查询")
    print("=" * 40)
    
    result = kb.query_for_diagnosis(
        equipment_info="110kV 变压器 型号: SZ11-50000/110",
        fault_description="绝缘油色谱分析显示乙炔含量异常升高",
    )
    print(f"结果数: {result.num_results}")
    print(f"\n诊断提示 (前500字):\n{result.prompt[:500]}...")
    
    # 测试查询 4: 调度决策
    print("\n" + "=" * 40)
    print("测试 4: 调度决策支持")
    print("=" * 40)
    
    result = kb.query_for_dispatch(
        grid_state="220kV 线路 L1 负载率达到 95%",
        alerts="L1 线路热稳定告警",
    )
    print(f"结果数: {result.num_results}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
