# -*- coding: utf-8 -*-
"""
PowerNexus - RAG 知识库模块

本模块负责构建和查询电力技术标准知识库。

主要组件:
- ingest: 文档加载、分块、嵌入与向量存储
- retriever: KnowledgeBase 知识库检索与 LLM 提示构建

使用示例:
    >>> from src.rag import KnowledgeBase, create_knowledge_base
    >>> from src.rag import DocumentIngestor, create_ingestor
    >>> 
    >>> # 创建知识库
    >>> kb = create_knowledge_base(use_mock=True)
    >>> 
    >>> # 查询
    >>> result = kb.query("变压器绝缘电阻测量要求")
    >>> print(result.prompt)  # LLM 提示
    >>> 
    >>> # 文档摄入
    >>> ingestor = create_ingestor(use_mock=True)
    >>> ingestor.ingest_texts(["标准文本..."])
"""

# 文档摄入模块
from .ingest import (
    # 配置
    IngestConfig,
    # 组件类
    DocumentLoader,
    TextChunker,
    EmbeddingModel,
    VectorStore,
    DocumentIngestor,
    # Mock 类
    MockEmbeddingModel,
    MockVectorStore,
    MockChromaClient,
    # 便捷函数
    create_ingestor,
    create_sample_documents,
    # 可用性标志
    CHROMADB_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)

# 知识库检索模块
from .retriever import (
    # 配置
    RetrieverConfig,
    # 核心类
    KnowledgeBase,
    RetrievalResult,
    PromptTemplates,
    # 便捷函数
    create_knowledge_base,
    init_knowledge_base_with_samples,
)

# 模块版本
__version__ = "0.1.0"

# 导出列表
__all__ = [
    # 配置
    "IngestConfig",
    "RetrieverConfig",
    # 摄入
    "DocumentLoader",
    "TextChunker",
    "EmbeddingModel",
    "VectorStore",
    "DocumentIngestor",
    "create_ingestor",
    "create_sample_documents",
    # 检索
    "KnowledgeBase",
    "RetrievalResult",
    "PromptTemplates",
    "create_knowledge_base",
    "init_knowledge_base_with_samples",
    # Mock
    "MockEmbeddingModel",
    "MockVectorStore",
    "MockChromaClient",
    # 标志
    "CHROMADB_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
]
