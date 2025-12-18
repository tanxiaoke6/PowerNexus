# -*- coding: utf-8 -*-
"""
PowerNexus - 文档摄入与向量化模块

本模块负责加载电力技术标准文档，进行分块处理，
并使用嵌入模型生成向量存入 ChromaDB。

特性:
- 支持 TXT、Markdown、PDF 文档加载
- 可配置的文档分块策略
- 使用 HuggingFace 嵌入模型
- ChromaDB 向量存储
- 支持 Mock 模式用于测试

作者: PowerNexus Team
日期: 2025-12-18
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import hashlib
import json

# 配置日志
logger = logging.getLogger(__name__)

# ============================================================================
# 尝试导入依赖库
# ============================================================================

CHROMADB_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    logger.info("ChromaDB 已成功导入")
except ImportError:
    logger.warning("ChromaDB 未安装，将使用 Mock 向量存储")
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("SentenceTransformers 已成功导入")
except ImportError:
    logger.warning("SentenceTransformers 未安装，将使用 Mock 嵌入模型")
    SentenceTransformer = None

try:
    import numpy as np
except ImportError:
    import numpy as np  # numpy 应该总是可用


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class IngestConfig:
    """
    文档摄入配置类
    
    Attributes:
        chunk_size: 分块大小 (字符数)
        chunk_overlap: 分块重叠 (字符数)
        embedding_model: 嵌入模型名称
        collection_name: ChromaDB 集合名称
        persist_directory: ChromaDB 持久化目录
        batch_size: 批量处理大小
    """
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "power_grid_standards"
    persist_directory: str = "./data/vector_db"
    batch_size: int = 100


# ============================================================================
# Mock 类（用于开发测试）
# ============================================================================

class MockEmbeddingModel:
    """
    Mock 嵌入模型
    
    当 SentenceTransformers 未安装时提供基本功能。
    生成固定维度的随机向量用于测试。
    """
    
    def __init__(self, model_name: str = "mock-model"):
        """
        初始化 Mock 嵌入模型
        
        Args:
            model_name: 模型名称（仅用于日志）
        """
        self.model_name = model_name
        self.embedding_dim = 384  # 与 all-MiniLM-L6-v2 维度一致
        logger.info(f"Mock 嵌入模型已初始化: {model_name}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        生成文本嵌入向量
        
        Args:
            sentences: 输入文本或文本列表
            show_progress_bar: 是否显示进度条（忽略）
            
        Returns:
            嵌入向量数组
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # 基于文本内容生成确定性的伪随机向量
        embeddings = []
        for text in sentences:
            # 使用文本哈希作为随机种子，确保相同文本生成相同向量
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            # 归一化
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class MockVectorStore:
    """
    Mock 向量存储
    
    当 ChromaDB 未安装时提供基本的内存向量存储功能。
    """
    
    def __init__(self, collection_name: str = "mock_collection"):
        """
        初始化 Mock 向量存储
        
        Args:
            collection_name: 集合名称
        """
        self.collection_name = collection_name
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        logger.info(f"Mock 向量存储已初始化: {collection_name}")
    
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        添加文档到存储
        
        Args:
            documents: 文档文本列表
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
        """
        n = len(documents)
        
        if ids is None:
            ids = [f"doc_{len(self.ids) + i}" for i in range(n)]
        if metadatas is None:
            metadatas = [{}] * n
        
        self.documents.extend(documents)
        self.embeddings.extend([np.array(e) for e in embeddings])
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        logger.debug(f"添加 {n} 个文档到 Mock 存储")
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        **kwargs,
    ) -> Dict[str, List]:
        """
        查询相似文档
        
        Args:
            query_embeddings: 查询向量
            n_results: 返回结果数量
            
        Returns:
            查询结果字典
        """
        if not self.embeddings:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
        
        query_vec = np.array(query_embeddings[0])
        
        # 计算余弦相似度
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_vec, emb) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8
            )
            similarities.append(sim)
        
        # 获取 top-k 索引
        top_k_indices = np.argsort(similarities)[-n_results:][::-1]
        
        return {
            "ids": [[self.ids[i] for i in top_k_indices]],
            "documents": [[self.documents[i] for i in top_k_indices]],
            "metadatas": [[self.metadatas[i] for i in top_k_indices]],
            "distances": [[1 - similarities[i] for i in top_k_indices]],
        }
    
    def count(self) -> int:
        """返回文档数量"""
        return len(self.documents)


class MockChromaClient:
    """Mock ChromaDB 客户端"""
    
    def __init__(self, **kwargs):
        self._collections: Dict[str, MockVectorStore] = {}
    
    def get_or_create_collection(
        self,
        name: str,
        **kwargs,
    ) -> MockVectorStore:
        """获取或创建集合"""
        if name not in self._collections:
            self._collections[name] = MockVectorStore(name)
        return self._collections[name]
    
    def delete_collection(self, name: str):
        """删除集合"""
        if name in self._collections:
            del self._collections[name]


# ============================================================================
# 文档加载器
# ============================================================================

class DocumentLoader:
    """
    文档加载器
    
    支持加载多种格式的文档文件。
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.markdown', '.pdf'}
    
    @staticmethod
    def load_file(file_path: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
        """
        加载单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            (文本内容, 元数据) 元组
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        extension = file_path.suffix.lower()
        
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": extension,
        }
        
        if extension in {'.txt', '.md', '.markdown'}:
            content = DocumentLoader._load_text_file(file_path)
        elif extension == '.pdf':
            content = DocumentLoader._load_pdf_file(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {extension}")
        
        return content, metadata
    
    @staticmethod
    def _load_text_file(file_path: Path) -> str:
        """加载文本文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"无法解码文件: {file_path}")
    
    @staticmethod
    def _load_pdf_file(file_path: Path) -> str:
        """加载 PDF 文件"""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            logger.warning("pypdf 未安装，无法加载 PDF 文件")
            return f"[PDF 文件: {file_path.name}，需要安装 pypdf]"
    
    @staticmethod
    def load_directory(
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        加载目录中的所有文档
        
        Args:
            directory: 目录路径
            recursive: 是否递归加载子目录
            
        Returns:
            [(文本内容, 元数据), ...] 列表
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        documents = []
        
        pattern = '**/*' if recursive else '*'
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                try:
                    content, metadata = DocumentLoader.load_file(file_path)
                    documents.append((content, metadata))
                    logger.debug(f"已加载: {file_path}")
                except Exception as e:
                    logger.warning(f"加载文件失败 {file_path}: {e}")
        
        logger.info(f"从目录加载了 {len(documents)} 个文档")
        return documents


# ============================================================================
# 文本分块器
# ============================================================================

class TextChunker:
    """
    文本分块器
    
    将长文本分割成适合嵌入的小块。
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        初始化分块器
        
        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠（字符数）
            separators: 分隔符列表（按优先级）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            分块列表
        """
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # 确定当前块的结束位置
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # 如果不是最后一块，尝试在分隔符处断开
            if end_pos < len(text):
                best_split = end_pos
                for sep in self.separators:
                    if not sep:
                        continue
                    # 在 chunk 范围内查找最后一个分隔符
                    last_sep = text.rfind(sep, current_pos, end_pos)
                    if last_sep > current_pos:
                        best_split = last_sep + len(sep)
                        break
                end_pos = best_split
            
            # 提取块
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            # 移动到下一个位置（考虑重叠）
            current_pos = end_pos - self.chunk_overlap
            if current_pos >= len(text) - self.chunk_overlap:
                break
        
        return chunks
    
    def split_documents(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        分割多个文档
        
        Args:
            documents: [(文本, 元数据), ...] 列表
            
        Returns:
            分块后的 [(文本, 元数据), ...] 列表
        """
        all_chunks = []
        
        for text, metadata in documents:
            chunks = self.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                all_chunks.append((chunk, chunk_metadata))
        
        logger.info(f"分块完成: {len(documents)} 个文档 -> {len(all_chunks)} 个块")
        return all_chunks


# ============================================================================
# 嵌入模型封装
# ============================================================================

class EmbeddingModel:
    """
    嵌入模型封装
    
    提供统一的嵌入接口，支持 Mock 模式。
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_mock: bool = False,
    ):
        """
        初始化嵌入模型
        
        Args:
            model_name: HuggingFace 模型名称
            use_mock: 是否强制使用 Mock 模型
        """
        self.model_name = model_name
        self._use_mock = use_mock or not SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self._use_mock:
            self._model = MockEmbeddingModel(model_name)
        else:
            self._model = SentenceTransformer(model_name)
        
        logger.info(
            f"嵌入模型初始化完成 | "
            f"模型: {model_name} | "
            f"Mock 模式: {self._use_mock}"
        )
    
    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        生成文本嵌入
        
        Args:
            texts: 输入文本或列表
            show_progress_bar: 是否显示进度条
            
        Returns:
            嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            show_progress_bar=show_progress_bar,
        )
        
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """返回嵌入维度"""
        if self._use_mock:
            return self._model.embedding_dim
        else:
            return self._model.get_sentence_embedding_dimension()


# ============================================================================
# 向量存储封装
# ============================================================================

class VectorStore:
    """
    向量存储封装
    
    提供统一的向量存储接口，支持 ChromaDB 和 Mock 模式。
    """
    
    def __init__(
        self,
        collection_name: str = "power_grid_standards",
        persist_directory: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录（None 表示内存模式）
            use_mock: 是否强制使用 Mock 存储
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._use_mock = use_mock or not CHROMADB_AVAILABLE
        
        if self._use_mock:
            self._client = MockChromaClient()
        else:
            if persist_directory:
                # 确保目录存在
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.Client()
        
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )
        
        logger.info(
            f"向量存储初始化完成 | "
            f"集合: {collection_name} | "
            f"Mock 模式: {self._use_mock}"
        )
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        添加文档到存储
        
        Args:
            texts: 文本列表
            embeddings: 嵌入向量数组
            metadatas: 元数据列表
            ids: 文档 ID 列表
        """
        n = len(texts)
        
        if ids is None:
            # 生成基于内容的唯一 ID
            ids = [
                hashlib.md5(text.encode()).hexdigest()[:16]
                for text in texts
            ]
        
        if metadatas is None:
            metadatas = [{}] * n
        
        # 转换嵌入为列表格式
        embeddings_list = embeddings.tolist()
        
        self._collection.add(
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids,
        )
        
        logger.info(f"添加 {n} 个文档到向量存储")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        
        # 转换为更友好的格式
        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            }
            documents.append(doc)
        
        return documents
    
    def count(self) -> int:
        """返回文档数量"""
        return self._collection.count()
    
    def clear(self):
        """清空集合"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
        )
        logger.info(f"集合 {self.collection_name} 已清空")


# ============================================================================
# 文档摄入器
# ============================================================================

class DocumentIngestor:
    """
    文档摄入器
    
    整合文档加载、分块、嵌入和存储的完整流程。
    """
    
    def __init__(
        self,
        config: Optional[IngestConfig] = None,
        use_mock: bool = False,
    ):
        """
        初始化文档摄入器
        
        Args:
            config: 摄入配置
            use_mock: 是否使用 Mock 模式
        """
        self.config = config or IngestConfig()
        self._use_mock = use_mock
        
        # 初始化组件
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        self.embedding_model = EmbeddingModel(
            model_name=self.config.embedding_model,
            use_mock=use_mock,
        )
        
        self.vector_store = VectorStore(
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
            use_mock=use_mock,
        )
        
        logger.info("文档摄入器初始化完成")
    
    def ingest_file(self, file_path: Union[str, Path]) -> int:
        """
        摄入单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            摄入的块数量
        """
        # 加载文档
        content, metadata = DocumentLoader.load_file(file_path)
        
        # 分块
        chunks = self.chunker.split_documents([(content, metadata)])
        
        if not chunks:
            logger.warning(f"文件 {file_path} 没有生成任何块")
            return 0
        
        # 提取文本和元数据
        texts = [chunk for chunk, _ in chunks]
        metadatas = [meta for _, meta in chunks]
        
        # 生成嵌入
        embeddings = self.embedding_model.embed(texts)
        
        # 存储
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info(f"文件 {file_path} 摄入完成: {len(chunks)} 个块")
        return len(chunks)
    
    def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> int:
        """
        摄入目录中的所有文档
        
        Args:
            directory: 目录路径
            recursive: 是否递归处理子目录
            
        Returns:
            摄入的总块数量
        """
        # 加载所有文档
        documents = DocumentLoader.load_directory(directory, recursive)
        
        if not documents:
            logger.warning(f"目录 {directory} 中没有找到文档")
            return 0
        
        # 分块
        chunks = self.chunker.split_documents(documents)
        
        if not chunks:
            return 0
        
        # 批量处理
        total_chunks = 0
        batch_size = self.config.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk for chunk, _ in batch]
            metadatas = [meta for _, meta in batch]
            
            # 生成嵌入
            embeddings = self.embedding_model.embed(texts)
            
            # 存储
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            
            total_chunks += len(batch)
            logger.debug(f"批次 {i // batch_size + 1} 处理完成: {len(batch)} 个块")
        
        logger.info(f"目录 {directory} 摄入完成: {total_chunks} 个块")
        return total_chunks
    
    def ingest_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        source: str = "manual",
    ) -> int:
        """
        直接摄入文本列表
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            source: 来源标识
            
        Returns:
            摄入的块数量
        """
        if not texts:
            return 0
        
        # 构建文档
        if metadatas is None:
            metadatas = [{"source": source} for _ in texts]
        
        documents = list(zip(texts, metadatas))
        
        # 分块
        chunks = self.chunker.split_documents(documents)
        
        if not chunks:
            return 0
        
        # 提取文本和元数据
        chunk_texts = [chunk for chunk, _ in chunks]
        chunk_metadatas = [meta for _, meta in chunks]
        
        # 生成嵌入
        embeddings = self.embedding_model.embed(chunk_texts)
        
        # 存储
        self.vector_store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
        )
        
        logger.info(f"手动摄入完成: {len(chunks)} 个块")
        return len(chunks)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取摄入统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "collection_name": self.config.collection_name,
            "total_documents": self.vector_store.count(),
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "use_mock": self._use_mock,
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_ingestor(
    collection_name: str = "power_grid_standards",
    persist_directory: str = "./data/vector_db",
    use_mock: bool = False,
    **kwargs,
) -> DocumentIngestor:
    """
    创建文档摄入器的便捷函数
    
    Args:
        collection_name: 集合名称
        persist_directory: 持久化目录
        use_mock: 是否使用 Mock 模式
        **kwargs: 其他配置参数
        
    Returns:
        DocumentIngestor 实例
    """
    config = IngestConfig(
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs,
    )
    return DocumentIngestor(config=config, use_mock=use_mock)


def create_sample_documents() -> List[str]:
    """
    创建示例电力技术标准文档
    
    Returns:
        示例文档列表
    """
    return [
        """
        GB/T 13398-2008 电力变压器 第1部分：总则
        
        1. 范围
        本标准规定了电力变压器的一般要求、试验、标志、包装、运输和储存。
        本标准适用于额定容量在10kVA及以上的油浸式电力变压器。
        
        2. 规范性引用文件
        GB 1094.1 电力变压器 第1部分：总则
        GB 1094.2 电力变压器 第2部分：温升
        
        3. 术语和定义
        3.1 额定容量：变压器在额定条件下工作时的视在功率。
        3.2 额定电压：变压器在额定条件下工作时的端电压。
        """,
        """
        DL/T 596-2018 电力设备预防性试验规程
        
        1. 总则
        1.1 为保证电力设备安全可靠运行，规范电力设备预防性试验工作，制定本规程。
        1.2 本规程适用于各类发电、变电、输电设备的预防性试验。
        
        2. 变压器试验
        2.1 绝缘电阻测量
        - 测量前设备应充分放电
        - 环境温度应在10~40℃范围内
        - 相对湿度不大于80%
        
        2.2 介质损耗因数测量
        - 应在设备运行温度下进行
        - 测量值应换算到20℃进行比较
        """,
        """
        GB 50150-2016 电气装置安装工程 电气设备交接试验标准
        
        4.2 绝缘子
        4.2.1 一般规定
        绝缘子在安装前应进行外观检查，不得有裂纹、破损、釉面脱落等缺陷。
        
        4.2.2 试验项目
        (1) 绝缘电阻测量
        (2) 交流耐压试验
        (3) 污秽度检查
        
        4.2.3 标准值
        - 绝缘电阻：不小于300MΩ
        - 交流耐压：按额定电压的1.5倍
        """,
        """
        电力系统继电保护技术规范
        
        第五章 线路保护
        
        5.1 距离保护
        距离保护是根据故障点到保护安装处的阻抗来判断故障位置的保护装置。
        
        5.1.1 保护范围
        - I段：保护线路全长的80~85%，无时限动作
        - II段：延伸到相邻线路，带0.5s时限
        - III段：远后备，带1~2s时限
        
        5.2 差动保护
        差动保护利用基尔霍夫电流定律，比较保护范围两端电流的差值来判断故障。
        
        5.2.1 动作特性
        - 差动启动电流：0.2~0.5倍额定电流
        - 制动系数：0.5~0.7
        """,
        """
        电网调度运行管理规程
        
        第三章 调度指令
        
        3.1 调度指令的下达
        调度员应使用规范的调度术语下达指令，并进行复诵确认。
        
        3.2 拓扑调整操作
        3.2.1 解环操作
        在进行解环操作前，应确认：
        - 解环后各段线路不过载
        - 电压水平满足要求
        - 继电保护整定值正确
        
        3.2.2 合环操作
        合环前应检查：
        - 两侧电压差不超过额定值的10%
        - 相位差不超过25°
        - 频率差不超过0.5Hz
        """,
    ]


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
    print("PowerNexus - 文档摄入模块测试")
    print("=" * 60)
    
    # 创建摄入器 (Mock 模式)
    ingestor = create_ingestor(use_mock=True)
    
    print(f"\n初始统计: {ingestor.get_stats()}")
    
    # 摄入示例文档
    sample_docs = create_sample_documents()
    count = ingestor.ingest_texts(sample_docs, source="power_grid_standards")
    print(f"\n摄入 {count} 个文档块")
    
    # 统计信息
    print(f"\n最终统计: {ingestor.get_stats()}")
    
    # 测试搜索
    print("\n测试向量搜索...")
    query = "变压器绝缘电阻测量要求"
    query_embedding = ingestor.embedding_model.embed(query)[0]
    results = ingestor.vector_store.search(query_embedding, top_k=3)
    
    print(f"\n查询: '{query}'")
    print("搜索结果:")
    for i, result in enumerate(results):
        print(f"  {i+1}. [距离: {result['distance']:.4f}]")
        print(f"     {result['text'][:100]}...")
    
    print("\n测试完成!")
