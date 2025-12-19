#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
摄入文档到 RAG 知识库的脚本

默认摄入 data/manuals 目录下的所有文档 (PDF, TXT, Markdown)

用法: 
    python tools/ingest_pdf.py                    # 摄入 data/manuals 下所有文档
    python tools/ingest_pdf.py path/to/file.pdf  # 摄入单个文件
    python tools/ingest_pdf.py path/to/folder    # 摄入指定文件夹
"""

import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# 默认摄入目录
DEFAULT_MANUALS_DIR = PROJECT_ROOT / "data" / "manuals"


def main():
    parser = argparse.ArgumentParser(description="摄入文档到 RAG 知识库")
    parser.add_argument(
        "path", 
        nargs="?",
        default=str(DEFAULT_MANUALS_DIR),
        help=f"文件或目录路径 (默认: {DEFAULT_MANUALS_DIR})"
    )
    parser.add_argument("--use-mock", action="store_true", help="使用 Mock 嵌入模型")
    args = parser.parse_args()
    
    target_path = Path(args.path)
    
    if not target_path.exists():
        print(f"错误: 路径不存在: {target_path}")
        sys.exit(1)
    
    # 导入模块
    from src.rag.ingest import DocumentLoader, DocumentIngestor, IngestConfig
    
    # 创建摄入器
    print("=" * 60)
    print("PowerNexus - 文档摄入工具")
    print("=" * 60)
    print(f"\n初始化摄入器...")
    
    config = IngestConfig()
    ingestor = DocumentIngestor(config=config, use_mock=args.use_mock)
    
    if target_path.is_file():
        # 摄入单个文件
        print(f"\n📄 摄入单个文件: {target_path}")
        result = ingestor.ingest_file(target_path)
        print(f"   结果: {result}")
    else:
        # 摄入目录下所有文件
        print(f"\n📁 摄入目录: {target_path}")
        
        # 列出支持的文件
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
        files = [f for f in target_path.rglob('*') if f.is_file() and f.suffix.lower() in supported_extensions]
        
        if not files:
            print(f"   ⚠️ 目录中没有找到支持的文档文件")
            print(f"   支持的格式: {', '.join(supported_extensions)}")
            sys.exit(0)
        
        print(f"   找到 {len(files)} 个文档:")
        for f in files:
            print(f"     - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        
        print(f"\n开始摄入...")
        result = ingestor.ingest_directory(target_path)
        print(f"\n摄入结果: {result}")
    
    # 显示统计
    print("\n" + "-" * 40)
    stats = ingestor.get_stats()
    print(f"📊 知识库统计:")
    print(f"   - 总文档数: {stats.get('total_documents', 'N/A')}")
    print(f"   - 集合名称: {stats.get('collection_name', 'N/A')}")
    
    print("\n✅ 摄入完成!")
    print("=" * 60)
    

if __name__ == "__main__":
    main()

