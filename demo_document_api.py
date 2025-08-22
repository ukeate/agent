#!/usr/bin/env python3
"""
文档处理API演示脚本
"""

import asyncio
import tempfile
from pathlib import Path

# 导入我们的文档处理模块
import sys
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

from ai.document_processing import DocumentProcessor
from ai.document_processing.chunkers import IntelligentChunker, ChunkStrategy


async def demo_document_processing():
    """演示文档处理功能"""
    print("🔍 智能文档处理系统演示")
    print("=" * 50)
    
    # 创建测试文档
    test_files = {
        "sample.txt": "这是一个测试文档。包含多个段落。\n\n这是第二段，用于测试分块功能。\n\n最后一段包含总结内容。",
        "sample.py": 'def hello_world():\n    """Hello World函数"""\n    print("Hello, World!")\n    return "success"\n\nif __name__ == "__main__":\n    hello_world()',
        "sample.md": "# 文档标题\n\n## 第一节\n\n这是第一节的内容。\n\n## 第二节\n\n这是第二节的内容，包含**粗体**文本。\n\n### 子节\n\n子节内容。"
    }
    
    # 创建临时文件
    temp_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        for filename, content in test_files.items():
            file_path = temp_dir_path / filename
            file_path.write_text(content, encoding='utf-8')
            temp_files.append(file_path)
        
        # 初始化文档处理器
        processor = DocumentProcessor(
            enable_ocr=False,
            extract_images=False
        )
        
        # 处理每个文档
        for file_path in temp_files:
            print(f"\n📄 处理文档: {file_path.name}")
            print("-" * 30)
            
            try:
                # 处理文档
                doc = await processor.process_document(file_path)
                
                print(f"✓ 文档ID: {doc.doc_id}")
                print(f"✓ 标题: {doc.title}")
                print(f"✓ 文件类型: {doc.file_type}")
                print(f"✓ 内容长度: {len(doc.content)} 字符")
                print(f"✓ 元数据: {doc.metadata}")
                
                # 演示分块功能
                print(f"\n🔧 智能分块演示:")
                chunker = IntelligentChunker()
                
                # 测试不同的分块策略
                strategies = [
                    (ChunkStrategy.SEMANTIC, "语义分块"),
                    (ChunkStrategy.FIXED, "固定长度分块"),
                    (ChunkStrategy.ADAPTIVE, "自适应分块")
                ]
                
                for strategy, name in strategies:
                    chunker.strategy = strategy
                    chunks = await chunker.chunk_document(
                        doc.content,
                        content_type=doc.file_type,
                        metadata=doc.metadata
                    )
                    
                    print(f"  - {name}: {len(chunks)} 个分块")
                    if chunks:
                        print(f"    首个分块预览: {chunks[0].content[:50]}...")
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
        
        # 演示批量处理
        print(f"\n🚀 批量处理演示:")
        print("-" * 30)
        
        try:
            batch_results = await processor.process_batch(
                temp_files,
                concurrent_limit=3,
                continue_on_error=True
            )
            
            print(f"✓ 批量处理完成，处理了 {len(batch_results)} 个文档")
            for doc in batch_results:
                print(f"  - {doc.title}: {doc.file_type}")
                
        except Exception as e:
            print(f"❌ 批量处理失败: {e}")
        
        # 演示支持的格式
        print(f"\n📋 支持的文档格式:")
        print("-" * 30)
        formats = processor.get_supported_formats()
        for fmt in sorted(formats):
            print(f"  - {fmt}")


async def demo_api_simulation():
    """模拟API调用演示"""
    print(f"\n🌐 API调用模拟演示:")
    print("=" * 50)
    
    # 模拟支持格式查询
    print("GET /api/v1/documents/supported-formats")
    processor = DocumentProcessor()
    formats = processor.get_supported_formats()
    
    response = {
        "formats": formats,
        "categories": {
            "documents": [".pdf", ".docx", ".pptx"],
            "spreadsheets": [".xlsx", ".xls", ".csv"],
            "text": [".txt", ".md", ".markdown", ".rst"],
            "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"],
            "web": [".html", ".htm", ".xml", ".json", ".yaml", ".yml"]
        }
    }
    
    print(f"✓ 返回: {len(formats)} 种支持的格式")
    print(f"✓ 分类: {list(response['categories'].keys())}")
    
    print(f"\n✅ 文档处理系统演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(demo_document_processing())
    asyncio.run(demo_api_simulation())