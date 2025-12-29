"""多模态RAG系统测试"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os
from src.ai.rag.multimodal_config import (
    MultimodalConfig,
    QueryType,
    QueryContext,
    ProcessedDocument
)
from src.ai.rag.multimodal_query_analyzer import MultimodalQueryAnalyzer
from src.ai.rag.document_processor import MultimodalDocumentProcessor
from src.ai.rag.multimodal_vectorstore import MultimodalVectorStore
from src.ai.rag.retrieval_strategy import SmartRetrievalStrategy
from src.ai.rag.context_assembler import MultimodalContextAssembler
from src.ai.rag.multimodal_qa_chain import MultimodalQAChain

class TestMultimodalQueryAnalyzer:
    """查询分析器测试"""
    
    def test_query_type_analyzer(self):
        """测试查询类型识别准确性"""
        analyzer = MultimodalQueryAnalyzer()
        
        # 测试文本查询
        context = analyzer.analyze_query("什么是Python编程语言？")
        assert context.query_type == QueryType.TEXT
        
        # 测试视觉查询
        context = analyzer.analyze_query("请分析这张图片中的内容")
        assert context.query_type == QueryType.VISUAL
        assert context.requires_image_search is True
        
        # 测试表格查询
        context = analyzer.analyze_query("显示销售数据表格中的统计信息")
        assert context.requires_table_search is True
        
        # 测试混合查询
        context = analyzer.analyze_query("结合图表和文档，分析产品性能")
        assert context.query_type == QueryType.MIXED
    
    def test_filter_extraction(self):
        """测试过滤条件提取"""
        analyzer = MultimodalQueryAnalyzer()
        
        # 测试文件类型过滤
        context = analyzer.analyze_query("查找所有PDF文件中的信息")
        assert "file_types" in context.filters
        assert "pdf" in context.filters["file_types"]
        
        # 测试精确匹配
        context = analyzer.analyze_query('查找包含"机器学习"的文档')
        assert "exact_match" in context.filters
        assert "机器学习" in context.filters["exact_match"]
    
    def test_retrieval_params_extraction(self):
        """测试检索参数提取"""
        analyzer = MultimodalQueryAnalyzer()
        
        # 测试top_k提取
        context = analyzer.analyze_query("返回前5个最相关的结果")
        assert context.top_k == 5
        
        # 测试相似度阈值
        context = analyzer.analyze_query("相似度大于0.8的结果")
        assert context.similarity_threshold == 0.8

class TestMultimodalDocumentProcessor:
    """文档处理器测试"""
    
    @pytest.mark.asyncio
    async def test_text_document_processing(self):
        """测试文本文档处理"""
        config = MultimodalConfig()
        processor = MultimodalDocumentProcessor(config)
        
        # 创建临时文本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("这是一个测试文档。\n它包含多行文本。\n用于测试文档处理功能。")
            temp_file = f.name
        
        try:
            # 处理文档
            processed_doc = await processor.process_document(temp_file)
            
            # 验证结果
            assert isinstance(processed_doc, ProcessedDocument)
            assert len(processed_doc.texts) > 0
            assert processed_doc.content_type == "text"
            assert processed_doc.doc_id is not None
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """测试批量文档处理"""
        config = MultimodalConfig()
        processor = MultimodalDocumentProcessor(config)
        
        # 创建多个临时文件
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                delete=False
            ) as f:
                f.write(f"测试文档 {i+1}")
                temp_files.append(f.name)
        
        try:
            # 批量处理
            processed_docs = await processor.process_batch(temp_files)
            
            # 验证结果
            assert len(processed_docs) == 3
            for doc in processed_docs:
                assert isinstance(doc, ProcessedDocument)
                assert doc.doc_id is not None
                
        finally:
            for temp_file in temp_files:
                os.unlink(temp_file)

class TestMultimodalVectorStore:
    """向量存储测试"""
    
    @pytest.mark.asyncio
    async def test_add_and_search_documents(self):
        """测试文档添加和搜索"""
        config = MultimodalConfig()
        config.chroma_persist_dir = tempfile.mkdtemp()
        vector_store = MultimodalVectorStore(config)
        
        # 创建测试文档
        test_doc = ProcessedDocument(
            doc_id="test_doc_1",
            source_file="test.txt",
            content_type="text",
            texts=[
                {"content": "Python是一种编程语言", "metadata": {}},
                {"content": "机器学习是人工智能的分支", "metadata": {}}
            ],
            summary="测试文档摘要",
            keywords=["Python", "机器学习"]
        )
        
        # 添加文档
        success = await vector_store.add_documents(test_doc)
        assert success is True
        
        # 搜索文档
        results = await vector_store.search(
            query="Python编程",
            search_type="text",
            top_k=5
        )
        
        assert results.total_results > 0
        assert len(results.texts) > 0
        
        # 清理
        vector_store.clear_all()
    
    def test_vector_store_statistics(self):
        """测试存储统计信息"""
        config = MultimodalConfig()
        config.chroma_persist_dir = tempfile.mkdtemp()
        vector_store = MultimodalVectorStore(config)
        
        stats = vector_store.get_statistics()
        
        assert "text_documents" in stats
        assert "image_documents" in stats
        assert "table_documents" in stats
        assert "embedding_dimension" in stats

class TestSmartRetrievalStrategy:
    """检索策略测试"""
    
    @pytest.mark.asyncio
    async def test_text_retrieval(self):
        """测试文本检索策略"""
        config = MultimodalConfig()
        config.chroma_persist_dir = tempfile.mkdtemp()
        
        vector_store = MultimodalVectorStore(config)
        query_analyzer = MultimodalQueryAnalyzer()
        strategy = SmartRetrievalStrategy(config, vector_store, query_analyzer)
        
        # 创建查询上下文
        query_context = QueryContext(
            query="Python编程语言",
            query_type=QueryType.TEXT
        )
        
        # 执行检索
        results = await strategy.retrieve(query_context)
        
        assert isinstance(results.texts, list)
        assert results.retrieval_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self):
        """测试混合检索策略"""
        config = MultimodalConfig()
        config.chroma_persist_dir = tempfile.mkdtemp()
        
        vector_store = MultimodalVectorStore(config)
        query_analyzer = MultimodalQueryAnalyzer()
        strategy = SmartRetrievalStrategy(config, vector_store, query_analyzer)
        
        # 创建混合查询上下文
        query_context = QueryContext(
            query="分析图表和文档中的数据",
            query_type=QueryType.MIXED,
            requires_image_search=True,
            requires_table_search=True
        )
        
        # 执行检索
        results = await strategy.retrieve(query_context)
        
        assert hasattr(results, "texts")
        assert hasattr(results, "images")
        assert hasattr(results, "tables")

class TestMultimodalContextAssembler:
    """上下文组装器测试"""
    
    def test_context_assembly(self):
        """测试上下文组装"""
        from src.ai.rag.multimodal_config import RetrievalResults, MultimodalContext
        
        assembler = MultimodalContextAssembler()
        
        # 创建测试检索结果
        retrieval_results = RetrievalResults(
            texts=[
                {"content": "测试文本1", "score": 0.9, "metadata": {"source": "doc1.txt"}},
                {"content": "测试文本2", "score": 0.8, "metadata": {"source": "doc2.txt"}}
            ],
            images=[
                {"description": "测试图像", "score": 0.85, "metadata": {}}
            ],
            tables=[
                {"description": "测试表格", "score": 0.75, "table_data": {}, "metadata": {}}
            ],
            sources=["doc1.txt", "doc2.txt"],
            total_results=4,
            retrieval_time_ms=100
        )
        
        # 组装上下文
        context = assembler.assemble_context(
            retrieval_results=retrieval_results,
            query="测试查询"
        )
        
        assert isinstance(context, MultimodalContext)
        assert len(context.texts) > 0
        assert len(context.images) > 0
        assert len(context.tables) > 0
        assert "query" in context.metadata
    
    def test_context_truncation(self):
        """测试上下文截断"""
        from src.ai.rag.multimodal_config import RetrievalResults
        
        assembler = MultimodalContextAssembler(max_context_length=100)
        
        # 创建超长内容
        long_text = "x" * 1000
        retrieval_results = RetrievalResults(
            texts=[{"content": long_text, "score": 0.9, "metadata": {}}],
            sources=["test.txt"],
            total_results=1
        )
        
        context = assembler.assemble_context(
            retrieval_results=retrieval_results,
            query="test"
        )
        
        # 验证截断
        assert len(context.texts) <= assembler.max_context_length
        assert context.metadata.get("truncated") is True

class TestMultimodalQAChain:
    """问答链测试"""
    
    @pytest.mark.asyncio
    async def test_qa_chain_basic(self):
        """测试基本问答功能"""
        config = MultimodalConfig()
        config.chroma_persist_dir = tempfile.mkdtemp()
        config.cache_enabled = False
        
        # 创建QA链（使用模拟的LLM客户端）
        qa_chain = MultimodalQAChain(config, llm_client=None)
        
        # 测试查询
        response = await qa_chain.arun(
            query="什么是Python？",
            stream=False,
            max_tokens=100
        )
        
        assert response.answer is not None
        assert response.processing_time > 0
        assert isinstance(response.context_used, dict)
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        config = MultimodalConfig()
        config.cache_enabled = True
        config.cache_ttl_seconds = 3600
        
        qa_chain = MultimodalQAChain(config)
        
        # 生成缓存键
        cache_key = qa_chain._get_cache_key("test query", {})
        assert cache_key is not None
        
        # 测试缓存更新
        test_response = {
            "answer": "test answer",
            "sources": [],
            "confidence": 0.9,
            "processing_time": 1.0,
            "context_used": {}
        }
        
        qa_chain._update_cache(cache_key, test_response)
        assert cache_key in qa_chain._query_cache
        
        # 测试缓存有效性
        cached = qa_chain._query_cache[cache_key]
        assert qa_chain._is_cache_valid(cached) is True
        
        # 清空缓存
        qa_chain.clear_cache()
        assert len(qa_chain._query_cache) == 0

@pytest.mark.integration
class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_text_query(self):
        """端到端文本查询测试"""
        config = MultimodalConfig()
        config.chroma_persist_dir = tempfile.mkdtemp()
        
        # 初始化系统
        processor = MultimodalDocumentProcessor(config)
        vector_store = MultimodalVectorStore(config)
        qa_chain = MultimodalQAChain(config)
        
        # 创建并处理测试文档
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Python是一种高级编程语言，广泛用于数据科学和机器学习。")
            temp_file = f.name
        
        try:
            # 处理并索引文档
            doc = await processor.process_document(temp_file)
            await vector_store.add_documents(doc)
            
            # 执行查询
            response = await qa_chain.arun(
                query="Python用于什么领域？",
                stream=False
            )
            
            # 验证响应
            assert response.answer is not None
            assert len(response.sources) > 0
            assert response.confidence > 0
            
        finally:
            os.unlink(temp_file)
            vector_store.clear_all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
