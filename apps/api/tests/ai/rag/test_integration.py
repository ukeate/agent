"""
RAG系统集成测试
"""

import asyncio
import os
import tempfile
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from services.rag_service import RAGService
from ai.rag.vectorizer import FileVectorizer
from ai.rag.retriever import SemanticRetriever, HybridRetriever


class TestRAGIntegration:
    """
RAG系统集成测试
测试整个RAG流程：文件索引 -> 向量化 -> 检索
"""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI客户端"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        
        # 模拟不同内容的嵌入结果
        def create_embedding_response(input_text):
            if isinstance(input_text, list):
                # 批量处理
                embeddings = []
                for i, text in enumerate(input_text):
                    # 根据文本内容生成不同的嵌入向量
                    base_val = hash(text) % 100 / 100.0
                    embedding = [base_val + j * 0.001 for j in range(1536)]
                    embeddings.append(MagicMock(embedding=embedding))
                return MagicMock(data=embeddings)
            else:
                # 单个文本
                base_val = hash(input_text) % 100 / 100.0
                embedding = [base_val + j * 0.001 for j in range(1536)]
                return MagicMock(data=[MagicMock(embedding=embedding)])
        
        mock_client.embeddings.create.side_effect = create_embedding_response
        return mock_client

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant客户端"""
        mock_client = MagicMock()
        
        # 模拟存储的数据
        stored_points = {}
        
        def mock_upsert(collection_name, points):
            if collection_name not in stored_points:
                stored_points[collection_name] = {}
            for point in points:
                stored_points[collection_name][point.id] = point
            return True
        
        def mock_search(collection_name, query_vector, limit=10, score_threshold=0.0, **kwargs):
            if collection_name not in stored_points:
                return []
            
            results = []
            for point_id, point in stored_points[collection_name].items():
                # 简单的相似度计算，确保总是有一些匹配
                # 使用前10个维度计算简单相似度
                if hasattr(point, 'vector') and len(point.vector) >= 10:
                    similarity = sum(a * b for a, b in zip(query_vector[:10], point.vector[:10])) / 10.0
                    # 确保相似度在合理范围内并高于阈值
                    similarity = abs(similarity) + 0.5  # 至少0.5分
                else:
                    similarity = 0.7  # 默认相似度
                
                if similarity >= score_threshold:
                    hit = MagicMock()
                    hit.id = point_id
                    hit.score = min(similarity, 1.0)
                    hit.payload = point.payload
                    results.append(hit)
            
            # 按分数排序
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
        
        def mock_scroll(collection_name, scroll_filter=None, limit=10, **kwargs):
            if collection_name not in stored_points:
                return [], None
            
            # 简单的过滤逻辑
            results = []
            for point_id, point in stored_points[collection_name].items():
                # 模拟文本匹配
                if scroll_filter and hasattr(scroll_filter, 'should'):
                    # 检查内容是否包含关键词
                    content = point.payload.get('content', '').lower()
                    found_match = False
                    for condition in scroll_filter.should:
                        if hasattr(condition, 'match') and hasattr(condition.match, 'text'):
                            if condition.match.text.lower() in content:
                                found_match = True
                                break
                    if found_match:
                        results.append(point)
                else:
                    results.append(point)
                
                if len(results) >= limit:
                    break
            
            return results, None
        
        mock_client.upsert = mock_upsert
        mock_client.search = mock_search
        mock_client.scroll = mock_scroll
        
        # 初始化时检查集合是否存在
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.create_collection = MagicMock()
        
        return mock_client

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis客户端"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # 无缓存
        mock_redis.setex = AsyncMock()
        return mock_redis

    @pytest_asyncio.fixture
    async def rag_service(self, mock_qdrant_client, mock_openai_client, mock_redis):
        """创建测试用的RAG服务"""
        # 创建Mock的EmbeddingService实例
        mock_embedding_service = AsyncMock()
        mock_embedding_service.embed_text.side_effect = lambda text: [hash(text) % 100 / 100.0 + j * 0.001 for j in range(1536)]
        mock_embedding_service.embed_batch.side_effect = lambda texts: [[hash(text) % 100 / 100.0 + j * 0.001 for j in range(1536)] for text in texts]
        
        with patch('ai.rag.embeddings.get_redis', return_value=mock_redis):
            with patch('core.qdrant.get_qdrant_client', return_value=mock_qdrant_client):
                service = RAGService()
                # 直接替换service中各组件的embedding_service属性
                service.vectorizer.embedding_service = mock_embedding_service
                service.vectorizer.client = mock_qdrant_client
                
                # 更新检索器
                service.semantic_retriever.embedding_service = mock_embedding_service
                service.semantic_retriever.client = mock_qdrant_client
                service.hybrid_retriever.semantic_retriever.embedding_service = mock_embedding_service
                service.hybrid_retriever.semantic_retriever.client = mock_qdrant_client
                service.hybrid_retriever.client = mock_qdrant_client
                
                await service.initialize()
                return service

    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self, rag_service):
        """
测试完整的RAG工作流程：
1. 索引文件
2. 执行查询
3. 验证结果
"""
        # 1. 创建测试文件
        test_content = '''
def fibonacci(n):
    """计算斜波那契数列"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """计算阶乘"""
    if n <= 1:
        return 1
    return n * factorial(n-1)

class Calculator:
    """简单计算器"""
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_content)
            test_file = f.name
        
        try:
            # 2. 索引文件
            index_result = await rag_service.index_file(test_file)
            
            # 调试输出
            print(f"Index result: {index_result}")
            
            # 验证索引结果
            if not index_result["success"]:
                print(f"Index failed with error: {index_result.get('error', 'Unknown error')}")
            assert index_result["success"] is True
            assert index_result["result"]["status"] == "indexed"
            assert index_result["result"]["chunks"] > 0
            
            # 3. 执行语义查询 - 使用multi类型搜索所有集合
            semantic_query_result = await rag_service.query(
                query="如何计算斜波那契数",
                search_type="multi",
                limit=5,
                score_threshold=0.1
            )
            
            # 调试输出
            print(f"Semantic query result: {semantic_query_result}")
            
            # 验证语义查询结果
            assert semantic_query_result["success"] is True
            assert semantic_query_result["count"] > 0
            
            # 验证返回的结果包含相关内容
            results = semantic_query_result["results"]
            assert len(results) > 0
            fibonacci_found = any("fibonacci" in r["content"].lower() for r in results)
            assert fibonacci_found, "Should find fibonacci function"
            
            # 4. 执行混合查询
            hybrid_query_result = await rag_service.query(
                query="Calculator class add method",
                search_type="hybrid",
                limit=5,
                score_threshold=0.1
            )
            
            # 验证混合查询结果
            assert hybrid_query_result["success"] is True
            assert hybrid_query_result["count"] > 0
            
            # 验证返回的结果包含 Calculator 相关内容
            hybrid_results = hybrid_query_result["results"]
            calculator_found = any("calculator" in r["content"].lower() for r in hybrid_results)
            assert calculator_found, "Should find Calculator class"
            
            # 5. 测试多集合查询
            multi_query_result = await rag_service.query(
                query="Python function",
                search_type="multi",
                limit=10,
                score_threshold=0.1
            )
            
            # 验证多集合查询结果
            assert multi_query_result["success"] is True
            
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.unlink(test_file)

    @pytest.mark.asyncio
    async def test_directory_indexing_and_search(self, rag_service):
        """测试目录索引和查询"""
        # 创建临时目录结构
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建多个测试文件
            files_content = {
                "math_utils.py": '''
def add(a, b):
    """加法操作"""
    return a + b

def subtract(a, b):
    """减法操作"""
    return a - b
''',
                "string_utils.py": '''
def capitalize_words(text):
    """将每个单词的首字母大写"""
    return ' '.join(word.capitalize() for word in text.split())

def reverse_string(text):
    """反转字符串"""
    return text[::-1]
''',
                "readme.md": '''
# 工具库文档

这是一个简单的工具库，包含数学和字符串处理函数。

## 数学工具
- add(): 加法运算
- subtract(): 减法运算

## 字符串工具
- capitalize_words(): 字符串大写
- reverse_string(): 字符串反转
'''
            }
            
            # 写入文件
            for filename, content in files_content.items():
                with open(os.path.join(temp_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # 索引目录
            index_result = await rag_service.index_directory(
                directory=temp_dir,
                recursive=False,
                force=True
            )
            
            # 验证索引结果
            assert index_result["success"] is True
            assert index_result["summary"]["indexed"] == 3  # 3个文件
            assert index_result["summary"]["errors"] == 0
            
            # 测试查询数学函数
            math_query_result = await rag_service.query(
                query="加法函数怎么使用",
                search_type="hybrid",
                limit=5,
                score_threshold=0.1
            )
            
            assert math_query_result["success"] is True
            assert math_query_result["count"] > 0
            
            # 验证找到数学相关内容
            math_results = math_query_result["results"]
            add_found = any("add" in r["content"].lower() for r in math_results)
            assert add_found, "Should find add function"
            
            # 测试查询字符串函数
            string_query_result = await rag_service.query(
                query="字符串反转功能",
                search_type="semantic",
                limit=5,
                score_threshold=0.1
            )
            
            assert string_query_result["success"] is True
            assert string_query_result["count"] > 0
            
            # 验证找到字符串相关内容
            string_results = string_query_result["results"]
            reverse_found = any("reverse" in r["content"].lower() for r in string_results)
            assert reverse_found, "Should find reverse function"

    @pytest.mark.asyncio
    async def test_index_update_workflow(self, rag_service):
        """测试索引更新工作流程"""
        original_content = '''
def old_function():
    """原有函数"""
    return "old"
'''
        
        updated_content = '''
def old_function():
    """更新后的函数"""
    return "updated"

def new_function():
    """新增函数"""
    return "new"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(original_content)
            test_file = f.name
        
        try:
            # 1. 索引原文件
            await rag_service.index_file(test_file)
            
            # 2. 查询原内容
            original_query = await rag_service.query(
                query="old function",
                search_type="semantic",
                limit=5,
                score_threshold=0.1
            )
            assert original_query["success"] is True
            assert original_query["count"] > 0
            
            # 3. 更新文件内容
            with open(test_file, 'w') as f:
                f.write(updated_content)
            
            # 4. 更新索引
            update_result = await rag_service.update_index([test_file])
            assert update_result["success"] is True
            assert update_result["summary"]["updated"] == 1
            
            # 5. 查询更新后的内容
            updated_query = await rag_service.query(
                query="new function",
                search_type="semantic",
                limit=5,
                score_threshold=0.1
            )
            assert updated_query["success"] is True
            assert updated_query["count"] > 0
            
            # 验证找到新函数
            new_results = updated_query["results"]
            new_found = any("new_function" in r["content"] for r in new_results)
            assert new_found, "Should find new_function after update"
            
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.unlink(test_file)

    @pytest.mark.asyncio
    async def test_error_handling(self, rag_service):
        """测试错误处理"""
        # 测试不存在的文件
        index_result = await rag_service.index_file("/non/existent/file.py")
        assert index_result["success"] is False
        assert "error" in index_result
        
        # 测试不支持的搜索类型
        query_result = await rag_service.query(
            query="test",
            search_type="unsupported_type"
        )
        assert query_result["success"] is False
        assert "error" in query_result

    @pytest.mark.asyncio
    async def test_performance_with_large_content(self, rag_service):
        """测试大内容的性能"""
        # 生成大量文本内容
        large_content = ""
        for i in range(100):
            large_content += f'''
def function_{i}(param):
    """函数 {i} - 处理参数 {{param}}"""
    result = param * {i} + {i}
    return result

class Class{i}:
    """类 {i} - 示例类"""
    
    def method_{i}(self, value):
        return value + {i}

'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            test_file = f.name
        
        try:
            # 测试索引大文件的性能
            import time
            start_time = time.time()
            
            index_result = await rag_service.index_file(test_file, force=True)
            
            index_time = time.time() - start_time
            
            # 验证索引成功
            assert index_result["success"] is True
            assert index_result["result"]["chunks"] > 50  # 应该有很多块
            
            # 验证索引时间在合理范围内（少于30秒）
            assert index_time < 30, f"Indexing took too long: {index_time}s"
            
            # 测试查询性能
            start_time = time.time()
            
            query_result = await rag_service.query(
                query="function_50 processing",
                search_type="hybrid",
                limit=10,
                score_threshold=0.1
            )
            
            query_time = time.time() - start_time
            
            # 验证查询成功
            assert query_result["success"] is True
            assert query_result["count"] > 0
            
            # 验证查询时间在合理范围内（少于5秒）
            assert query_time < 5, f"Query took too long: {query_time}s"
            
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.unlink(test_file)