"""
RAG系统业务逻辑层
"""

import logging
from typing import Dict, List, Optional

from src.ai.rag.embeddings import embedding_service
from src.ai.rag.hybrid_search import get_hybrid_search_engine, SearchStrategy
from src.ai.rag.retriever import hybrid_retriever, query_classifier, semantic_retriever
from src.ai.rag.vectorizer import file_vectorizer
from src.core.config import get_settings
from src.core.qdrant import qdrant_manager

settings = get_settings()

logger = logging.getLogger(__name__)


class RAGService:
    """RAG服务类"""

    def __init__(self):
        self.vectorizer = file_vectorizer
        self.semantic_retriever = semantic_retriever
        self.hybrid_retriever = hybrid_retriever
        self.query_classifier = query_classifier
        self.bm42_search_engine = get_hybrid_search_engine()

    async def initialize(self):
        """初始化RAG系统"""
        try:
            # 初始化Qdrant集合
            await qdrant_manager.initialize_collections()
            logger.info("RAG system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False

    async def query(
        self,
        query: str,
        search_type: str = "bm42_hybrid",
        limit: int = 10,
        score_threshold: float = 0.5,
        filter_dict: Optional[Dict] = None,
        strategy: Optional[str] = None,
    ) -> Dict:
        """执行检索查询"""
        try:
            # 分类查询意图
            intent = self.query_classifier.classify(query)
            
            # 根据意图选择集合
            if intent["type"] == "code":
                collection = "code"
            else:
                collection = "documents"
            
            # 映射搜索策略
            search_strategy = None
            if strategy:
                strategy_map = {
                    "vector_only": SearchStrategy.VECTOR_ONLY,
                    "bm25_only": SearchStrategy.BM25_ONLY,
                    "hybrid_rrf": SearchStrategy.HYBRID_RRF,
                    "hybrid_weighted": SearchStrategy.HYBRID_WEIGHTED,
                    "adaptive": SearchStrategy.ADAPTIVE,
                }
                search_strategy = strategy_map.get(strategy)
            
            # 执行检索
            if search_type == "bm42_hybrid":
                # 使用新的BM42混合搜索引擎
                search_results = await self.bm42_search_engine.search(
                    query=query,
                    collection=collection,
                    limit=limit,
                    filters=filter_dict,
                    strategy=search_strategy
                )
                # 转换为旧格式兼容
                results = []
                for result in search_results:
                    results.append({
                        "id": result.id,
                        "score": result.final_score,
                        "content": result.content,
                        "file_path": result.file_path,
                        "file_type": result.file_type,
                        "chunk_index": result.chunk_index,
                        "metadata": result.metadata,
                        "semantic_score": result.semantic_score,
                        "keyword_score": result.keyword_score,
                        "collection": result.collection
                    })
            elif search_type == "bm42_multi":
                # 跨集合BM42搜索
                search_results = await self.bm42_search_engine.multi_collection_search(
                    query=query,
                    collections=["documents", "code"],
                    limit=limit,
                    strategy=search_strategy
                )
                # 转换为旧格式兼容
                results = []
                for result in search_results:
                    results.append({
                        "id": result.id,
                        "score": result.final_score,
                        "content": result.content,
                        "file_path": result.file_path,
                        "file_type": result.file_type,
                        "chunk_index": result.chunk_index,
                        "metadata": result.metadata,
                        "semantic_score": result.semantic_score,
                        "keyword_score": result.keyword_score,
                        "collection": result.collection
                    })
            elif search_type == "semantic":
                results = await self.semantic_retriever.search(
                    query=query,
                    collection=collection,
                    limit=limit,
                    score_threshold=score_threshold,
                    filter_dict=filter_dict,
                )
            elif search_type == "hybrid":
                # 保留旧的混合搜索以向后兼容
                results = await self.hybrid_retriever.hybrid_search(
                    query=query,
                    collection=collection,
                    limit=limit,
                    score_threshold=score_threshold,
                )
                # 重新排序以增加多样性
                results = await self.hybrid_retriever.rerank_results(
                    query, results
                )
            elif search_type == "multi":
                results = await self.semantic_retriever.multi_collection_search(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                )
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            return {
                "success": True,
                "query": query,
                "intent": intent,
                "results": results,
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "count": 0,
            }

    async def index_file(self, file_path: str, force: bool = False) -> Dict:
        """索引单个文件"""
        try:
            result = await self.vectorizer.vectorize_file(file_path, force)
            return {
                "success": True,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file": file_path,
            }

    async def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        force: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> Dict:
        """索引目录"""
        try:
            results = await self.vectorizer.vectorize_directory(
                directory=directory,
                recursive=recursive,
                force=force,
                extensions=extensions,
            )
            
            # 统计结果
            indexed = sum(1 for r in results if r["status"] == "indexed")
            skipped = sum(1 for r in results if r["status"] == "skipped")
            errors = sum(1 for r in results if r["status"] == "error")
            
            return {
                "success": True,
                "directory": directory,
                "results": results,
                "summary": {
                    "total": len(results),
                    "indexed": indexed,
                    "skipped": skipped,
                    "errors": errors,
                },
            }
        except Exception as e:
            logger.error(f"Failed to index directory {directory}: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory": directory,
            }

    async def update_index(self, file_paths: List[str]) -> Dict:
        """更新索引"""
        try:
            results = await self.vectorizer.update_index(file_paths)
            
            # 统计结果
            updated = sum(1 for r in results if r["status"] == "indexed")
            removed = sum(1 for r in results if r["status"] == "removed")
            unchanged = sum(1 for r in results if r["status"] == "unchanged")
            errors = sum(1 for r in results if r["status"] == "error")
            
            return {
                "success": True,
                "results": results,
                "summary": {
                    "total": len(results),
                    "updated": updated,
                    "removed": removed,
                    "unchanged": unchanged,
                    "errors": errors,
                },
            }
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        try:
            logger.info("RAG service: Starting to get index stats")
            stats = self.vectorizer.get_index_stats()
            logger.info(f"RAG service: Got stats from vectorizer: {stats}")
            
            # 检查Qdrant健康状态
            health = await qdrant_manager.health_check()
            
            # 处理stats中的错误情况并计算总存储大小
            processed_stats = {}
            total_disk_size = 0
            for collection_name, collection_data in stats.items():
                logger.info(f"RAG service: Processing collection {collection_name}: {collection_data}")
                if "error" in collection_data:
                    # 如果集合有错误，返回默认值和错误信息
                    processed_stats[collection_name] = {
                        "vectors_count": 0,
                        "points_count": 0,
                        "segments_count": 0,
                        "status": "error",
                        "estimated_disk_size": 0,
                        "error": collection_data["error"]
                    }
                else:
                    processed_stats[collection_name] = collection_data
                    # 累加估算磁盘存储大小
                    estimated_size = collection_data.get("estimated_disk_size", 0)
                    total_disk_size += estimated_size
                    logger.info(f"RAG service: Collection {collection_name} data: {collection_data}")
                    logger.info(f"RAG service: Collection {collection_name} estimated_size: {estimated_size}, total so far: {total_disk_size}")
            
            logger.info(f"RAG service: Final total_disk_size: {total_disk_size}")
            return {
                "success": True,
                "stats": processed_stats,
                "total_disk_size": total_disk_size,
                "health": {
                    "qdrant": health,
                    "openai": settings.OPENAI_API_KEY != "",
                },
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {
                "success": False,
                "stats": {
                    "documents": {
                        "vectors_count": 0,
                        "points_count": 0,
                        "segments_count": 0,
                        "status": "error",
                        "error": str(e)
                    },
                    "code": {
                        "vectors_count": 0,
                        "points_count": 0,
                        "segments_count": 0,
                        "status": "error",
                        "error": str(e)
                    }
                },
                "health": {
                    "qdrant": False,
                    "openai": settings.OPENAI_API_KEY != "",
                },
                "error": str(e),
            }

    async def reset_index(self, collection: Optional[str] = None) -> Dict:
        """重置索引"""
        try:
            client = qdrant_manager.get_client()
            
            if collection:
                # 重置特定集合
                client.delete_collection(collection)
                logger.info(f"Deleted collection: {collection}")
                
                # 重新创建
                await qdrant_manager.initialize_collections()
                message = f"Reset collection: {collection}"
            else:
                # 重置所有集合
                for coll_name in ["documents", "code"]:
                    try:
                        client.delete_collection(coll_name)
                        logger.info(f"Deleted collection: {coll_name}")
                    except:
                        pass
                
                # 重新创建所有集合
                await qdrant_manager.initialize_collections()
                message = "Reset all collections"
            
            return {
                "success": True,
                "message": message,
            }
        except Exception as e:
            logger.error(f"Failed to reset index: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# 全局RAG服务实例
rag_service = RAGService()

# 初始化函数
async def initialize_rag_service():
    """初始化RAG服务"""
    return await rag_service.initialize()