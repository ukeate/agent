"""多模态向量存储管理"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import asyncio

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from .multimodal_config import (
    MultimodalConfig,
    EmbeddingModel,
    VectorStoreType,
    ProcessedDocument,
    RetrievalResults
)

logger = logging.getLogger(__name__)


class NomicEmbeddings:
    """Nomic嵌入模型封装"""
    
    def __init__(self, model: str = "nomic-embed-text-v1.5", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("NOMIC_API_KEY")
        self.dimension = 768  # Nomic默认维度
        self._validate_api_key()
        
    def _validate_api_key(self):
        """验证API密钥"""
        if not self.api_key:
            logger.warning("Nomic API key not provided, using mock embeddings")
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档"""
        if not texts:
            return []
            
        if self.api_key:
            # TODO: 实现真实的Nomic API调用
            # 这里应该调用Nomic API
            logger.warning("Using mock embeddings. Implement actual Nomic API call for production.")
        
        # 生成确定性的模拟嵌入向量（用于开发和测试）
        import hashlib
        import numpy as np
        
        embeddings = []
        for text in texts:
            # 使用文本内容的哈希来生成确定性的嵌入向量
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            # 设置种子以确保确定性
            np.random.seed(int(text_hash[:8], 16) % (2**32))
            embedding = np.random.randn(self.dimension).tolist()
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """同步嵌入文档"""
        return asyncio.run(self.aembed_documents(texts))
    
    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询"""
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
    
    def embed_query(self, text: str) -> List[float]:
        """同步嵌入查询"""
        return asyncio.run(self.aembed_query(text))


class MultimodalVectorStore:
    """统一的多模态向量存储管理"""
    
    def __init__(self, config: MultimodalConfig):
        """初始化多模态向量存储
        
        Args:
            config: 多模态配置
        """
        self.config = config
        self._initialize_stores()
        self._initialize_retrievers()
        
    def _initialize_stores(self):
        """初始化向量存储"""
        # 初始化嵌入模型
        if self.config.text_embedding_model == EmbeddingModel.NOMIC_TEXT:
            self.text_embeddings = NomicEmbeddings(
                model=self.config.text_embedding_model,
                api_key=self.config.nomic_api_key
            )
        else:
            self.text_embeddings = OpenAIEmbeddings(
                model=self.config.text_embedding_model,
                openai_api_key=self.config.openai_api_key
            )
        
        if self.config.vision_embedding_model == EmbeddingModel.NOMIC_VISION:
            self.vision_embeddings = NomicEmbeddings(
                model=self.config.vision_embedding_model,
                api_key=self.config.nomic_api_key
            )
        else:
            # 视觉嵌入暂时使用文本嵌入
            self.vision_embeddings = self.text_embeddings
        
        # 初始化向量存储
        if self.config.vector_store_type == VectorStoreType.CHROMA:
            # 文本向量存储
            self.text_store = Chroma(
                collection_name="multimodal_text",
                embedding_function=self.text_embeddings,
                persist_directory=os.path.join(
                    self.config.chroma_persist_dir, "text"
                )
            )
            
            # 图像向量存储
            self.image_store = Chroma(
                collection_name="multimodal_image",
                embedding_function=self.vision_embeddings,
                persist_directory=os.path.join(
                    self.config.chroma_persist_dir, "image"
                )
            )
            
            # 表格向量存储
            self.table_store = Chroma(
                collection_name="multimodal_table",
                embedding_function=self.text_embeddings,
                persist_directory=os.path.join(
                    self.config.chroma_persist_dir, "table"
                )
            )
        else:
            # 其他向量存储类型的实现
            raise NotImplementedError(
                f"Vector store type {self.config.vector_store_type} not implemented"
            )
        
        # 文档存储(用于存储原始文档)
        self.docstore = InMemoryStore()
        
    def _initialize_retrievers(self):
        """初始化检索器"""
        # 多向量检索器
        self.text_retriever = MultiVectorRetriever(
            vectorstore=self.text_store,
            docstore=self.docstore,
            id_key="doc_id",
            search_kwargs={"k": self.config.retrieval_top_k}
        )
        
        self.image_retriever = MultiVectorRetriever(
            vectorstore=self.image_store,
            docstore=self.docstore,
            id_key="doc_id",
            search_kwargs={"k": self.config.retrieval_top_k}
        )
        
        self.table_retriever = MultiVectorRetriever(
            vectorstore=self.table_store,
            docstore=self.docstore,
            id_key="doc_id",
            search_kwargs={"k": self.config.retrieval_top_k}
        )
    
    async def add_documents(self, processed_doc: ProcessedDocument) -> bool:
        """添加处理后的文档到向量存储
        
        Args:
            processed_doc: 处理后的文档
            
        Returns:
            是否成功添加
        """
        try:
            doc_id = processed_doc.doc_id
            
            # 存储文本块
            if processed_doc.texts:
                text_docs = []
                for idx, text_chunk in enumerate(processed_doc.texts):
                    doc = Document(
                        page_content=text_chunk.get("content", ""),
                        metadata={
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}_text_{idx}",
                            "source": processed_doc.source_file,
                            "content_type": "text",
                            **text_chunk.get("metadata", {})
                        }
                    )
                    text_docs.append(doc)
                
                # 添加到文本存储
                if text_docs:
                    self.text_store.add_documents(text_docs)
            
            # 存储图像描述
            if processed_doc.images:
                image_docs = []
                for idx, image in enumerate(processed_doc.images):
                    doc = Document(
                        page_content=image.get("description", ""),
                        metadata={
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}_image_{idx}",
                            "source": processed_doc.source_file,
                            "content_type": "image",
                            "image_path": image.get("path", ""),
                            **image.get("metadata", {})
                        }
                    )
                    image_docs.append(doc)
                
                # 添加到图像存储
                if image_docs:
                    self.image_store.add_documents(image_docs)
            
            # 存储表格数据
            if processed_doc.tables:
                table_docs = []
                for idx, table in enumerate(processed_doc.tables):
                    # 将表格转换为文本描述
                    table_text = self._table_to_text(table)
                    doc = Document(
                        page_content=table_text,
                        metadata={
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}_table_{idx}",
                            "source": processed_doc.source_file,
                            "content_type": "table",
                            "table_data": table.get("data", {}),
                            **table.get("metadata", {})
                        }
                    )
                    table_docs.append(doc)
                
                # 添加到表格存储
                if table_docs:
                    self.table_store.add_documents(table_docs)
            
            # 存储文档摘要
            if processed_doc.summary:
                summary_doc = Document(
                    page_content=processed_doc.summary,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_summary",
                        "source": processed_doc.source_file,
                        "content_type": "summary",
                        "keywords": processed_doc.keywords
                    }
                )
                self.text_store.add_documents([summary_doc])
            
            # 存储原始文档到docstore
            self.docstore.mset([(doc_id, processed_doc.dict())])
            
            logger.info(f"Successfully added document {doc_id} to vector stores")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector stores: {e}")
            return False
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """将表格转换为文本描述
        
        Args:
            table: 表格数据
            
        Returns:
            表格的文本描述
        """
        # 简单的表格到文本转换
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        text_parts = []
        if headers:
            text_parts.append("Headers: " + ", ".join(headers))
        
        for idx, row in enumerate(rows[:5]):  # 只取前5行作为示例
            text_parts.append(f"Row {idx+1}: " + ", ".join(str(v) for v in row))
        
        if len(rows) > 5:
            text_parts.append(f"... and {len(rows)-5} more rows")
        
        return "\n".join(text_parts)
    
    async def search(
        self,
        query: str,
        search_type: str = "all",
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResults:
        """搜索向量存储
        
        Args:
            query: 查询文本
            search_type: 搜索类型 (text, image, table, all)
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            检索结果
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.config.retrieval_top_k
        results = RetrievalResults()
        
        try:
            # 文本搜索
            if search_type in ["text", "all"]:
                text_results = await self._search_text(query, top_k, filters)
                results.texts = text_results
            
            # 图像搜索
            if search_type in ["image", "all"]:
                image_results = await self._search_images(query, top_k, filters)
                results.images = image_results
            
            # 表格搜索
            if search_type in ["table", "all"]:
                table_results = await self._search_tables(query, top_k, filters)
                results.tables = table_results
            
            # 收集来源
            sources = set()
            for item in results.texts + results.images + results.tables:
                if "source" in item.get("metadata", {}):
                    sources.add(item["metadata"]["source"])
            results.sources = list(sources)
            
            # 统计信息
            results.total_results = len(results.texts) + len(results.images) + len(results.tables)
            results.retrieval_time_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
        
        return results
    
    async def _search_text(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索文本内容"""
        try:
            # 使用相似度搜索
            docs = self.text_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filters
            )
            
            results = []
            for doc, score in docs:
                if score >= self.config.similarity_threshold:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []
    
    async def _search_images(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索图像内容"""
        try:
            docs = self.image_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filters
            )
            
            results = []
            for doc, score in docs:
                if score >= self.config.similarity_threshold:
                    results.append({
                        "description": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []
    
    async def _search_tables(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索表格内容"""
        try:
            docs = self.table_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filters
            )
            
            results = []
            for doc, score in docs:
                if score >= self.config.similarity_threshold:
                    results.append({
                        "description": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                        "table_data": doc.metadata.get("table_data", {})
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching tables: {e}")
            return []
    
    def clear_all(self):
        """清空所有向量存储"""
        try:
            # 清空各个存储
            self.text_store.delete_collection()
            self.image_store.delete_collection()
            self.table_store.delete_collection()
            
            # 重新初始化
            self._initialize_stores()
            self._initialize_retrievers()
            
            logger.info("All vector stores cleared")
        except Exception as e:
            logger.error(f"Error clearing vector stores: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            stats = {
                "text_documents": len(self.text_store.get()),
                "image_documents": len(self.image_store.get()),
                "table_documents": len(self.table_store.get()),
                "total_documents": len(self.docstore.store),
                "embedding_dimension": self.config.embedding_dimension
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}