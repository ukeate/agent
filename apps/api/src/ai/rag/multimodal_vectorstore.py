"""多模态向量存储管理"""

import os
from typing import List, Dict, Any, Optional
from threading import Lock
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
from .multimodal_config import (
    MultimodalConfig,
    VectorStoreType,
    ProcessedDocument,
    RetrievalResults
)

logger = get_logger(__name__)

class _LazyHuggingFaceEmbeddings:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._lock = Lock()

    def _get(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    self._embeddings = HuggingFaceEmbeddings(model_name=self._model_name)
        return self._embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get().embed_query(text)

class MultimodalVectorStore:
    """统一的多模态向量存储管理"""
    
    def __init__(self, config: MultimodalConfig):
        """初始化多模态向量存储
        
        Args:
            config: 多模态配置
        """
        self.config = config
        self._initialize_stores()
        
    def _initialize_stores(self):
        """初始化向量存储"""
        self.text_embeddings = _LazyHuggingFaceEmbeddings(self.config.text_embedding_model.value)
        self.vision_embeddings = _LazyHuggingFaceEmbeddings(self.config.vision_embedding_model.value)
        
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
    
    async def add_documents(self, processed_doc: ProcessedDocument) -> bool:
        """添加处理后的文档到向量存储
        
        Args:
            processed_doc: 处理后的文档
            
        Returns:
            是否成功添加
        """
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
                self.text_store.add_documents(filter_complex_metadata(text_docs))
            
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
                self.image_store.add_documents(filter_complex_metadata(image_docs))
            
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
                self.table_store.add_documents(filter_complex_metadata(table_docs))
            
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
            self.text_store.add_documents(filter_complex_metadata([summary_doc]))
            
        logger.info(f"Successfully added document {doc_id} to vector stores")
        return True
    
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
        filters = filters or None
        results = RetrievalResults()
        
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

        return results
    
    async def _search_text(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索文本内容"""
        docs = self.text_store.similarity_search_with_relevance_scores(
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
    
    async def _search_images(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索图像内容"""
        docs = self.image_store.similarity_search_with_relevance_scores(
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
    
    async def _search_tables(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索表格内容"""
        docs = self.table_store.similarity_search_with_relevance_scores(
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
    
    def clear_all(self):
        """清空所有向量存储"""
        self.text_store.delete_collection()
        self.image_store.delete_collection()
        self.table_store.delete_collection()

        self._initialize_stores()
        logger.info("All vector stores cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        text_documents = int(self.text_store._collection.count())
        image_documents = int(self.image_store._collection.count())
        table_documents = int(self.table_store._collection.count())
        return {
            "text_documents": text_documents,
            "image_documents": image_documents,
            "table_documents": table_documents,
            "total_documents": text_documents + image_documents + table_documents,
            "embedding_dimension": int(self.config.embedding_dimension),
        }
from src.core.logging import get_logger
