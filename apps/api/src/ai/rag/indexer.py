"""
RAG文档索引器
提供文档分块和向量化能力
"""

from typing import Any, Dict, List, Optional
import re
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from .models import Document, DocumentChunk, DocumentMetadata, IndexStats

from src.core.logging import get_logger
logger = get_logger(__name__)

class TextSplitter:
    """文本分割器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """分割文本为块"""
        if not text.strip():
            return []
        
        chunks = []
        current_chunk = ""
        
        # 按分隔符递归分割
        splits = self._split_by_separators(text, self.separators)
        
        for split in splits:
            # 如果单个分割就超过块大小，进一步分割
            if len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 强制按字符分割超长文本
                for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                    chunk = split[i:i + self.chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                continue
            
            # 检查是否可以添加到当前块
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
            else:
                # 当前块已满，开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 处理重叠
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + split
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]  # 过滤太短的块
    
    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """按分隔符递归分割"""
        if not separators:
            return [text]
        
        separator = separators[0]
        if separator == "":
            # 最后的分隔符是空字符，按字符分割
            return list(text)
        
        splits = text.split(separator)
        result = []
        
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                # 递归使用下一个分隔符
                result.extend(self._split_by_separators(split, separators[1:]))
        
        return result

class DocumentIndexer:
    """文档索引器基类"""
    
    def __init__(self, text_splitter: Optional[TextSplitter] = None):
        self.text_splitter = text_splitter or TextSplitter()
        self.stats = IndexStats()
    
    def process_document(self, content: str, metadata: DocumentMetadata) -> Document:
        """处理文档，生成文档块"""
        document = Document(content=content, metadata=metadata)
        
        # 分割文档为块
        chunks_text = self.text_splitter.split_text(content)
        
        current_pos = 0
        for i, chunk_text in enumerate(chunks_text):
            # 找到块在原文中的位置
            start_pos = content.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            chunk = DocumentChunk(
                document_id=document.id,
                content=chunk_text,
                chunk_index=i,
                start_char=start_pos,
                end_char=end_pos,
                metadata={
                    "source": metadata.source,
                    "title": metadata.title,
                    "chunk_size": len(chunk_text)
                }
            )
            
            document.add_chunk(chunk)
            current_pos = end_pos
        
        logger.info(f"文档处理完成: {document.id}, 生成 {len(chunks_text)} 个块")
        return document
    
    def update_stats(self, documents: List[Document]):
        """更新索引统计"""
        self.stats.total_documents = len(documents)
        self.stats.total_chunks = sum(doc.get_chunk_count() for doc in documents)
        self.stats.last_updated = utc_now()
        
        # 计算平均块大小
        if self.stats.total_chunks > 0:
            total_size = sum(
                len(chunk.content) 
                for doc in documents 
                for chunk in doc.chunks
            )
            self.stats.average_chunk_size = total_size / self.stats.total_chunks
        
        # 统计语言和文档类型
        languages = set()
        doc_types = set()
        for doc in documents:
            languages.add(doc.metadata.language)
            doc_types.add(doc.metadata.document_type)
        
        self.stats.languages = list(languages)
        self.stats.document_types = list(doc_types)

class VectorIndexer(DocumentIndexer):
    """向量索引器"""
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        text_splitter: Optional[TextSplitter] = None
    ):
        super().__init__(text_splitter)
        self.embedding_model = embedding_model
        self._indexed_chunks: Dict[str, DocumentChunk] = {}
    
    async def index_document(self, document: Document) -> Document:
        """为文档生成向量索引"""
        if not self.embedding_model:
            logger.warning("未配置嵌入模型，跳过向量化")
            return document
        
        logger.info(f"开始向量化文档: {document.id}")
        
        # 批量生成嵌入向量
        chunk_texts = [chunk.content for chunk in document.chunks]
        
        try:
            embeddings = await self._generate_embeddings(chunk_texts)
            
            # 为每个块设置嵌入向量
            for chunk, embedding in zip(document.chunks, embeddings):
                chunk.set_embedding(embedding)
                self._indexed_chunks[chunk.id] = chunk
            
            self.stats.indexed_chunks += len(document.chunks)
            logger.info(f"文档向量化完成: {document.id}")
            
        except Exception as e:
            logger.error(f"文档向量化失败: {document.id}, 错误: {str(e)}")
            raise
        
        return document
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        if not self.embedding_model:
            raise RuntimeError("embedding model not configured for indexer")
        
        # 调用实际嵌入模型（需实现embed接口）
        if not hasattr(self.embedding_model, "embed"):
            raise RuntimeError("embedding model missing embed() method")
        embeddings = await self.embedding_model.embed(texts)
        return embeddings
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """根据ID获取已索引的块"""
        return self._indexed_chunks.get(chunk_id)
    
    def get_indexed_chunks(self) -> List[DocumentChunk]:
        """获取所有已索引的块"""
        return list(self._indexed_chunks.values())
    
    async def reindex_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """重新索引单个块"""
        if not self.embedding_model:
            return chunk
        
        embeddings = await self._generate_embeddings([chunk.content])
        chunk.set_embedding(embeddings[0])
        self._indexed_chunks[chunk.id] = chunk
        
        return chunk

# 工厂函数
def create_document_indexer(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: Optional[Any] = None
) -> DocumentIndexer:
    """创建文档索引器"""
    text_splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if embedding_model:
        return VectorIndexer(embedding_model, text_splitter)
    else:
        return DocumentIndexer(text_splitter)

# 预定义的文本分割配置
SPLITTER_CONFIGS = {
    "default": {"chunk_size": 1000, "chunk_overlap": 200},
    "small": {"chunk_size": 500, "chunk_overlap": 100},
    "large": {"chunk_size": 2000, "chunk_overlap": 400},
    "code": {
        "chunk_size": 1500, 
        "chunk_overlap": 300,
        "separators": ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]
    },
    "markdown": {
        "chunk_size": 1000,
        "chunk_overlap": 200, 
        "separators": ["\n## ", "\n### ", "\n\n", "\n", "。", ".", " ", ""]
    }
}

def get_text_splitter(config_name: str = "default") -> TextSplitter:
    """获取预配置的文本分割器"""
    config = SPLITTER_CONFIGS.get(config_name, SPLITTER_CONFIGS["default"])
    return TextSplitter(**config)
