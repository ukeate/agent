"""智能内容分块系统"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import nltk
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    Language
)

logger = logging.getLogger(__name__)

# 尝试下载NLTK数据
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


class ChunkStrategy(Enum):
    """分块策略枚举"""
    SEMANTIC = "semantic"  # 语义分块
    FIXED = "fixed"  # 固定大小分块
    ADAPTIVE = "adaptive"  # 自适应分块
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口
    HIERARCHICAL = "hierarchical"  # 层次分块


@dataclass
class DocumentChunk:
    """文档块数据类"""
    chunk_id: str
    content: str
    chunk_index: int
    chunk_type: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: Optional[List[str]] = None
    overlap_with_previous: int = 0
    overlap_with_next: int = 0


class IntelligentChunker:
    """智能内容分块器
    
    支持多种分块策略，保持上下文连贯性
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: ChunkStrategy = ChunkStrategy.SEMANTIC,
        preserve_structure: bool = True,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ):
        """初始化智能分块器
        
        Args:
            chunk_size: 目标块大小
            chunk_overlap: 块重叠大小
            strategy: 分块策略
            preserve_structure: 是否保留文档结构
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.preserve_structure = preserve_structure
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # 初始化不同的分割器
        self._init_splitters()
    
    def _init_splitters(self):
        """初始化各种文本分割器"""
        # 递归字符分割器（默认）
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        # 字符分割器
        self.char_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        
        # Token分割器
        try:
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size // 4,  # 估算token数
                chunk_overlap=self.chunk_overlap // 4
            )
        except:
            self.token_splitter = None
        
        # Markdown分割器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        
        # 代码分割器
        self.python_splitter = PythonCodeTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    async def chunk_document(
        self,
        content: str,
        content_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """对文档内容进行分块
        
        Args:
            content: 文档内容
            content_type: 内容类型
            metadata: 元数据
            
        Returns:
            文档块列表
        """
        metadata = metadata or {}
        
        # 根据策略选择分块方法
        if self.strategy == ChunkStrategy.SEMANTIC:
            chunks = await self._semantic_chunking(content, content_type)
        elif self.strategy == ChunkStrategy.ADAPTIVE:
            chunks = await self._adaptive_chunking(content, content_type, metadata)
        elif self.strategy == ChunkStrategy.SLIDING_WINDOW:
            chunks = await self._sliding_window_chunking(content)
        elif self.strategy == ChunkStrategy.HIERARCHICAL:
            chunks = await self._hierarchical_chunking(content, content_type)
        else:  # FIXED
            chunks = await self._fixed_chunking(content)
        
        # 后处理：添加元数据和关系
        processed_chunks = self._post_process_chunks(chunks, metadata)
        
        return processed_chunks
    
    async def _semantic_chunking(
        self,
        content: str,
        content_type: str
    ) -> List[DocumentChunk]:
        """语义分块：基于段落、章节等逻辑单元
        
        Args:
            content: 文档内容
            content_type: 内容类型
            
        Returns:
            文档块列表
        """
        chunks = []
        
        # 根据内容类型选择合适的分割器
        if content_type == "markdown":
            # 使用Markdown分割器
            md_chunks = self.markdown_splitter.split_text(content)
            for idx, chunk_dict in enumerate(md_chunks):
                chunk_content = chunk_dict.get("content", "")
                if not chunk_content:
                    continue
                
                # 如果块太大，进一步分割
                if len(chunk_content) > self.max_chunk_size:
                    sub_chunks = self.recursive_splitter.split_text(chunk_content)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        chunks.append(self._create_chunk(
                            content=sub_chunk,
                            index=len(chunks),
                            chunk_type="markdown_section",
                            metadata={"headers": chunk_dict.get("metadata", {})}
                        ))
                else:
                    chunks.append(self._create_chunk(
                        content=chunk_content,
                        index=len(chunks),
                        chunk_type="markdown_section",
                        metadata={"headers": chunk_dict.get("metadata", {})}
                    ))
        
        elif content_type == "code" or content_type == "python":
            # 使用代码分割器
            if self.python_splitter:
                code_chunks = self.python_splitter.split_text(content)
                for idx, chunk in enumerate(code_chunks):
                    chunks.append(self._create_chunk(
                        content=chunk,
                        index=idx,
                        chunk_type="code_block",
                        metadata={"language": "python"}
                    ))
            else:
                # 降级到递归分割
                chunks = await self._fixed_chunking(content)
        
        else:
            # 使用智能段落检测
            paragraphs = self._detect_paragraphs(content)
            
            for para_idx, paragraph in enumerate(paragraphs):
                # 如果段落太长，进一步分割
                if len(paragraph) > self.max_chunk_size:
                    sentences = self._split_sentences(paragraph)
                    current_chunk = []
                    current_size = 0
                    
                    for sentence in sentences:
                        if current_size + len(sentence) > self.chunk_size and current_chunk:
                            chunks.append(self._create_chunk(
                                content=" ".join(current_chunk),
                                index=len(chunks),
                                chunk_type="paragraph_part",
                                metadata={"paragraph_index": para_idx}
                            ))
                            # 保留部分重叠
                            if self.chunk_overlap > 0:
                                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                                current_chunk = overlap_sentences + [sentence]
                                current_size = sum(len(s) for s in current_chunk)
                            else:
                                current_chunk = [sentence]
                                current_size = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_size += len(sentence)
                    
                    # 添加最后的块
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            content=" ".join(current_chunk),
                            index=len(chunks),
                            chunk_type="paragraph_part",
                            metadata={"paragraph_index": para_idx}
                        ))
                
                elif len(paragraph) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        content=paragraph,
                        index=len(chunks),
                        chunk_type="paragraph",
                        metadata={"paragraph_index": para_idx}
                    ))
        
        return chunks
    
    async def _adaptive_chunking(
        self,
        content: str,
        content_type: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """自适应分块：根据内容密度动态调整
        
        Args:
            content: 文档内容
            content_type: 内容类型
            metadata: 元数据
            
        Returns:
            文档块列表
        """
        chunks = []
        
        # 计算内容密度
        density = self._calculate_content_density(content)
        
        # 根据密度调整块大小
        if density > 0.8:  # 高密度内容（如代码、表格）
            adjusted_chunk_size = int(self.chunk_size * 0.7)
        elif density < 0.3:  # 低密度内容（如对话、列表）
            adjusted_chunk_size = int(self.chunk_size * 1.3)
        else:
            adjusted_chunk_size = self.chunk_size
        
        # 确保在允许范围内
        adjusted_chunk_size = max(
            self.min_chunk_size,
            min(adjusted_chunk_size, self.max_chunk_size)
        )
        
        # 使用调整后的大小进行分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=adjusted_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        text_chunks = splitter.split_text(content)
        
        for idx, chunk in enumerate(text_chunks):
            chunks.append(self._create_chunk(
                content=chunk,
                index=idx,
                chunk_type="adaptive",
                metadata={
                    "density": density,
                    "adjusted_size": adjusted_chunk_size
                }
            ))
        
        return chunks
    
    async def _sliding_window_chunking(self, content: str) -> List[DocumentChunk]:
        """滑动窗口分块：高重叠率保持上下文
        
        Args:
            content: 文档内容
            
        Returns:
            文档块列表
        """
        chunks = []
        
        # 增加重叠率
        high_overlap = min(int(self.chunk_size * 0.5), 500)
        
        # 计算滑动步长
        step_size = self.chunk_size - high_overlap
        
        start = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            
            # 尝试在句子边界结束
            if end < len(content):
                # 查找最近的句子结束
                last_period = chunk_content.rfind('. ')
                if last_period > self.chunk_size * 0.7:
                    chunk_content = chunk_content[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(self._create_chunk(
                content=chunk_content,
                index=len(chunks),
                chunk_type="sliding_window",
                metadata={
                    "window_start": start,
                    "window_end": end,
                    "overlap": high_overlap if start > 0 else 0
                }
            ))
            
            start += step_size
        
        return chunks
    
    async def _hierarchical_chunking(
        self,
        content: str,
        content_type: str
    ) -> List[DocumentChunk]:
        """层次分块：创建多级块结构
        
        Args:
            content: 文档内容
            content_type: 内容类型
            
        Returns:
            文档块列表
        """
        chunks = []
        
        # 第一级：大块（章节级别）
        large_chunks = self._split_by_sections(content)
        
        for section_idx, section in enumerate(large_chunks):
            # 创建父块
            parent_chunk = self._create_chunk(
                content=section[:500] + "..." if len(section) > 500 else section,
                index=len(chunks),
                chunk_type="section",
                metadata={
                    "level": 1,
                    "section_index": section_idx,
                    "full_content_length": len(section)
                }
            )
            chunks.append(parent_chunk)
            parent_id = parent_chunk.chunk_id
            
            # 第二级：中块（段落级别）
            paragraphs = self._detect_paragraphs(section)
            child_ids = []
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph) > self.chunk_size:
                    # 第三级：小块（句子级别）
                    sentences = self._split_sentences(paragraph)
                    para_child_ids = []
                    
                    current_chunk = []
                    for sent in sentences:
                        if sum(len(s) for s in current_chunk) + len(sent) > self.chunk_size:
                            if current_chunk:
                                child_chunk = self._create_chunk(
                                    content=" ".join(current_chunk),
                                    index=len(chunks),
                                    chunk_type="sentence_group",
                                    metadata={
                                        "level": 3,
                                        "section_index": section_idx,
                                        "paragraph_index": para_idx
                                    }
                                )
                                child_chunk.parent_chunk_id = parent_id
                                chunks.append(child_chunk)
                                para_child_ids.append(child_chunk.chunk_id)
                                current_chunk = []
                        current_chunk.append(sent)
                    
                    # 添加剩余句子
                    if current_chunk:
                        child_chunk = self._create_chunk(
                            content=" ".join(current_chunk),
                            index=len(chunks),
                            chunk_type="sentence_group",
                            metadata={
                                "level": 3,
                                "section_index": section_idx,
                                "paragraph_index": para_idx
                            }
                        )
                        child_chunk.parent_chunk_id = parent_id
                        chunks.append(child_chunk)
                        para_child_ids.append(child_chunk.chunk_id)
                    
                    child_ids.extend(para_child_ids)
                
                elif len(paragraph) >= self.min_chunk_size:
                    # 直接作为第二级块
                    child_chunk = self._create_chunk(
                        content=paragraph,
                        index=len(chunks),
                        chunk_type="paragraph",
                        metadata={
                            "level": 2,
                            "section_index": section_idx,
                            "paragraph_index": para_idx
                        }
                    )
                    child_chunk.parent_chunk_id = parent_id
                    chunks.append(child_chunk)
                    child_ids.append(child_chunk.chunk_id)
            
            # 更新父块的子块ID
            parent_chunk.child_chunk_ids = child_ids
        
        return chunks
    
    async def _fixed_chunking(self, content: str) -> List[DocumentChunk]:
        """固定大小分块
        
        Args:
            content: 文档内容
            
        Returns:
            文档块列表
        """
        chunks = []
        
        text_chunks = self.recursive_splitter.split_text(content)
        
        for idx, chunk in enumerate(text_chunks):
            chunks.append(self._create_chunk(
                content=chunk,
                index=idx,
                chunk_type="fixed",
                metadata={}
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        index: int,
        chunk_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """创建文档块
        
        Args:
            content: 块内容
            index: 块索引
            chunk_type: 块类型
            metadata: 元数据
            
        Returns:
            文档块
        """
        import hashlib
        
        # 生成块ID
        chunk_id = f"chunk_{hashlib.md5(f'{index}:{content[:50]}'.encode()).hexdigest()[:8]}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            chunk_index=index,
            chunk_type=chunk_type,
            metadata=metadata or {},
            start_char=0,  # 将在后处理中更新
            end_char=len(content),  # 将在后处理中更新
        )
    
    def _detect_paragraphs(self, content: str) -> List[str]:
        """检测段落
        
        Args:
            content: 文本内容
            
        Returns:
            段落列表
        """
        # 使用双换行符分割
        paragraphs = re.split(r'\n\s*\n', content)
        
        # 过滤空段落
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子
        
        Args:
            text: 文本内容
            
        Returns:
            句子列表
        """
        try:
            # 使用NLTK分句
            import nltk
            sentences = nltk.sent_tokenize(text)
        except:
            # 降级到简单分句
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _split_by_sections(self, content: str) -> List[str]:
        """按章节分割
        
        Args:
            content: 文本内容
            
        Returns:
            章节列表
        """
        # 查找章节标记
        section_patterns = [
            r'^#{1,3}\s+.+$',  # Markdown标题
            r'^Chapter\s+\d+',  # Chapter标记
            r'^Section\s+\d+',  # Section标记
            r'^\d+\.\s+.+$',  # 数字标题
        ]
        
        sections = []
        current_section = []
        
        for line in content.splitlines():
            is_section_start = False
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_section_start = True
                    break
            
            if is_section_start and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # 添加最后的章节
        if current_section:
            sections.append('\n'.join(current_section))
        
        # 如果没有找到章节，返回整个内容
        if not sections:
            sections = [content]
        
        return sections
    
    def _calculate_content_density(self, content: str) -> float:
        """计算内容密度
        
        Args:
            content: 文本内容
            
        Returns:
            密度值（0-1）
        """
        if not content:
            return 0.0
        
        # 计算各种指标
        total_chars = len(content)
        alpha_chars = sum(1 for c in content if c.isalpha())
        digit_chars = sum(1 for c in content if c.isdigit())
        space_chars = sum(1 for c in content if c.isspace())
        special_chars = total_chars - alpha_chars - digit_chars - space_chars
        
        # 计算密度
        if total_chars == 0:
            return 0.0
        
        # 高密度：更多特殊字符和数字（如代码、公式）
        # 低密度：更多空格和短句（如对话、列表）
        density = (special_chars + digit_chars * 0.5) / total_chars
        
        return min(1.0, max(0.0, density))
    
    def _post_process_chunks(
        self,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """后处理块：添加元数据和关系
        
        Args:
            chunks: 原始块列表
            metadata: 全局元数据
            
        Returns:
            处理后的块列表
        """
        if not chunks:
            return chunks
        
        # 计算字符位置
        current_pos = 0
        for chunk in chunks:
            chunk.start_char = current_pos
            chunk.end_char = current_pos + len(chunk.content)
            current_pos = chunk.end_char
            
            # 添加全局元数据
            chunk.metadata.update(metadata)
        
        # 计算重叠
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]
            
            # 查找内容重叠
            overlap = self._find_overlap(current.content, next_chunk.content)
            if overlap:
                current.overlap_with_next = len(overlap)
                next_chunk.overlap_with_previous = len(overlap)
        
        return chunks
    
    def _find_overlap(self, text1: str, text2: str) -> str:
        """查找两个文本的重叠部分
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            重叠文本
        """
        max_overlap = min(len(text1), len(text2), self.chunk_overlap)
        
        for i in range(max_overlap, 0, -1):
            if text1[-i:] == text2[:i]:
                return text1[-i:]
        
        return ""