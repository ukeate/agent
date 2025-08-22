"""多模态上下文组装器"""

import logging
import base64
from typing import List, Dict, Any, Optional
import json

from .multimodal_config import RetrievalResults, MultimodalContext

logger = logging.getLogger(__name__)


class MultimodalContextAssembler:
    """多模态上下文组装"""
    
    def __init__(self, max_context_length: int = 8000):
        """初始化上下文组装器
        
        Args:
            max_context_length: 最大上下文长度
        """
        self.max_context_length = max_context_length
    
    def assemble_context(
        self,
        retrieval_results: RetrievalResults,
        query: str,
        include_images: bool = True,
        include_tables: bool = True
    ) -> MultimodalContext:
        """组装多模态上下文
        
        Args:
            retrieval_results: 检索结果
            query: 原始查询
            include_images: 是否包含图像
            include_tables: 是否包含表格
            
        Returns:
            组装后的多模态上下文
        """
        context = MultimodalContext()
        
        # 组装文本上下文
        context.texts = self._format_text_chunks(retrieval_results.texts)
        
        # 处理图像内容
        if include_images and retrieval_results.images:
            context.images = self._encode_images_for_llm(retrieval_results.images)
        
        # 处理表格数据
        if include_tables and retrieval_results.tables:
            context.tables = self._format_table_data(retrieval_results.tables)
        
        # 添加元数据
        context.metadata = self._build_metadata(retrieval_results, query)
        
        # 确保上下文不超过最大长度
        context = self._truncate_context(context)
        
        return context
    
    def _format_text_chunks(self, texts: List[Dict[str, Any]]) -> str:
        """格式化文本块
        
        Args:
            texts: 文本结果列表
            
        Returns:
            格式化的文本字符串
        """
        if not texts:
            return ""
        
        formatted_chunks = []
        
        for idx, text_item in enumerate(texts, 1):
            content = text_item.get("content", "")
            score = text_item.get("score", 0.0)
            metadata = text_item.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            # 格式化单个文本块
            chunk_text = f"""
[文本片段 {idx}] (相关度: {score:.2f})
来源: {source}
内容:
{content}
"""
            formatted_chunks.append(chunk_text.strip())
        
        # 用分隔符连接所有文本块
        return "\n---\n".join(formatted_chunks)
    
    def _encode_images_for_llm(self, images: List[Dict[str, Any]]) -> List[str]:
        """为LLM编码图像
        
        Args:
            images: 图像结果列表
            
        Returns:
            Base64编码的图像列表
        """
        encoded_images = []
        
        for image_item in images[:5]:  # 限制最多5张图片
            # 如果已经有base64编码
            if "base64" in image_item:
                encoded_images.append(image_item["base64"])
            # 如果有图像路径，尝试读取和编码
            elif "image_path" in image_item.get("metadata", {}):
                try:
                    image_path = image_item["metadata"]["image_path"]
                    with open(image_path, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode()
                        encoded_images.append(encoded)
                except Exception as e:
                    logger.warning(f"Failed to encode image: {e}")
            
            # 添加图像描述作为替代文本
            description = image_item.get("description", "")
            if description and not encoded_images:
                # 如果无法获取图像，至少提供描述
                encoded_images.append(f"[图像描述: {description}]")
        
        return encoded_images
    
    def _format_table_data(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化表格数据
        
        Args:
            tables: 表格结果列表
            
        Returns:
            格式化的表格数据列表
        """
        formatted_tables = []
        
        for idx, table_item in enumerate(tables, 1):
            table_data = table_item.get("table_data", {})
            description = table_item.get("description", "")
            metadata = table_item.get("metadata", {})
            score = table_item.get("score", 0.0)
            
            formatted_table = {
                "id": idx,
                "description": description,
                "score": score,
                "source": metadata.get("source", "Unknown")
            }
            
            # 处理表格数据
            if isinstance(table_data, dict):
                # 如果是结构化数据
                if "headers" in table_data and "rows" in table_data:
                    formatted_table["headers"] = table_data["headers"]
                    formatted_table["rows"] = table_data["rows"][:10]  # 限制行数
                    if len(table_data.get("rows", [])) > 10:
                        formatted_table["truncated"] = True
                        formatted_table["total_rows"] = len(table_data["rows"])
                else:
                    formatted_table["data"] = table_data
            else:
                # 如果是文本描述
                formatted_table["text_representation"] = str(table_data)
            
            formatted_tables.append(formatted_table)
        
        return formatted_tables
    
    def _build_metadata(
        self,
        retrieval_results: RetrievalResults,
        query: str
    ) -> Dict[str, Any]:
        """构建上下文元数据
        
        Args:
            retrieval_results: 检索结果
            query: 原始查询
            
        Returns:
            元数据字典
        """
        metadata = {
            "query": query,
            "total_results": retrieval_results.total_results,
            "sources": retrieval_results.sources,
            "retrieval_time_ms": retrieval_results.retrieval_time_ms,
            "result_counts": {
                "texts": len(retrieval_results.texts),
                "images": len(retrieval_results.images),
                "tables": len(retrieval_results.tables)
            }
        }
        
        # 添加质量指标
        if retrieval_results.texts:
            text_scores = [t.get("score", 0) for t in retrieval_results.texts]
            metadata["text_avg_score"] = sum(text_scores) / len(text_scores)
        
        if retrieval_results.images:
            image_scores = [i.get("score", 0) for i in retrieval_results.images]
            metadata["image_avg_score"] = sum(image_scores) / len(image_scores)
        
        if retrieval_results.tables:
            table_scores = [t.get("score", 0) for t in retrieval_results.tables]
            metadata["table_avg_score"] = sum(table_scores) / len(table_scores)
        
        return metadata
    
    def _truncate_context(self, context: MultimodalContext) -> MultimodalContext:
        """截断上下文以适应最大长度
        
        Args:
            context: 原始上下文
            
        Returns:
            截断后的上下文
        """
        # 估算当前上下文长度
        current_length = len(context.texts)
        
        # 为表格预留空间（每个表格估算200字符）
        table_length = len(json.dumps(context.tables, ensure_ascii=False))
        current_length += table_length
        
        # 为图像预留空间（每个图像描述估算100字符）
        image_length = len(context.images) * 100
        current_length += image_length
        
        # 如果超过最大长度，截断文本
        if current_length > self.max_context_length:
            available_text_length = self.max_context_length - table_length - image_length
            if available_text_length > 0:
                context.texts = context.texts[:available_text_length]
                context.metadata["truncated"] = True
            
            # 如果还是太长，减少表格数量
            if current_length > self.max_context_length and context.tables:
                context.tables = context.tables[:3]  # 最多保留3个表格
                context.metadata["tables_truncated"] = True
            
            # 如果还是太长，减少图像数量
            if current_length > self.max_context_length and context.images:
                context.images = context.images[:3]  # 最多保留3张图像
                context.metadata["images_truncated"] = True
        
        return context
    
    def create_prompt_context(
        self,
        context: MultimodalContext,
        query: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """创建用于LLM的提示上下文
        
        Args:
            context: 多模态上下文
            query: 用户查询
            system_prompt: 系统提示（可选）
            
        Returns:
            格式化的提示字符串
        """
        prompt_parts = []
        
        # 添加系统提示
        if system_prompt:
            prompt_parts.append(f"系统指令:\n{system_prompt}\n")
        
        # 添加检索到的文本上下文
        if context.texts:
            prompt_parts.append("相关文本内容:")
            prompt_parts.append(context.texts)
            prompt_parts.append("")
        
        # 添加表格信息
        if context.tables:
            prompt_parts.append("相关表格数据:")
            for table in context.tables:
                table_desc = f"表格 {table['id']}: {table.get('description', '')}"
                prompt_parts.append(table_desc)
                
                if "headers" in table and "rows" in table:
                    # 简单的表格文本表示
                    headers = " | ".join(table["headers"])
                    prompt_parts.append(f"列标题: {headers}")
                    
                    # 显示前几行数据
                    for row_idx, row in enumerate(table["rows"][:3]):
                        row_text = " | ".join(str(cell) for cell in row)
                        prompt_parts.append(f"行 {row_idx+1}: {row_text}")
                    
                    if table.get("truncated"):
                        prompt_parts.append(f"... (共 {table.get('total_rows')} 行)")
                
                prompt_parts.append("")
        
        # 添加图像描述
        if context.images:
            prompt_parts.append("相关图像:")
            for idx, image in enumerate(context.images, 1):
                if isinstance(image, str) and image.startswith("[图像描述:"):
                    prompt_parts.append(f"图像 {idx}: {image}")
                else:
                    prompt_parts.append(f"图像 {idx}: [已提供图像数据]")
            prompt_parts.append("")
        
        # 添加元数据摘要
        if context.metadata:
            prompt_parts.append("检索信息摘要:")
            prompt_parts.append(f"- 总结果数: {context.metadata.get('total_results', 0)}")
            prompt_parts.append(f"- 来源数: {len(context.metadata.get('sources', []))}")
            
            if "text_avg_score" in context.metadata:
                prompt_parts.append(f"- 文本平均相关度: {context.metadata['text_avg_score']:.2f}")
            
            prompt_parts.append("")
        
        # 添加用户查询
        prompt_parts.append(f"用户问题:\n{query}")
        prompt_parts.append("\n请基于以上提供的相关信息回答用户的问题。")
        
        return "\n".join(prompt_parts)
    
    def format_for_streaming(
        self,
        context: MultimodalContext,
        chunk_size: int = 500
    ) -> List[str]:
        """将上下文格式化为流式传输的块
        
        Args:
            context: 多模态上下文
            chunk_size: 每个块的大小
            
        Returns:
            文本块列表
        """
        # 将完整上下文分割成块
        full_text = self.create_prompt_context(context, "")
        
        chunks = []
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks