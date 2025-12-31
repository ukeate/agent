"""
多模态内容处理器
"""

import json
from pathlib import Path
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Dict, Any, List, Optional, Union
import hashlib
import aiofiles
import cv2
import numpy as np
from PIL import Image
import PyPDF2
from docx import Document
from src.core.redis import get_redis
from .client import OpenAIMultimodalClient
from .types import (
    ContentType, ProcessingStatus, MultimodalContent, 
    ProcessingResult, ProcessingOptions, ModelPriority
)
from .config import ModelConfig

from src.core.logging import get_logger
logger = get_logger(__name__)

class MultimodalProcessor:
    """多模态内容处理器"""
    
    def __init__(
        self, 
        openai_client: OpenAIMultimodalClient,
        storage_path: str = "/tmp/multimodal"
    ):
        self.openai = openai_client
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    async def process_content(
        self,
        content: MultimodalContent,
        options: Optional[ProcessingOptions] = None
    ) -> ProcessingResult:
        """处理多模态内容"""
        start_time = utc_now()
        options = options or ProcessingOptions()
        
        try:
            # 检查缓存
            if options.enable_cache:
                cached_result = await self._check_cache(content.content_id)
                if cached_result:
                    logger.info(f"使用缓存结果: {content.content_id}")
                    cached_result.status = ProcessingStatus.CACHED
                    return cached_result
            
            # 根据内容类型选择处理方法
            if content.content_type == ContentType.IMAGE:
                result_data = await self._process_image(content, options)
            elif content.content_type == ContentType.DOCUMENT:
                result_data = await self._process_document(content, options)
            elif content.content_type == ContentType.VIDEO:
                result_data = await self._process_video(content, options)
            elif content.content_type == ContentType.TEXT:
                result_data = await self._process_text(content, options)
            else:
                raise ValueError(f"不支持的内容类型: {content.content_type}")
            
            # 计算处理时间
            processing_time = (utc_now() - start_time).total_seconds()
            
            # 创建处理结果
            result = ProcessingResult(
                content_id=content.content_id,
                status=ProcessingStatus.COMPLETED,
                extracted_data=result_data.get("extracted_data", {}),
                confidence_score=result_data.get("confidence", 0.8),
                processing_time=processing_time,
                model_used=result_data.get("model"),
                tokens_used=result_data.get("usage"),
                created_at=start_time
            )
            
            # 缓存结果
            if options.enable_cache:
                await self._cache_result(content.content_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"处理内容失败 {content.content_id}: {e}")
            processing_time = (utc_now() - start_time).total_seconds()
            
            return ProcessingResult(
                content_id=content.content_id,
                status=ProcessingStatus.FAILED,
                extracted_data={},
                confidence_score=0.0,
                processing_time=processing_time,
                error_message=str(e),
                created_at=start_time
            )
    
    async def _process_image(
        self,
        content: MultimodalContent,
        options: ProcessingOptions
    ) -> Dict[str, Any]:
        """处理图像内容"""
        # 读取图像文件
        async with aiofiles.open(content.file_path, 'rb') as f:
            image_data = await f.read()
        
        # 构建分析提示
        prompts = []
        if options.extract_text:
            prompts.append("提取所有可见的文本内容")
        if options.extract_objects:
            prompts.append("识别图像中的主要对象和场景")
        if options.extract_sentiment:
            prompts.append("分析图像的情感色调")
        
        base_prompt = f"""请分析这张图像并提供以下信息：
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(prompts))}

请以JSON格式返回结果，包含以下字段：
- description: 图像的详细描述
- objects: 识别到的主要对象列表（如果需要）
- text_content: 提取的文本内容（如果有）
- sentiment: 情感分析（positive/negative/neutral）（如果需要）
- confidence: 置信度分数（0-1）
"""
        
        # 使用GPT-4o处理图像
        openai_result = await self.openai.process_image(
            image_data, 
            base_prompt,
            max_tokens=options.max_tokens,
            priority=options.priority,
            complexity=options.complexity
        )
        
        # 解析响应
        try:
            extracted_data = json.loads(openai_result.get("content", "{}"))
        except json.JSONDecodeError:
            extracted_data = {
                "description": openai_result.get("content", ""),
                "confidence": 0.7
            }
        
        # 添加元数据
        extracted_data.update({
            "file_size": content.file_size,
            "mime_type": content.mime_type,
            "processing_method": "openai_multimodal"
        })
        
        return {
            "extracted_data": extracted_data,
            "model": openai_result.get("model"),
            "usage": openai_result.get("usage"),
            "confidence": extracted_data.get("confidence", 0.8)
        }
    
    async def _process_document(
        self,
        content: MultimodalContent,
        options: ProcessingOptions
    ) -> Dict[str, Any]:
        """处理文档内容"""
        
        # 检查是否有OpenAI文件ID
        openai_file_id = content.metadata.get('openai_file_id') if content.metadata else None
        
        if openai_file_id and content.file_path.endswith('.pdf'):
            # 使用文件ID处理PDF
            document_text = None
        else:
            # 提取文档文本
            document_text = await self._extract_document_text(content.file_path)
        
        # 文档分析提示
        analysis_prompt = """请分析这份文档并提供以下信息：
1. 文档的主要主题和内容摘要
2. 关键信息点和重要数据
3. 文档的结构和组织方式
4. 任何表格、图表或特殊格式的内容

请以JSON格式返回结果，包含以下字段：
- summary: 文档摘要
- key_points: 关键信息点列表
- structure: 文档结构描述
- data_elements: 提取的数据元素
- document_type: 推断的文档类型
- confidence: 置信度分数（0-1）
"""
        
        # 使用GPT-4o处理文档
        openai_result = await self.openai.process_document(
            document_text=document_text,
            file_id=openai_file_id,
            prompt=analysis_prompt,
            max_tokens=options.max_tokens,
            priority=options.priority,
            complexity=options.complexity
        )
        
        # 解析响应
        try:
            extracted_data = json.loads(openai_result.get("content", "{}"))
        except json.JSONDecodeError:
            extracted_data = {
                "summary": openai_result.get("content", ""),
                "document_type": "text",
                "confidence": 0.7
            }
        
        # 添加元数据
        extracted_data.update({
            "text_length": len(document_text) if document_text else 0,
            "file_size": content.file_size,
            "mime_type": content.mime_type,
            "processing_method": "openai_document_analysis"
        })
        
        return {
            "extracted_data": extracted_data,
            "model": openai_result.get("model"),
            "usage": openai_result.get("usage"),
            "confidence": extracted_data.get("confidence", 0.8)
        }
    
    async def _process_video(
        self,
        content: MultimodalContent,
        options: ProcessingOptions
    ) -> Dict[str, Any]:
        """处理视频内容"""
        # 提取关键帧
        frames = await self._extract_video_frames(
            content.file_path,
            max_frames=ModelConfig.MAX_VIDEO_FRAMES
        )
        
        extracted_data = {
            "frame_count": len(frames),
            "key_frames": [],
            "overall_summary": "",
            "processing_method": "frame_extraction_analysis",
            "confidence": 0.0
        }
        
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        confidence_scores = []
        
        # 分析每个关键帧
        for i, frame_data in enumerate(frames[:5]):  # 限制分析前5帧
            try:
                frame_result = await self.openai.process_video_frame(
                    frame_data,
                    i,
                    "描述这一帧的内容和主要元素",
                    priority=options.priority
                )
                
                # 解析帧分析结果
                frame_content = frame_result.get('content', '')
                
                extracted_data["key_frames"].append({
                    "frame_index": i,
                    "analysis": frame_content,
                    "timestamp": i * ModelConfig.FRAME_EXTRACTION_INTERVAL
                })
                
                # 累计token使用
                if frame_result.get("usage"):
                    for key in total_tokens:
                        total_tokens[key] += frame_result["usage"].get(key, 0)
                
                confidence_scores.append(0.8)  # 默认置信度
                
            except Exception as e:
                logger.warning(f"分析帧 {i} 失败: {e}")
                confidence_scores.append(0.5)
        
        # 生成整体摘要
        if extracted_data["key_frames"]:
            summaries = [f["analysis"] for f in extracted_data["key_frames"]]
            extracted_data["overall_summary"] = " ".join(summaries[:3])
        
        # 计算平均置信度
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.7
        extracted_data["confidence"] = avg_confidence
        
        return {
            "extracted_data": extracted_data,
            "model": "gpt-4o",
            "usage": total_tokens,
            "confidence": avg_confidence
        }
    
    async def _process_text(
        self,
        content: MultimodalContent,
        options: ProcessingOptions
    ) -> Dict[str, Any]:
        """处理纯文本内容"""
        # 读取文本内容
        if content.file_path:
            async with aiofiles.open(content.file_path, 'r', encoding='utf-8') as f:
                text_content = await f.read()
        else:
            text_content = content.metadata.get("text", "") if content.metadata else ""
        
        # 文本分析提示
        analysis_prompt = f"""请分析以下文本：

{text_content[:2000]}  # 限制长度

提供以下信息的JSON格式结果：
- summary: 文本摘要
- key_points: 关键点列表
- sentiment: 情感倾向
- topics: 主要话题
- confidence: 置信度分数（0-1）
"""
        
        # 使用文档处理方法
        openai_result = await self.openai.process_document(
            document_text=text_content,
            prompt=analysis_prompt,
            max_tokens=options.max_tokens,
            priority=options.priority,
            complexity=options.complexity
        )
        
        # 解析响应
        try:
            extracted_data = json.loads(openai_result.get("content", "{}"))
        except json.JSONDecodeError:
            extracted_data = {
                "summary": openai_result.get("content", ""),
                "confidence": 0.7
            }
        
        return {
            "extracted_data": extracted_data,
            "model": openai_result.get("model"),
            "usage": openai_result.get("usage"),
            "confidence": extracted_data.get("confidence", 0.8)
        }
    
    async def _extract_document_text(self, file_path: str) -> str:
        """从文档中提取文本"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt' or suffix == '.md':
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
                    
            elif suffix == '.pdf':
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
                
            elif suffix == '.docx':
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
                
            else:
                # 尝试作为纯文本读取
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return await f.read()
                    
        except Exception as e:
            logger.error(f"提取文档文本失败: {e}")
            return ""
    
    async def _extract_video_frames(
        self, 
        video_path: str, 
        max_frames: int = 10
    ) -> List[bytes]:
        """从视频中提取关键帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // max_frames)
            
            frame_count = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # 调整帧大小以减少处理成本
                    height, width = frame.shape[:2]
                    if width > 1024:
                        scale = 1024 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # 将帧转换为JPEG格式的字节数据
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frames.append(buffer.tobytes())
                
                frame_count += 1
        
        finally:
            cap.release()
        
        logger.info(f"从视频提取了 {len(frames)} 个关键帧")
        return frames
    
    async def _check_cache(self, content_id: str) -> Optional[ProcessingResult]:
        """检查缓存"""
        try:
            redis_client = get_redis()
            cache_key = f"{ModelConfig.CACHE_KEY_PREFIX}{content_id}"
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return ProcessingResult(
                    content_id=data['content_id'],
                    status=ProcessingStatus.CACHED,
                    extracted_data=data['extracted_data'],
                    confidence_score=data['confidence_score'],
                    processing_time=data['processing_time'],
                    model_used=data.get('model_used'),
                    tokens_used=data.get('tokens_used')
                )
        except Exception as e:
            logger.warning(f"检查缓存失败: {e}")
        
        return None
    
    async def _cache_result(self, content_id: str, result: ProcessingResult):
        """缓存处理结果"""
        try:
            redis_client = get_redis()
            cache_key = f"{ModelConfig.CACHE_KEY_PREFIX}{content_id}"
            
            cache_data = {
                'content_id': result.content_id,
                'extracted_data': result.extracted_data,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'model_used': result.model_used,
                'tokens_used': result.tokens_used
            }
            
            await redis_client.setex(
                cache_key,
                ModelConfig.CACHE_TTL,
                json.dumps(cache_data, default=str)
            )
            
            logger.info(f"缓存处理结果: {content_id}")
            
        except Exception as e:
            logger.warning(f"缓存结果失败: {e}")
