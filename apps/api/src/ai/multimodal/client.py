"""
OpenAI GPT-4o多模态API客户端
支持图像、文档、视频等多种内容类型的处理
"""

import base64
import json
import time
from typing import Dict, Any, List, Optional, Union, BinaryIO
from pathlib import Path
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
import aiohttp
from src.core.config import get_settings
from src.ai.openai_client import OpenAIClient
from .types import ModelPriority, ModelComplexity, ContentType
from .config import ModelConfig

from src.core.logging import get_logger
logger = get_logger(__name__)

class ModelSelector:
    """智能模型选择器"""
    
    @classmethod
    def select_model(
        cls, 
        content_type: str, 
        priority: ModelPriority = ModelPriority.BALANCED,
        complexity: ModelComplexity = ModelComplexity.MEDIUM,
        requires_file_upload: bool = False
    ) -> str:
        """根据内容类型和优先级选择最合适的模型"""
        
        # 如果需要文件上传功能，排除不支持的模型
        if requires_file_upload:
            available_models = [
                model for model, config in ModelConfig.MODEL_CONFIGS.items()
                if config.get("supports_file_upload", False)
            ]
        else:
            available_models = list(ModelConfig.MODEL_CONFIGS.keys())
        
        # 根据优先级选择模型
        if priority == ModelPriority.COST:
            if complexity == ModelComplexity.SIMPLE:
                return "gpt-5-nano" if "gpt-5-nano" in available_models else "gpt-4o-mini"
            else:
                return "gpt-4o-mini"
        elif priority == ModelPriority.QUALITY:
            if complexity == ModelComplexity.COMPLEX:
                return "gpt-5" if "gpt-5" in ModelConfig.MODEL_CONFIGS else "gpt-4o"
            else:
                return "gpt-4o"
        elif priority == ModelPriority.SPEED:
            return "gpt-5-nano" if "gpt-5-nano" in available_models else "gpt-4o-mini"
        else:  # BALANCED
            if content_type == "pdf":
                return "gpt-4o"  # 最佳PDF处理能力
            elif content_type == "video":
                return "gpt-5" if "gpt-5" in ModelConfig.MODEL_CONFIGS else "gpt-4o"
            elif complexity == ModelComplexity.SIMPLE:
                return "gpt-4o-mini"
            else:
                return "gpt-4o"
    
    @classmethod
    def get_model_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算模型使用成本"""
        config = ModelConfig.get_model_config(model)
        input_cost = (input_tokens / 1000) * config["cost_per_1k_tokens"]["input"]
        output_cost = (output_tokens / 1000) * config["cost_per_1k_tokens"]["output"]
        return input_cost + output_cost

class OpenAIMultimodalClient(OpenAIClient):
    """OpenAI GPT-4o多模态API客户端（继承并扩展基础客户端）"""
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化多模态客户端"""
        super().__init__(api_key)
        self.model_selector = ModelSelector()
        self.base_url = ModelConfig.OPENAI_BASE_URL
        # 为文件上传创建专用session
        self.upload_session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if not self.upload_session:
            self.upload_session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=ModelConfig.DEFAULT_TIMEOUT)
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.upload_session:
            await self.upload_session.close()
            self.upload_session = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)),
    )
    async def process_image(
        self,
        image_data: bytes,
        prompt: str,
        max_tokens: int = 1000,
        priority: ModelPriority = ModelPriority.BALANCED,
        complexity: ModelComplexity = ModelComplexity.MEDIUM,
        detail: str = "high"
    ) -> Dict[str, Any]:
        """处理图像内容"""
        # 编码图像为base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # 智能选择模型
        selected_model = self.model_selector.select_model("image", priority, complexity)
        
        # 构建消息内容
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": detail
                        }
                    }
                ]
            }
        ]
        
        logger.info(
            "开始多模态图像处理",
            model=selected_model,
            image_size=len(image_data),
            detail=detail,
            priority=priority
        )
        
        return await self._process_with_openai(
            messages=messages,
            model=selected_model,
            max_tokens=max_tokens,
            operation_type="图像处理"
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)),
    )
    async def process_document(
        self,
        document_text: Optional[str] = None,
        file_id: Optional[str] = None,
        prompt: str = "",
        max_tokens: int = 1000,
        priority: ModelPriority = ModelPriority.BALANCED,
        complexity: ModelComplexity = ModelComplexity.MEDIUM
    ) -> Dict[str, Any]:
        """处理文档内容 - 支持文本或文件ID"""
        if not document_text and not file_id:
            raise ValueError("必须提供document_text或file_id中的一个")
        
        # 构建消息内容
        content = [{"type": "text", "text": prompt}]
        
        if file_id:
            content.append({
                "type": "text", 
                "text": f"请分析文件ID为 {file_id} 的文档。"
            })
            content_type = "pdf"
        elif document_text:
            content.append({
                "type": "text",
                "text": f"文档内容：\n{document_text}"
            })
            content_type = "text"
        
        # 智能选择模型
        selected_model = self.model_selector.select_model(
            content_type, 
            priority, 
            complexity,
            requires_file_upload=(file_id is not None)
        )
        
        messages = [{"role": "user", "content": content}]
        
        logger.info(
            "开始多模态文档处理",
            model=selected_model,
            has_file_id=bool(file_id),
            text_length=len(document_text) if document_text else 0
        )
        
        return await self._process_with_openai(
            messages=messages,
            model=selected_model,
            max_tokens=max_tokens,
            operation_type="文档处理"
        )
    
    async def upload_file(
        self,
        file_path: Union[str, Path],
        purpose: str = "assistants"
    ) -> Dict[str, Any]:
        """上传文件到OpenAI并返回文件ID"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检查文件大小
        file_size = file_path.stat().st_size
        if file_size > ModelConfig.MAX_FILE_SIZE:
            raise ValueError(f"文件大小超过限制: {file_size} > {ModelConfig.MAX_FILE_SIZE}")
        
        try:
            logger.info(
                "开始上传文件到OpenAI",
                file_name=file_path.name,
                file_size=file_size,
                purpose=purpose
            )
            
            # 确保session存在
            if not self.upload_session:
                await self.__aenter__()
            
            # 准备文件上传
            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=file_path.name)
                data.add_field('purpose', purpose)
                
                # 上传到OpenAI
                async with self.upload_session.post(
                    f"{self.base_url}{ModelConfig.FILE_UPLOAD_ENDPOINT}",
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        file_id = result.get('id')
                        
                        logger.info(
                            "文件上传成功",
                            file_id=file_id,
                            file_name=file_path.name
                        )
                        
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(
                            "文件上传失败",
                            status=response.status,
                            error=error_text
                        )
                        raise Exception(f"文件上传失败: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(
                "文件上传错误",
                error=str(e),
                file_name=file_path.name
            )
            raise
    
    async def process_video_frame(
        self,
        frame_data: bytes,
        frame_index: int,
        prompt: str,
        priority: ModelPriority = ModelPriority.BALANCED
    ) -> Dict[str, Any]:
        """处理视频帧"""
        frame_prompt = f"视频帧 #{frame_index}: {prompt}"
        
        # 使用图像处理方法处理视频帧
        return await self.process_image(
            frame_data,
            frame_prompt,
            max_tokens=500,  # 减少token使用
            priority=priority,
            complexity=ModelComplexity.SIMPLE,  # 视频帧通常使用简单分析
            detail="low"  # 使用低分辨率以节省成本
        )
    
    async def health_check_multimodal(self) -> Dict[str, Any]:
        """检查多模态API健康状态"""
        try:
            # 创建一个简单的1x1像素图像进行测试
            test_image = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            )
            
            result = await self.process_image(
                test_image,
                "这是什么？",
                max_tokens=10,
                priority=ModelPriority.COST,
                complexity=ModelComplexity.SIMPLE
            )
            
            return {
                "healthy": True,
                "multimodal_enabled": True,
                "model_used": result.get("model"),
                "response_received": bool(result.get("content")),
                "duration": result.get("duration", 0)
            }
            
        except Exception as e:
            logger.error("多模态健康检查失败", error=str(e))
            return {
                "healthy": False,
                "multimodal_enabled": False,
                "error": str(e)
            }
    
    async def _process_with_openai(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        operation_type: str = "处理"
    ) -> Dict[str, Any]:
        """通用的OpenAI API调用处理方法（消除重复代码）"""
        start_time = time.time()
        
        try:
            # 设置模型并调用API
            self.model = model
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                user="multimodal_processor"
            )
            
            duration = time.time() - start_time
            
            # 统一的响应处理
            result = {
                "content": response.choices[0].message.content,
                "model": model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": self._extract_usage_info(response.usage),
                "duration": duration,
                "cost": None
            }
            
            # 计算成本
            if result["usage"]:
                result["cost"] = self.model_selector.get_model_cost(
                    model,
                    result["usage"]["prompt_tokens"],
                    result["usage"]["completion_tokens"]
                )
            
            logger.info(
                f"多模态{operation_type}成功",
                model=model,
                duration=f"{duration:.2f}s",
                tokens_used=result["usage"]["total_tokens"] if result["usage"] else 0,
                cost=result["cost"]
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"多模态{operation_type}失败",
                error=str(e),
                model=model,
                duration=f"{duration:.2f}s"
            )
            raise
    
    def _extract_usage_info(self, usage) -> Optional[Dict[str, int]]:
        """提取使用信息的辅助方法"""
        if not usage:
            return None
        
        return {
            "prompt_tokens": usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
            "completion_tokens": usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
            "total_tokens": usage.total_tokens if hasattr(usage, 'total_tokens') else 0,
        }
