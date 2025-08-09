"""
OpenAI API客户端封装
支持GPT-4o-mini模型的调用和工具使用
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..core.config import get_settings
from ..core.constants import TimeoutConstants

logger = structlog.get_logger(__name__)


class OpenAIClient:
    """OpenAI API客户端"""

    def __init__(self, api_key: Optional[str] = None):
        """初始化OpenAI客户端"""
        settings = get_settings()
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥未配置")
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=float(TimeoutConstants.OPENAI_CLIENT_TIMEOUT_SECONDS),  # 使用常量定义的超时时间
            max_retries=0,  # 禁用内置重试，使用自定义重试
        )
        self.model = "gpt-4o-mini"
        self.max_retries = 3
        self.base_delay = 1.0
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)),
    )
    async def create_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """创建聊天完成"""
        start_time = time.time()
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
                
            if tools:
                kwargs["tools"] = tools
                
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
                
            logger.info(
                "开始OpenAI API调用",
                model=self.model,
                message_count=len(messages),
                has_tools=bool(tools),
                temperature=temperature,
            )
                
            response = await self.client.chat.completions.create(**kwargs)
            
            duration = time.time() - start_time
            
            logger.info(
                "OpenAI API调用成功",
                model=self.model,
                duration=f"{duration:.2f}s",
                tokens_used=response.usage.total_tokens if response.usage else 0,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason,
                has_tool_calls=bool(response.choices[0].message.tool_calls),
            )
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                } if response.usage else None,
                "duration": duration,
            }
            
        except openai.RateLimitError as e:
            duration = time.time() - start_time
            logger.error(
                "OpenAI API速率限制",
                error=str(e),
                duration=f"{duration:.2f}s",
                model=self.model,
            )
            raise
        except openai.APITimeoutError as e:
            duration = time.time() - start_time
            logger.error(
                "OpenAI API超时",
                error=str(e),
                duration=f"{duration:.2f}s",
                model=self.model,
            )
            raise
        except openai.AuthenticationError as e:
            duration = time.time() - start_time
            logger.error(
                "OpenAI API认证错误",
                error=str(e),
                duration=f"{duration:.2f}s",
            )
            raise
        except openai.APIError as e:
            duration = time.time() - start_time
            logger.error(
                "OpenAI API错误",
                error=str(e),
                error_type=type(e).__name__,
                duration=f"{duration:.2f}s",
                model=self.model,
            )
            raise
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "OpenAI客户端未知错误",
                error=str(e),
                error_type=type(e).__name__,
                duration=f"{duration:.2f}s",
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)),
    )
    async def create_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """创建流式聊天完成"""
        start_time = time.time()
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
                
            if tools:
                kwargs["tools"] = tools
                
            logger.info(
                "开始OpenAI流式API调用",
                model=self.model,
                message_count=len(messages),
                has_tools=bool(tools),
            )
                
            stream = await self.client.chat.completions.create(**kwargs)
            
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                if chunk.choices[0].delta.content:
                    yield {
                        "content": chunk.choices[0].delta.content,
                        "finish_reason": chunk.choices[0].finish_reason,
                    }
                    
            duration = time.time() - start_time
            logger.info(
                "OpenAI流式API调用完成",
                model=self.model,
                duration=f"{duration:.2f}s",
                chunk_count=chunk_count,
            )
                    
        except openai.RateLimitError as e:
            duration = time.time() - start_time
            logger.error(
                "OpenAI流式API速率限制",
                error=str(e),
                duration=f"{duration:.2f}s",
                model=self.model,
            )
            raise
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "OpenAI流式API错误",
                error=str(e),
                error_type=type(e).__name__,
                duration=f"{duration:.2f}s",
            )
            raise

    async def create_embeddings(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-small",
    ) -> List[List[float]]:
        """创建文本嵌入向量"""
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            response = await self.client.embeddings.create(
                model=model,
                input=texts,
            )
            
            logger.info(
                "OpenAI Embeddings API调用成功",
                model=model,
                texts_count=len(texts),
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0,
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error("OpenAI Embeddings API错误", error=str(e))
            raise

    def format_messages_for_openai(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """格式化消息为OpenAI格式"""
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # 添加历史对话
        if conversation_history:
            messages.extend(conversation_history)
            
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        return messages

    def format_tools_for_openai(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将MCP工具格式转换为OpenAI格式"""
        openai_tools = []
        
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}),
                }
            }
            openai_tools.append(openai_tool)
            
        return openai_tools

    async def health_check(self) -> Dict[str, Any]:
        """检查OpenAI API健康状态"""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            
            duration = time.time() - start_time
            
            result = {
                "healthy": True,
                "duration": duration,
                "model": self.model,
                "response_received": bool(response.choices[0].message.content),
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }
            
            logger.info(
                "OpenAI API健康检查成功",
                duration=f"{duration:.2f}s",
                model=self.model,
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            result = {
                "healthy": False,
                "duration": duration,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            
            logger.error(
                "OpenAI API健康检查失败",
                error=str(e),
                error_type=type(e).__name__,
                duration=f"{duration:.2f}s",
            )
            
            return result


# 全局客户端实例
_openai_client: Optional[OpenAIClient] = None


async def get_openai_client() -> OpenAIClient:
    """获取全局OpenAI客户端实例"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client


async def close_openai_client():
    """关闭OpenAI客户端"""
    global _openai_client
    if _openai_client:
        await _openai_client.client.close()
        _openai_client = None