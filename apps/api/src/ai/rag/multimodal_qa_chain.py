"""多模态RAG问答链"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime

from src.ai.openai_client import get_openai_client
from .multimodal_config import (
    MultimodalConfig,
    QueryContext,
    QAResponse,
    RetrievalResults
)
from .multimodal_vectorstore import MultimodalVectorStore
from .multimodal_query_analyzer import MultimodalQueryAnalyzer
from .retrieval_strategy import SmartRetrievalStrategy
from .context_assembler import MultimodalContextAssembler

logger = logging.getLogger(__name__)


class MultimodalQAChain:
    """多模态RAG问答链"""
    
    def __init__(
        self,
        config: MultimodalConfig,
        retriever: Optional[SmartRetrievalStrategy] = None,
        llm_client: Optional[Any] = None
    ):
        """初始化问答链
        
        Args:
            config: 多模态配置
            retriever: 检索器（可选）
            llm_client: LLM客户端（可选）
        """
        self.config = config
        
        # 初始化组件
        self.vector_store = MultimodalVectorStore(config)
        self.query_analyzer = MultimodalQueryAnalyzer()
        self.retriever = retriever or SmartRetrievalStrategy(
            config, self.vector_store, self.query_analyzer
        )
        self.context_assembler = MultimodalContextAssembler()
        self.llm = llm_client or get_openai_client()
        
        # 缓存
        self._query_cache = {}
        self._cache_max_size = 100
    
    async def arun(
        self,
        query: str,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> QAResponse:
        """异步运行问答链
        
        Args:
            query: 用户查询
            stream: 是否流式响应
            max_tokens: 最大token数
            temperature: 生成温度
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            问答响应
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._get_cache_key(query, kwargs)
        if self.config.cache_enabled and cache_key in self._query_cache:
            cached_response = self._query_cache[cache_key]
            if self._is_cache_valid(cached_response):
                logger.info(f"Cache hit for query: {query[:50]}...")
                cached_response["cache_hit"] = True
                return QAResponse(**cached_response)
        
        try:
            # 分析查询
            query_context = self.query_analyzer.analyze_query(
                query=query,
                files=kwargs.get("context_files", [])
            )
            
            # 检索相关内容
            retrieval_results = await self.retriever.retrieve(query_context)
            
            # 组装上下文
            context = self.context_assembler.assemble_context(
                retrieval_results=retrieval_results,
                query=query,
                include_images=kwargs.get("include_images", True),
                include_tables=kwargs.get("include_tables", True)
            )
            
            # 生成回答
            if stream:
                response = await self._generate_streaming_response(
                    query=query,
                    context=context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
            else:
                response = await self._generate_response(
                    query=query,
                    context=context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
            
            # 构建响应
            processing_time = time.time() - start_time
            qa_response = QAResponse(
                answer=response["content"],
                sources=retrieval_results.sources,
                confidence=response.get("confidence", 0.8),
                processing_time=processing_time,
                context_used={
                    "text_chunks": len(retrieval_results.texts),
                    "images": len(retrieval_results.images),
                    "tables": len(retrieval_results.tables)
                }
            )
            
            # 更新缓存
            if self.config.cache_enabled:
                self._update_cache(cache_key, qa_response.dict())
            
            return qa_response
            
        except Exception as e:
            logger.error(f"Error in QA chain: {e}")
            return QAResponse(
                answer=f"抱歉，处理您的问题时出现错误: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                context_used={}
            )
    
    async def _generate_response(
        self,
        query: str,
        context: Any,
        max_tokens: Optional[int],
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """生成非流式响应
        
        Args:
            query: 查询
            context: 上下文
            max_tokens: 最大token数
            temperature: 温度
            system_prompt: 系统提示
            
        Returns:
            响应字典
        """
        # 构建提示
        prompt = self.context_assembler.create_prompt_context(
            context=context,
            query=query,
            system_prompt=system_prompt or self._get_default_system_prompt()
        )
        
        # 准备消息
        messages = [
            {"role": "system", "content": self._get_default_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # 如果有图像，添加图像内容
        if context.images:
            # GPT-4V格式的图像消息
            for image_data in context.images[:3]:  # 限制图像数量
                if not isinstance(image_data, str) or not image_data.startswith("[图像描述:"):
                    messages[-1]["content"] = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                    break
        
        # 调用LLM
        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens or 1000,
                temperature=temperature,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            content = response.choices[0].message.content
            
            # 计算置信度（基于完成原因和使用的token）
            confidence = self._calculate_confidence(
                response.choices[0].finish_reason,
                response.usage.total_tokens if response.usage else 0
            )
            
            return {
                "content": content,
                "confidence": confidence,
                "model": response.model,
                "usage": response.usage.dict() if response.usage else {}
            }
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    async def _generate_streaming_response(
        self,
        query: str,
        context: Any,
        max_tokens: Optional[int],
        temperature: float,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """生成流式响应
        
        Args:
            query: 查询
            context: 上下文
            max_tokens: 最大token数
            temperature: 温度
            system_prompt: 系统提示
            
        Returns:
            响应字典
        """
        # 构建提示
        prompt = self.context_assembler.create_prompt_context(
            context=context,
            query=query,
            system_prompt=system_prompt or self._get_default_system_prompt()
        )
        
        messages = [
            {"role": "system", "content": self._get_default_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # 调用流式LLM
        try:
            stream = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens or 1000,
                temperature=temperature,
                stream=True
            )
            
            # 收集流式响应
            full_content = []
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_content.append(chunk.choices[0].delta.content)
            
            content = "".join(full_content)
            
            return {
                "content": content,
                "confidence": 0.8,  # 流式响应默认置信度
                "model": "gpt-4o-mini",
                "streaming": True
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise
    
    async def stream_response(
        self,
        query: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式生成响应
        
        Args:
            query: 查询
            **kwargs: 其他参数
            
        Yields:
            响应文本块
        """
        # 分析查询
        query_context = self.query_analyzer.analyze_query(
            query=query,
            files=kwargs.get("context_files", [])
        )
        
        # 检索相关内容
        retrieval_results = await self.retriever.retrieve(query_context)
        
        # 组装上下文
        context = self.context_assembler.assemble_context(
            retrieval_results=retrieval_results,
            query=query
        )
        
        # 构建提示
        prompt = self.context_assembler.create_prompt_context(
            context=context,
            query=query,
            system_prompt=kwargs.get("system_prompt", self._get_default_system_prompt())
        )
        
        messages = [
            {"role": "system", "content": self._get_default_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # 流式调用LLM
        try:
            stream = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"\n\n[错误: {str(e)}]"
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示
        
        Returns:
            系统提示
        """
        return """你是一个有帮助的AI助手，专门处理多模态信息（文本、图像、表格）来回答用户问题。

请遵循以下原则：
1. 基于提供的上下文信息准确回答问题
2. 如果信息不足，诚实说明并提供可能的推断
3. 适当引用来源以增加可信度
4. 对于图像内容，基于描述信息进行合理解释
5. 对于表格数据，准确提取和总结关键信息
6. 保持回答简洁、相关且有条理

请用中文回答。"""
    
    def _calculate_confidence(self, finish_reason: str, total_tokens: int) -> float:
        """计算响应置信度
        
        Args:
            finish_reason: 完成原因
            total_tokens: 使用的token总数
            
        Returns:
            置信度分数
        """
        # 基础置信度
        confidence = 0.7
        
        # 根据完成原因调整
        if finish_reason == "stop":
            confidence += 0.2  # 正常完成
        elif finish_reason == "length":
            confidence -= 0.1  # 达到长度限制
        
        # 根据token使用调整
        if total_tokens > 500:
            confidence += 0.1  # 充分的上下文和回答
        
        return min(max(confidence, 0.0), 1.0)
    
    def _get_cache_key(self, query: str, kwargs: Dict) -> str:
        """生成缓存键
        
        Args:
            query: 查询
            kwargs: 其他参数
            
        Returns:
            缓存键
        """
        import hashlib
        
        # 包含查询和关键参数
        key_parts = [
            query,
            str(kwargs.get("include_images", True)),
            str(kwargs.get("include_tables", True)),
            str(kwargs.get("context_files", []))
        ]
        
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_response: Dict) -> bool:
        """检查缓存是否有效
        
        Args:
            cached_response: 缓存的响应
            
        Returns:
            是否有效
        """
        if "timestamp" not in cached_response:
            return False
        
        # 检查缓存时间
        cache_time = cached_response["timestamp"]
        current_time = datetime.now().timestamp()
        
        # 缓存有效期
        if current_time - cache_time > self.config.cache_ttl_seconds:
            return False
        
        return True
    
    def _update_cache(self, key: str, response: Dict):
        """更新缓存
        
        Args:
            key: 缓存键
            response: 响应数据
        """
        # 添加时间戳
        response["timestamp"] = datetime.now().timestamp()
        
        # 限制缓存大小
        if len(self._query_cache) >= self._cache_max_size:
            # 移除最旧的缓存项
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k].get("timestamp", 0)
            )
            del self._query_cache[oldest_key]
        
        self._query_cache[key] = response
    
    def clear_cache(self):
        """清空缓存"""
        self._query_cache.clear()
        logger.info("Query cache cleared")