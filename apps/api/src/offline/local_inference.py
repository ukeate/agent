"""
离线本地推理引擎

集成本地模型推理，支持：
- 离线CoT推理
- 推理结果缓存
- 降级策略
- 推理优化
"""

import asyncio
import json
import hashlib
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum

from .model_cache import ModelCacheManager
from ..models.schemas.offline import OfflineMode, NetworkStatus


class InferenceMode(str, Enum):
    """推理模式枚举"""
    LOCAL_ONLY = "local_only"
    REMOTE_ONLY = "remote_only"
    HYBRID = "hybrid"
    AUTO = "auto"


class ModelType(str, Enum):
    """模型类型枚举"""
    LANGUAGE_MODEL = "language_model"
    EMBEDDING_MODEL = "embedding_model"
    REASONING_MODEL = "reasoning_model"
    VISION_MODEL = "vision_model"


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    model_type: ModelType
    prompt: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


@dataclass
class InferenceResult:
    """推理结果"""
    request_id: str
    response: str
    model_used: str
    inference_time_ms: float
    token_count: int
    is_cached: bool
    confidence_score: Optional[float] = None
    reasoning_steps: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = None


@dataclass
class CachedInference:
    """缓存的推理结果"""
    prompt_hash: str
    result: InferenceResult
    created_at: datetime
    access_count: int
    last_accessed: datetime


class InferenceCache:
    """推理结果缓存管理器"""
    
    def __init__(self, max_cache_size: int = 1000, ttl_hours: int = 24):
        self.max_cache_size = max_cache_size
        self.ttl_hours = ttl_hours
        self._cache: Dict[str, CachedInference] = {}
    
    def _create_cache_key(self, request: InferenceRequest) -> str:
        """创建缓存键"""
        cache_data = {
            'model_type': request.model_type.value,
            'prompt': request.prompt,
            'parameters': request.parameters,
            'context': request.context
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """获取缓存的推理结果"""
        cache_key = self._create_cache_key(request)
        
        if cache_key not in self._cache:
            return None
        
        cached_item = self._cache[cache_key]
        
        # 检查是否过期
        if utc_now() - cached_item.created_at > timedelta(hours=self.ttl_hours):
            del self._cache[cache_key]
            return None
        
        # 更新访问统计
        cached_item.access_count += 1
        cached_item.last_accessed = utc_now()
        
        # 标记为缓存结果
        result = cached_item.result
        result.is_cached = True
        result.request_id = request.request_id  # 更新请求ID
        
        return result
    
    def put(self, request: InferenceRequest, result: InferenceResult):
        """缓存推理结果"""
        cache_key = self._create_cache_key(request)
        
        # 清理空间
        self._cleanup_cache()
        
        cached_item = CachedInference(
            prompt_hash=cache_key,
            result=result,
            created_at=utc_now(),
            access_count=0,
            last_accessed=utc_now()
        )
        
        self._cache[cache_key] = cached_item
    
    def _cleanup_cache(self):
        """清理缓存空间"""
        if len(self._cache) < self.max_cache_size:
            return
        
        # 按访问时间和频率排序
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        # 删除最旧的条目
        items_to_remove = len(self._cache) - int(self.max_cache_size * 0.8)
        for i in range(items_to_remove):
            cache_key, _ = sorted_items[i]
            del self._cache[cache_key]
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if not self._cache:
            return {
                'total_entries': 0,
                'cache_hit_rate': 0.0,
                'average_access_count': 0.0
            }
        
        total_access = sum(item.access_count for item in self._cache.values())
        avg_access = total_access / len(self._cache)
        
        return {
            'total_entries': len(self._cache),
            'total_access_count': total_access,
            'average_access_count': avg_access,
            'oldest_entry': min(item.created_at for item in self._cache.values()),
            'newest_entry': max(item.created_at for item in self._cache.values())
        }


class LocalInferenceEngine:
    """本地推理引擎"""
    
    def __init__(self, model_cache: ModelCacheManager):
        self.model_cache = model_cache
        self.inference_cache = InferenceCache()
        self.current_mode = InferenceMode.AUTO
        self.network_status = NetworkStatus.UNKNOWN
        
        # 模型实例缓存
        self._loaded_models: Dict[str, Any] = {}
        
        # 性能统计
        self._inference_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'local_inferences': 0,
            'remote_inferences': 0,
            'failed_inferences': 0,
            'average_inference_time': 0.0
        }
    
    def set_network_status(self, status: NetworkStatus):
        """设置网络状态"""
        self.network_status = status
        
        # 自动调整推理模式
        if self.current_mode == InferenceMode.AUTO:
            if status == NetworkStatus.DISCONNECTED:
                self._effective_mode = InferenceMode.LOCAL_ONLY
            elif status == NetworkStatus.WEAK:
                self._effective_mode = InferenceMode.HYBRID
            else:
                self._effective_mode = InferenceMode.REMOTE_ONLY
        else:
            self._effective_mode = self.current_mode
    
    async def _load_model_for_inference(self, model_type: ModelType) -> Optional[Any]:
        """为推理加载模型"""
        # 根据模型类型选择合适的模型ID
        model_mapping = {
            ModelType.LANGUAGE_MODEL: "llm_compressed",
            ModelType.EMBEDDING_MODEL: "embedding_compressed",
            ModelType.REASONING_MODEL: "reasoning_compressed",
            ModelType.VISION_MODEL: "vision_compressed"
        }
        
        model_id = model_mapping.get(model_type)
        if not model_id:
            return None
        
        # 检查是否已加载
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
        
        # 从缓存加载
        model = await self.model_cache.load_model(model_id)
        if model:
            self._loaded_models[model_id] = model
        
        return model
    
    async def _local_inference(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """执行本地推理"""
        start_time = utc_now()
        
        try:
            # 加载模型
            model = await self._load_model_for_inference(request.model_type)
            if not model:
                return None
            
            # 执行推理
            if request.model_type == ModelType.LANGUAGE_MODEL:
                response = await self._language_model_inference(model, request)
            elif request.model_type == ModelType.REASONING_MODEL:
                response = await self._reasoning_model_inference(model, request)
            elif request.model_type == ModelType.EMBEDDING_MODEL:
                response = await self._embedding_model_inference(model, request)
            elif request.model_type == ModelType.VISION_MODEL:
                response = await self._vision_model_inference(model, request)
            else:
                return None
            
            # 计算推理时间
            inference_time = (utc_now() - start_time).total_seconds() * 1000
            
            result = InferenceResult(
                request_id=request.request_id,
                response=response.get('text', ''),
                model_used=f"local_{request.model_type.value}",
                inference_time_ms=inference_time,
                token_count=response.get('token_count', 0),
                is_cached=False,
                confidence_score=response.get('confidence', None),
                reasoning_steps=response.get('reasoning_steps', None),
                metadata={'inference_mode': 'local'}
            )
            
            self._inference_stats['local_inferences'] += 1
            return result
            
        except Exception as e:
            print(f"Local inference failed: {e}")
            return None
    
    async def _language_model_inference(self, model: Any, request: InferenceRequest) -> Dict[str, Any]:
        """语言模型推理"""
        # 这里是模拟实现，实际需要根据具体模型API调用
        prompt = request.prompt
        
        # 简单的CoT推理模拟
        if "step by step" in prompt.lower() or "reasoning" in request.parameters:
            reasoning_steps = [
                {"step": 1, "thought": "分析问题", "content": "理解用户的问题"},
                {"step": 2, "thought": "制定策略", "content": "确定解决方法"},
                {"step": 3, "thought": "执行推理", "content": "应用逻辑推理"},
                {"step": 4, "thought": "生成答案", "content": "形成最终回复"}
            ]
        else:
            reasoning_steps = None
        
        # 模拟推理延时
        await asyncio.sleep(0.1)
        
        return {
            'text': f"Local inference response for: {prompt[:100]}...",
            'token_count': len(prompt.split()) + 20,
            'confidence': 0.85,
            'reasoning_steps': reasoning_steps
        }
    
    async def _reasoning_model_inference(self, model: Any, request: InferenceRequest) -> Dict[str, Any]:
        """推理模型推理"""
        # 实现Chain of Thought推理
        prompt = request.prompt
        
        # 分解推理步骤
        reasoning_steps = []
        
        # 步骤1：问题分解
        reasoning_steps.append({
            "step": 1,
            "type": "problem_decomposition",
            "thought": "分解问题为子问题",
            "content": "将复杂问题分解为可管理的部分"
        })
        
        # 步骤2：信息收集
        reasoning_steps.append({
            "step": 2,
            "type": "information_gathering",
            "thought": "收集相关信息",
            "content": "基于上下文和知识库收集信息"
        })
        
        # 步骤3：逻辑推理
        reasoning_steps.append({
            "step": 3,
            "type": "logical_reasoning",
            "thought": "应用逻辑推理",
            "content": "使用演绎和归纳推理"
        })
        
        # 步骤4：结论生成
        reasoning_steps.append({
            "step": 4,
            "type": "conclusion_generation",
            "thought": "生成最终结论",
            "content": "综合信息形成答案"
        })
        
        await asyncio.sleep(0.2)  # 推理时间更长
        
        return {
            'text': f"Reasoning result: {prompt[:100]}...",
            'token_count': len(prompt.split()) + 50,
            'confidence': 0.9,
            'reasoning_steps': reasoning_steps
        }
    
    async def _embedding_model_inference(self, model: Any, request: InferenceRequest) -> Dict[str, Any]:
        """嵌入模型推理"""
        await asyncio.sleep(0.05)
        
        # 模拟向量嵌入
        import random
        embedding = [random.random() for _ in range(768)]
        
        return {
            'text': json.dumps(embedding),
            'token_count': len(request.prompt.split()),
            'confidence': 0.95
        }
    
    async def _vision_model_inference(self, model: Any, request: InferenceRequest) -> Dict[str, Any]:
        """视觉模型推理"""
        await asyncio.sleep(0.3)
        
        return {
            'text': f"Vision analysis: {request.prompt[:50]}...",
            'token_count': 30,
            'confidence': 0.8
        }
    
    async def _remote_inference(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """执行远程推理（模拟）"""
        # 这里应该调用远程API
        # 为了演示，我们模拟网络调用
        
        if self.network_status == NetworkStatus.DISCONNECTED:
            return None
        
        await asyncio.sleep(0.5)  # 模拟网络延时
        
        start_time = utc_now()
        inference_time = (utc_now() - start_time).total_seconds() * 1000
        
        result = InferenceResult(
            request_id=request.request_id,
            response=f"Remote inference response for: {request.prompt[:100]}...",
            model_used=f"remote_{request.model_type.value}",
            inference_time_ms=inference_time,
            token_count=len(request.prompt.split()) + 25,
            is_cached=False,
            confidence_score=0.95,
            metadata={'inference_mode': 'remote'}
        )
        
        self._inference_stats['remote_inferences'] += 1
        return result
    
    async def infer(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """执行推理"""
        self._inference_stats['total_requests'] += 1
        
        # 首先检查缓存
        cached_result = self.inference_cache.get(request)
        if cached_result:
            self._inference_stats['cache_hits'] += 1
            return cached_result
        
        result = None
        
        # 根据模式选择推理方式
        if self._effective_mode == InferenceMode.LOCAL_ONLY:
            result = await self._local_inference(request)
        
        elif self._effective_mode == InferenceMode.REMOTE_ONLY:
            result = await self._remote_inference(request)
        
        elif self._effective_mode == InferenceMode.HYBRID:
            # 先尝试本地，失败则远程
            result = await self._local_inference(request)
            if not result:
                result = await self._remote_inference(request)
        
        elif self._effective_mode == InferenceMode.AUTO:
            # 根据网络状态自动选择
            if self.network_status == NetworkStatus.CONNECTED:
                result = await self._remote_inference(request)
            else:
                result = await self._local_inference(request)
        
        if not result:
            self._inference_stats['failed_inferences'] += 1
            return None
        
        # 缓存结果
        self.inference_cache.put(request, result)
        
        # 更新统计
        self._update_inference_stats(result.inference_time_ms)
        
        return result
    
    def _update_inference_stats(self, inference_time: float):
        """更新推理统计"""
        total_inferences = (
            self._inference_stats['local_inferences'] + 
            self._inference_stats['remote_inferences']
        )
        
        if total_inferences > 0:
            current_avg = self._inference_stats['average_inference_time']
            self._inference_stats['average_inference_time'] = (
                (current_avg * (total_inferences - 1) + inference_time) / total_inferences
            )
    
    async def batch_infer(self, requests: List[InferenceRequest]) -> List[Optional[InferenceResult]]:
        """批量推理"""
        tasks = [self.infer(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            result if not isinstance(result, Exception) else None
            for result in results
        ]
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """获取推理统计"""
        cache_stats = self.inference_cache.get_stats()
        
        total_requests = self._inference_stats['total_requests']
        cache_hit_rate = (
            self._inference_stats['cache_hits'] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            **self._inference_stats,
            'cache_hit_rate': cache_hit_rate,
            'current_mode': self._effective_mode.value,
            'network_status': self.network_status.value,
            'loaded_models': list(self._loaded_models.keys()),
            'cache_stats': cache_stats
        }
    
    def clear_cache(self):
        """清空推理缓存"""
        self.inference_cache.clear()
    
    def unload_models(self):
        """卸载模型"""
        self._loaded_models.clear()
    
    def set_inference_mode(self, mode: InferenceMode):
        """设置推理模式"""
        self.current_mode = mode
        if mode != InferenceMode.AUTO:
            self._effective_mode = mode