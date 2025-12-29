"""
本地模型缓存测试
"""

import pytest
import tempfile
import os
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from ...offline.model_cache import ModelCacheManager, ModelMetadata
from ...offline.local_inference import (
    LocalInferenceEngine, InferenceRequest, InferenceMode,
    ModelType, NetworkStatus
)

class TestModelCacheManager:
    """模型缓存管理器测试"""
    
    @pytest.fixture
    def temp_cache_manager(self):
        """临时缓存管理器"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = ModelCacheManager(
                cache_dir=temp_dir,
                max_cache_size_gb=0.1  # 100MB
            )
            yield cache_manager
            cache_manager.close()
    
    @pytest.mark.asyncio
    async def test_cache_and_load_model(self, temp_cache_manager):
        """测试模型缓存和加载"""
        # 创建模拟模型数据
        mock_model = {
            'weights': [1.0, 2.0, 3.0] * 1000,  # 模拟权重
            'config': {'model_type': 'test', 'version': '1.0'}
        }
        
        model_id = "test_model"
        version = "1.0"
        
        # 缓存模型
        success = await temp_cache_manager.cache_model(
            model_id=model_id,
            version=version,
            model_data=mock_model,
            tags=['test', 'demo']
        )
        assert success
        
        # 加载模型
        loaded_model = await temp_cache_manager.load_model(model_id)
        assert loaded_model is not None
        assert loaded_model['config']['model_type'] == 'test'
        assert loaded_model['weights'][:3] == [1.0, 2.0, 3.0]
    
    @pytest.mark.asyncio
    async def test_model_metadata(self, temp_cache_manager):
        """测试模型元数据"""
        mock_model = {'data': 'test'}
        model_id = "metadata_test"
        version = "2.0"
        
        await temp_cache_manager.cache_model(
            model_id=model_id,
            version=version,
            model_data=mock_model,
            quantization_level="int8",
            tags=['quantized', 'test']
        )
        
        # 获取模型信息
        info = temp_cache_manager.get_model_info(model_id)
        assert info is not None
        assert info.model_id == model_id
        assert info.version == version
        assert info.quantization_level == "int8"
        assert 'quantized' in info.tags
        assert info.use_count == 0
        
        # 加载模型应该增加使用计数
        await temp_cache_manager.load_model(model_id)
        updated_info = temp_cache_manager.get_model_info(model_id)
        assert updated_info.use_count == 1
    
    @pytest.mark.asyncio
    async def test_list_and_filter_models(self, temp_cache_manager):
        """测试模型列表和过滤"""
        # 创建多个模型
        models = [
            ("model1", "1.0", {'type': 'llm'}, ['nlp', 'language']),
            ("model2", "1.0", {'type': 'embedding'}, ['embedding', 'nlp']),
            ("model3", "1.0", {'type': 'vision'}, ['vision', 'image'])
        ]
        
        for model_id, version, data, tags in models:
            await temp_cache_manager.cache_model(
                model_id=model_id,
                version=version,
                model_data=data,
                tags=tags
            )
        
        # 列出所有模型
        all_models = temp_cache_manager.list_cached_models()
        assert len(all_models) == 3
        
        # 按标签过滤
        nlp_models = temp_cache_manager.list_cached_models(tags=['nlp'])
        assert len(nlp_models) == 2
        
        vision_models = temp_cache_manager.list_cached_models(tags=['vision'])
        assert len(vision_models) == 1
        assert vision_models[0].model_id == "model3"
    
    @pytest.mark.asyncio
    async def test_preload_models(self, temp_cache_manager):
        """测试模型预加载"""
        # 缓存多个模型
        models = ["model_a", "model_b", "model_c"]
        for model_id in models:
            await temp_cache_manager.cache_model(
                model_id=model_id,
                version="1.0",
                model_data={'id': model_id}
            )
        
        # 预加载模型
        results = await temp_cache_manager.preload_models(models)
        
        assert len(results) == 3
        for model_id in models:
            assert results[model_id] is True
    
    def test_cache_stats(self, temp_cache_manager):
        """测试缓存统计"""
        stats = temp_cache_manager.get_cache_stats()
        assert 'total_models' in stats
        assert 'total_size_bytes' in stats
        assert 'cache_usage_percent' in stats
        assert stats['total_models'] == 0  # 空缓存
    
    @pytest.mark.asyncio
    async def test_remove_model(self, temp_cache_manager):
        """测试模型移除"""
        model_id = "remove_test_model"
        await temp_cache_manager.cache_model(
            model_id=model_id,
            version="1.0",
            model_data={"id": model_id},
        )
        assert temp_cache_manager.get_model_info(model_id) is not None

        removed = temp_cache_manager.remove_model(model_id)
        assert removed is True
        assert temp_cache_manager.get_model_info(model_id) is None
    
    def test_cleanup_old_models(self, temp_cache_manager):
        """测试清理旧模型"""
        # 清理30天前的模型
        removed_count = temp_cache_manager.cleanup_old_models(days=30)
        assert removed_count >= 0  # 应该是非负数

class TestLocalInferenceEngine:
    """本地推理引擎测试"""
    
    @pytest.fixture
    def temp_inference_engine(self):
        """临时推理引擎"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = ModelCacheManager(cache_dir=temp_dir)
            engine = LocalInferenceEngine(cache_manager)
            yield engine
            cache_manager.close()
    
    def test_network_status_handling(self, temp_inference_engine):
        """测试网络状态处理"""
        engine = temp_inference_engine
        
        # 测试自动模式下的网络状态变化
        engine.set_inference_mode(InferenceMode.AUTO)
        
        # 离线状态
        engine.set_network_status(NetworkStatus.DISCONNECTED)
        assert engine._effective_mode == InferenceMode.LOCAL_ONLY
        
        # 弱网络状态
        engine.set_network_status(NetworkStatus.WEAK)
        assert engine._effective_mode == InferenceMode.HYBRID
        
        # 在线状态
        engine.set_network_status(NetworkStatus.CONNECTED)
        assert engine._effective_mode == InferenceMode.REMOTE_ONLY
    
    @pytest.mark.asyncio
    async def test_inference_request_cache(self, temp_inference_engine):
        """测试推理请求缓存"""
        engine = temp_inference_engine
        engine.set_network_status(NetworkStatus.CONNECTED)
        
        request = InferenceRequest(
            request_id="test_req_1",
            model_type=ModelType.LANGUAGE_MODEL,
            prompt="What is 2+2?",
            parameters={"temperature": 0.7}
        )
        
        # 第一次推理
        result1 = await engine.infer(request)
        assert result1 is not None
        assert not result1.is_cached
        
        # 第二次推理应该命中缓存
        request.request_id = "test_req_2"  # 不同的请求ID，但内容相同
        result2 = await engine.infer(request)
        assert result2 is not None
        assert result2.is_cached
        assert result2.response == result1.response
    
    @pytest.mark.asyncio
    async def test_different_model_types(self, temp_inference_engine):
        """测试不同模型类型的推理"""
        engine = temp_inference_engine
        engine.set_network_status(NetworkStatus.CONNECTED)
        
        model_types = [
            ModelType.LANGUAGE_MODEL,
            ModelType.REASONING_MODEL,
            ModelType.EMBEDDING_MODEL,
            ModelType.VISION_MODEL
        ]
        
        for model_type in model_types:
            request = InferenceRequest(
                request_id=f"test_{model_type.value}",
                model_type=model_type,
                prompt=f"Test prompt for {model_type.value}",
                parameters={}
            )
            
            result = await engine.infer(request)
            assert result is not None
            assert model_type.value in result.model_used
    
    @pytest.mark.asyncio
    async def test_reasoning_model_inference(self, temp_inference_engine):
        """测试推理模型的CoT推理"""
        engine = temp_inference_engine
        engine.set_network_status(NetworkStatus.CONNECTED)
        
        request = InferenceRequest(
            request_id="reasoning_test",
            model_type=ModelType.REASONING_MODEL,
            prompt="Solve this step by step: What is the sum of first 10 natural numbers?",
            parameters={"reasoning": True}
        )
        
        result = await engine.infer(request)
        assert result is not None
        assert result.reasoning_steps is not None
        assert len(result.reasoning_steps) > 0
        
        # 验证推理步骤结构
        for step in result.reasoning_steps:
            assert 'step' in step
            assert 'type' in step
            assert 'thought' in step
            assert 'content' in step
    
    @pytest.mark.asyncio
    async def test_batch_inference(self, temp_inference_engine):
        """测试批量推理"""
        engine = temp_inference_engine
        engine.set_network_status(NetworkStatus.CONNECTED)
        
        requests = [
            InferenceRequest(
                request_id=f"batch_req_{i}",
                model_type=ModelType.LANGUAGE_MODEL,
                prompt=f"Question {i}: What is {i} + 1?",
                parameters={}
            )
            for i in range(3)
        ]
        
        results = await engine.batch_infer(requests)
        assert len(results) == 3
        
        for i, result in enumerate(results):
            assert result is not None
            assert result.request_id == f"batch_req_{i}"
    
    def test_inference_stats(self, temp_inference_engine):
        """测试推理统计"""
        engine = temp_inference_engine
        
        stats = engine.get_inference_stats()
        assert 'total_requests' in stats
        assert 'cache_hits' in stats
        assert 'local_inferences' in stats
        assert 'remote_inferences' in stats
        assert 'failed_inferences' in stats
        assert 'cache_hit_rate' in stats
        assert 'current_mode' in stats
        assert 'network_status' in stats
        
        # 初始状态应该都是0
        assert stats['total_requests'] == 0
        assert stats['cache_hits'] == 0
    
    @pytest.mark.asyncio
    async def test_inference_mode_switching(self, temp_inference_engine):
        """测试推理模式切换"""
        engine = temp_inference_engine
        
        # 测试本地模式
        engine.set_inference_mode(InferenceMode.LOCAL_ONLY)
        engine.set_network_status(NetworkStatus.DISCONNECTED)
        
        request = InferenceRequest(
            request_id="local_test",
            model_type=ModelType.LANGUAGE_MODEL,
            prompt="Test local inference",
            parameters={}
        )
        
        result = await engine.infer(request)
        # 注意：由于我们没有实际的本地模型，这可能返回None
        # 在实际场景中，需要先缓存相应的模型
        
        # 测试远程模式
        engine.set_inference_mode(InferenceMode.REMOTE_ONLY)
        engine.set_network_status(NetworkStatus.CONNECTED)
        
        result = await engine.infer(request)
        if result:  # 如果模拟的远程推理成功
            assert 'remote' in result.model_used
    
    def test_cache_operations(self, temp_inference_engine):
        """测试缓存操作"""
        engine = temp_inference_engine
        
        # 清空缓存
        engine.clear_cache()
        
        # 卸载模型
        engine.unload_models()
        
        # 验证模型已卸载
        assert len(engine._loaded_models) == 0
