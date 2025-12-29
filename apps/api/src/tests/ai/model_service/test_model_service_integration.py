"""模型服务平台集成测试"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import torch
import torch.nn as nn
import numpy as np
from ....ai.model_service.registry import (
    ModelRegistry,
    ModelRegistrationRequest, 
    ModelFormat,
    PyTorchLoader,
    ONNXLoader,
    HuggingFaceLoader
)
from ....ai.model_service.inference import (
    InferenceEngine,
    InferenceRequest,
    InferenceStatus
)
from ....ai.model_service.deployment import (
    DeploymentManager,
    DeploymentType,
    DeploymentConfig
)
from ....ai.model_service.online_learning import (
    OnlineLearningEngine,
    FeedbackData,
    ABTestConfig
)
from ....ai.model_service.monitoring import MonitoringSystem
from src.core.logging import setup_logging

logger = get_logger(__name__)

class TestModelServiceIntegration:
    """模型服务平台集成测试"""

    @pytest.fixture
    def temp_storage(self):
        """临时存储目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def simple_pytorch_model(self):
        """简单的PyTorch模型"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        return SimpleModel()
    
    @pytest.fixture
    def model_registry(self, temp_storage):
        """模型注册表"""
        return ModelRegistry(temp_storage)
    
    @pytest.fixture
    def inference_engine(self, model_registry):
        """推理引擎"""
        return InferenceEngine(model_registry, device="cpu")
    
    @pytest.fixture
    def deployment_manager(self, model_registry):
        """部署管理器"""
        return DeploymentManager(model_registry)
    
    @pytest.fixture
    def online_learning_engine(self, model_registry, deployment_manager, temp_storage):
        """在线学习引擎"""
        learning_storage = Path(temp_storage) / "online_learning"
        return OnlineLearningEngine(model_registry, deployment_manager, str(learning_storage))
    
    @pytest.fixture
    def monitoring_system(self, temp_storage):
        """监控系统"""
        monitoring_storage = Path(temp_storage) / "monitoring"
        return MonitoringSystem(str(monitoring_storage))
    
    @pytest.mark.asyncio
    async def test_end_to_end_model_lifecycle(
        self, 
        model_registry, 
        inference_engine, 
        simple_pytorch_model,
        temp_storage
    ):
        """测试完整的模型生命周期"""
        
        # 1. 注册模型
        model_path = Path(temp_storage) / "test_model.pt"
        torch.save(simple_pytorch_model, model_path)
        
        registration_request = ModelRegistrationRequest(
            name="test-model",
            version="1.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch",
            description="测试模型",
            tags=["test", "integration"]
        )
        
        model_id = await model_registry.register_model(registration_request, str(model_path))
        assert model_id is not None
        
        # 2. 验证模型注册
        model_info = model_registry.get_model("test-model", "1.0")
        assert model_info is not None
        assert model_info.name == "test-model"
        assert model_info.version == "1.0"
        
        # 3. 加载模型到推理引擎
        success = inference_engine.load_model("test-model", "1.0")
        assert success
        
        # 4. 执行推理
        inference_request = InferenceRequest(
            request_id="test-request",
            model_name="test-model",
            model_version="1.0",
            inputs={"data": [[1.0] * 10]},
            batch_size=1
        )
        
        result = await inference_engine.inference(inference_request)
        assert result.status == InferenceStatus.COMPLETED
        assert result.outputs is not None
        
        # 5. 验证模型列表
        models = model_registry.list_models()
        assert len(models) == 1
        assert models[0].name == "test-model"
        
        # 6. 清理
        success = model_registry.delete_model("test-model", "1.0")
        assert success
    
    @pytest.mark.asyncio 
    async def test_online_learning_workflow(
        self,
        model_registry,
        online_learning_engine,
        simple_pytorch_model,
        temp_storage
    ):
        """测试在线学习工作流"""
        
        # 1. 准备和注册模型
        model_path = Path(temp_storage) / "learning_model.pt"
        torch.save(simple_pytorch_model, model_path)
        
        registration_request = ModelRegistrationRequest(
            name="learning-model",
            version="1.0", 
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        await model_registry.register_model(registration_request, str(model_path))
        
        # 2. 启动在线学习会话
        config = {
            "learning_rate": 1e-3,
            "batch_size": 16,
            "buffer_size": 100,
            "update_frequency": 10
        }
        
        session_id = await online_learning_engine.start_online_learning(
            "learning-model", "1.0", config
        )
        assert session_id is not None
        
        # 3. 提交反馈数据
        for i in range(15):  # 提交足够的反馈以触发更新
            feedback_data = {
                "inputs": {"data": [float(j) for j in range(10)]},
                "expected_output": 1.0,
                "actual_output": 0.8,
                "feedback_type": "regression",
                "quality_score": 1.0
            }
            
            await online_learning_engine.collect_feedback(
                session_id,
                f"pred-{i}",
                feedback_data
            )
        
        # 4. 触发模型更新
        update_result = await online_learning_engine.update_model(session_id)
        assert "status" in update_result
        
        # 5. 检查学习统计
        stats = online_learning_engine.get_learning_stats(session_id)
        assert stats is not None
        assert stats["feedback_count"] == 15
        assert stats["session_id"] == session_id
        
        # 6. 停止学习会话
        success = await online_learning_engine.stop_learning_session(session_id)
        assert success
    
    @pytest.mark.asyncio
    async def test_ab_testing_workflow(self, online_learning_engine):
        """测试A/B测试工作流"""
        
        # 1. 创建A/B测试
        test_config = ABTestConfig(
            test_id="test-ab-001",
            name="模型对比测试",
            description="比较两个模型的性能",
            control_model="model-a",
            treatment_models=["model-b"],
            traffic_split={"model-a": 0.5, "model-b": 0.5},
            success_metrics=["accuracy"],
            minimum_sample_size=100
        )
        
        test_id = online_learning_engine.ab_test_engine.create_ab_test(test_config)
        assert test_id == "test-ab-001"
        
        # 2. 模拟用户分配和数据收集
        user_assignments = {}
        for i in range(200):  # 足够的样本量
            user_id = f"user-{i}"
            assigned_model = online_learning_engine.assign_model_for_user(test_id, user_id)
            assert assigned_model in ["model-a", "model-b"]
            user_assignments[user_id] = assigned_model
            
            # 模拟测试指标
            accuracy = 0.85 if assigned_model == "model-a" else 0.88
            accuracy += np.random.normal(0, 0.05)  # 添加噪声
            
            online_learning_engine.record_ab_test_metrics(
                test_id, user_id, {"accuracy": accuracy}
            )
        
        # 3. 分析测试结果
        results = online_learning_engine.get_ab_test_results(test_id)
        assert results["status"] == "completed"
        assert "models" in results
        assert len(results["models"]) == 2
        
        # 4. 验证流量分配相对均匀
        model_a_users = sum(1 for model in user_assignments.values() if model == "model-a")
        model_b_users = sum(1 for model in user_assignments.values() if model == "model-b")
        
        # 允许10%的偏差
        assert abs(model_a_users - model_b_users) < 20
    
    @pytest.mark.asyncio
    async def test_monitoring_system_integration(
        self,
        monitoring_system,
        inference_engine,
        model_registry,
        simple_pytorch_model,
        temp_storage
    ):
        """测试监控系统集成"""
        
        # 1. 启动监控
        await monitoring_system.start_monitoring(collection_interval=1)
        
        # 2. 准备测试模型
        model_path = Path(temp_storage) / "monitor_model.pt"
        torch.save(simple_pytorch_model, model_path)
        
        registration_request = ModelRegistrationRequest(
            name="monitor-model",
            version="1.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        await model_registry.register_model(registration_request, str(model_path))
        inference_engine.load_model("monitor-model", "1.0")
        
        # 3. 执行一些推理请求以生成指标
        for i in range(10):
            start_time = time.time()
            
            inference_request = InferenceRequest(
                request_id=f"monitor-request-{i}",
                model_name="monitor-model", 
                model_version="1.0",
                inputs={"data": [[1.0] * 10]}
            )
            
            result = await inference_engine.inference(inference_request)
            latency_ms = (time.time() - start_time) * 1000
            
            # 记录性能指标
            monitoring_system.record_model_inference(
                "monitor-model:1.0",
                latency_ms,
                result.status == InferenceStatus.COMPLETED
            )
        
        # 4. 等待指标收集
        await asyncio.sleep(2)
        
        # 5. 验证监控数据
        overview = monitoring_system.get_system_overview()
        assert "model_metrics" in overview
        
        model_metrics = monitoring_system.model_monitor.get_model_metrics("monitor-model:1.0")
        assert model_metrics is not None
        assert model_metrics["request_count"] >= 10
        assert model_metrics["success_count"] >= 10
        
        # 6. 测试告警系统
        # 模拟高延迟以触发告警
        for _ in range(5):
            monitoring_system.record_model_inference("monitor-model:1.0", 2000, True)
        
        # 评估告警规则
        monitoring_system.alert_manager.evaluate_rules(monitoring_system.metrics_collector)
        
        # 7. 获取优化建议
        recommendations = monitoring_system.get_resource_recommendations()
        assert isinstance(recommendations, list)
        
        # 8. 停止监控
        await monitoring_system.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_deployment_workflow_simulation(self, deployment_manager, model_registry, simple_pytorch_model, temp_storage):
        """测试部署工作流（模拟）"""
        
        # 1. 注册模型
        model_path = Path(temp_storage) / "deploy_model.pt"
        torch.save(simple_pytorch_model, model_path)
        
        registration_request = ModelRegistrationRequest(
            name="deploy-model",
            version="1.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        await model_registry.register_model(registration_request, str(model_path))
        
        # 2. 模拟部署（由于需要Docker/K8s环境，这里只测试配置生成）
        config = {
            "replicas": 2,
            "cpu_request": "200m",
            "memory_request": "512Mi", 
            "port": 8080
        }
        
        # 3. 测试Docker配置生成
        docker_manager = deployment_manager.docker_manager
        model_info = {"model_path": str(model_path), "format": "pytorch"}
        deployment_config = DeploymentConfig(
            deployment_type=DeploymentType.DOCKER,
            model_name="deploy-model",
            model_version="1.0",
            **config
        )
        
        dockerfile = docker_manager._generate_dockerfile(model_info, deployment_config)
        assert "FROM python:" in dockerfile
        assert "EXPOSE 8080" in dockerfile
        
        # 4. 测试Kubernetes配置生成
        k8s_manager = deployment_manager.k8s_manager
        k8s_config = k8s_manager._generate_k8s_config("deploy-model", "test-deploy", config)
        
        assert k8s_config["kind"] == "Deployment"
        assert k8s_config["spec"]["replicas"] == 2
        
        # 5. 获取部署指标
        metrics = deployment_manager.get_deployment_metrics()
        assert "total_deployments" in metrics
    
    @pytest.mark.asyncio
    async def test_performance_stress_test(
        self,
        inference_engine,
        model_registry,
        simple_pytorch_model,
        temp_storage
    ):
        """性能压力测试"""
        
        # 1. 准备模型
        model_path = Path(temp_storage) / "stress_model.pt"
        torch.save(simple_pytorch_model, model_path)
        
        registration_request = ModelRegistrationRequest(
            name="stress-model",
            version="1.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        await model_registry.register_model(registration_request, str(model_path))
        inference_engine.load_model("stress-model", "1.0")
        
        # 2. 并发推理测试
        concurrent_requests = 50
        requests = []
        
        for i in range(concurrent_requests):
            request = InferenceRequest(
                request_id=f"stress-{i}",
                model_name="stress-model",
                model_version="1.0",
                inputs={"data": [[float(j) for j in range(10)]]},
                batch_size=1
            )
            requests.append(inference_engine.inference(request))
        
        # 3. 执行并发推理
        start_time = time.time()
        results = await asyncio.gather(*requests, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 4. 验证结果
        successful_results = [r for r in results if not isinstance(r, Exception) and r.status == InferenceStatus.COMPLETED]
        failed_results = [r for r in results if isinstance(r, Exception) or (hasattr(r, 'status') and r.status != InferenceStatus.COMPLETED)]
        
        success_rate = len(successful_results) / len(results)
        qps = len(results) / total_time
        
        logger.info(f"\n压力测试结果:")
        logger.info(f"  总请求数: {len(results)}")
        logger.info(f"  成功数: {len(successful_results)}")
        logger.error(f"  失败数: {len(failed_results)}")
        logger.info(f"  成功率: {success_rate:.2%}")
        logger.info(f"  QPS: {qps:.2f}")
        logger.info(f"  总耗时: {total_time:.2f}s")
        
        # 5. 断言性能要求
        assert success_rate >= 0.95  # 95%成功率
        assert qps >= 10  # 至少10 QPS
        
        # 6. 检查延迟分布
        latencies = [r.processing_time_ms for r in successful_results if r.processing_time_ms]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            logger.info(f"  平均延迟: {avg_latency:.2f}ms")
            logger.info(f"  P95延迟: {p95_latency:.2f}ms")
            
            assert avg_latency < 1000  # 平均延迟小于1秒
            assert p95_latency < 2000  # P95延迟小于2秒
    
    def test_model_format_compatibility(self, model_registry, temp_storage):
        """测试不同模型格式的兼容性"""
        
        # 1. 测试PyTorch模型
        simple_model = nn.Linear(10, 1)
        pytorch_path = Path(temp_storage) / "pytorch_model.pt"
        torch.save(simple_model, pytorch_path)
        
        pytorch_metadata = PyTorchLoader.extract_metadata(str(pytorch_path))
        # framework字段已从extract_metadata返回值中移除
        assert "parameter_count" in pytorch_metadata
        
        # 2. 测试ONNX模型（模拟）
        try:
            import onnx
            from onnx import helper, TensorProto
            
            # 创建简单的ONNX模型
            X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 10])
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
            
            # 定义权重
            W = helper.make_tensor('W', TensorProto.FLOAT, [10, 1], [0.1] * 10)
            
            # 定义节点
            node_def = helper.make_node('MatMul', ['X', 'W'], ['Y'])
            
            # 创建图
            graph_def = helper.make_graph([node_def], 'test_model', [X], [Y], [W])
            model_def = helper.make_model(graph_def, producer_name='test')
            
            onnx_path = Path(temp_storage) / "onnx_model.onnx"
            onnx.save(model_def, onnx_path)
            
            onnx_metadata = ONNXLoader.extract_metadata(str(onnx_path))
            # framework字段已从extract_metadata返回值中移除
            assert "input_schema" in onnx_metadata
            
        except ImportError:
            logger.info("ONNX未安装，跳过ONNX兼容性测试")
        
        # 3. 测试HuggingFace模型（模拟）
        try:
            # 这里只测试元数据提取逻辑，不实际下载模型
            with patch('transformers.AutoConfig.from_pretrained') as mock_config:
                mock_config.return_value = MagicMock(
                    model_type="bert",
                    vocab_size=30522,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12
                )
                
                hf_metadata = HuggingFaceLoader.extract_metadata("bert-base-uncased")
                # framework字段已从extract_metadata返回值中移除
                assert hf_metadata["model_type"] == "bert"
                assert hf_metadata["vocab_size"] == 30522
                
        except ImportError:
            logger.info("Transformers未安装，跳过HuggingFace兼容性测试")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self,
        inference_engine,
        model_registry,
        online_learning_engine,
        temp_storage
    ):
        """测试错误处理和恢复机制"""
        
        # 1. 测试不存在的模型推理
        inference_request = InferenceRequest(
            request_id="error-test-1",
            model_name="nonexistent-model",
            model_version="1.0",
            inputs={"data": [[1.0] * 10]}
        )
        
        result = await inference_engine.inference(inference_request)
        assert result.status == InferenceStatus.FAILED
        assert result.error is not None
        
        # 2. 测试错误的输入数据
        # 先注册一个正常模型
        simple_model = nn.Linear(10, 1)
        model_path = Path(temp_storage) / "error_test_model.pt"
        torch.save(simple_model, model_path)
        
        from ....ai.model_service.registry import ModelRegistrationRequest, ModelFormat
        registration_request = ModelRegistrationRequest(
            name="error-test-model",
            version="1.0",
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        await model_registry.register_model(registration_request, str(model_path))
        inference_engine.load_model("error-test-model", "1.0")
        
        # 测试错误输入
        bad_request = InferenceRequest(
            request_id="error-test-2", 
            model_name="error-test-model",
            model_version="1.0",
            inputs={"bad_key": "invalid_data"}  # 错误的输入格式
        )
        
        result = await inference_engine.inference(bad_request)
        assert result.status == InferenceStatus.FAILED
        
        # 3. 测试在线学习的错误处理
        with pytest.raises(ValueError):
            await online_learning_engine.start_online_learning(
                "nonexistent-model", "1.0", {}
            )
        
        # 4. 测试模型注册的重复处理
        # 尝试注册同名同版本的模型
        duplicate_request = ModelRegistrationRequest(
            name="error-test-model",
            version="1.0",  # 相同的版本
            format=ModelFormat.PYTORCH,
            framework="pytorch"
        )
        
        # 第二次注册应该成功（覆盖）或处理重复
        try:
            model_id2 = await model_registry.register_model(duplicate_request, str(model_path))
            assert model_id2 is not None
        except Exception as e:
            # 如果不允许重复，应该有明确的错误信息
            assert "已存在" in str(e) or "duplicate" in str(e).lower()
    
    @pytest.mark.asyncio 
    async def test_resource_cleanup_and_management(
        self,
        inference_engine,
        model_registry,
        monitoring_system,
        online_learning_engine,
        temp_storage
    ):
        """测试资源清理和管理"""
        
        # 1. 创建多个模型
        models_created = []
        for i in range(3):
            model = nn.Linear(10, 1)
            model_path = Path(temp_storage) / f"cleanup_model_{i}.pt"
            torch.save(model, model_path)
            
            registration_request = ModelRegistrationRequest(
                name=f"cleanup-model-{i}",
                version="1.0",
                format=ModelFormat.PYTORCH,
                framework="pytorch"
            )
            
            model_id = await model_registry.register_model(registration_request, str(model_path))
            models_created.append((f"cleanup-model-{i}", "1.0"))
            
            # 加载到推理引擎
            inference_engine.load_model(f"cleanup-model-{i}", "1.0")
        
        # 2. 验证模型已加载
        loaded_models = inference_engine.get_loaded_models()
        assert len(loaded_models) == 3
        
        # 3. 测试批量卸载
        for model_name, version in models_created:
            success = inference_engine.unload_model(model_name, version)
            assert success
        
        # 4. 验证模型已卸载
        loaded_models = inference_engine.get_loaded_models()
        assert len(loaded_models) == 0
        
        # 5. 测试缓存清理
        inference_engine.clear_cache()
        
        # 6. 创建学习会话并清理过期会话
        session_config = {"learning_rate": 1e-4}
        session_id = await online_learning_engine.start_online_learning(
            "cleanup-model-0", "1.0", session_config
        )
        
        # 模拟过期清理（设置很短的过期时间）
        await online_learning_engine.cleanup_expired_sessions(max_age_days=0)
        
        # 7. 测试监控数据清理
        monitoring_system.metrics_collector.clear_old_metrics(retention_hours=0)
        
        # 8. 删除所有测试模型
        for model_name, version in models_created:
            success = model_registry.delete_model(model_name, version)
            assert success
        
        # 9. 验证注册表为空
        remaining_models = model_registry.list_models()
        test_models = [m for m in remaining_models if "cleanup-model" in m.name]
        assert len(test_models) == 0

if __name__ == "__main__":
    setup_logging()
    pytest.main([__file__, "-v"])
from src.core.logging import get_logger
