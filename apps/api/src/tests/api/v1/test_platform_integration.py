"""平台集成API端点测试"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from api.v1.platform_integration import router
from ai.platform_integration.models import (
    ComponentRegistration,
    ComponentInfo,
    ComponentType,
    ComponentStatus,
    WorkflowRequest,
    PlatformHealthStatus,
    PerformanceMetrics
)


@pytest.fixture
def test_app():
    """测试应用"""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    
    return app


@pytest.fixture
def client(test_app):
    """测试客户端"""
    return TestClient(test_app)


@pytest.fixture
def sample_component_registration():
    """示例组件注册数据"""
    return {
        "component_id": "test_component",
        "component_type": "fine_tuning",
        "name": "Test Component",
        "version": "1.0.0",
        "health_endpoint": "http://localhost:8001/health",
        "api_endpoint": "http://localhost:8001",
        "metadata": {"description": "Test component"}
    }


@pytest.fixture
def sample_workflow_request():
    """示例工作流请求数据"""
    return {
        "workflow_type": "full_fine_tuning",
        "parameters": {
            "model_name": "test_model",
            "data_config": {
                "dataset": "test_dataset",
                "batch_size": 32
            }
        },
        "priority": 1
    }


class TestComponentManagementAPI:
    """组件管理API测试"""

    def test_register_component_success(self, client, sample_component_registration):
        """测试成功注册组件"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_component_info = ComponentInfo(
                component_id="test_component",
                component_type=ComponentType.FINE_TUNING,
                name="Test Component",
                version="1.0.0",
                status=ComponentStatus.HEALTHY,
                health_endpoint="http://localhost:8001/health",
                api_endpoint="http://localhost:8001",
                metadata={"description": "Test component"},
                registered_at=datetime.now(),
                last_heartbeat=datetime.now()
            )
            
            mock_integrator._register_component_from_registration = AsyncMock(return_value=mock_component_info)
            mock_get_integrator.return_value = mock_integrator
            
            response = client.post("/platform/components/register", json=sample_component_registration)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["component_id"] == "test_component"
            assert data["component_status"] == "healthy"

    def test_register_component_error(self, client, sample_component_registration):
        """测试注册组件失败"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_integrator._register_component_from_registration = AsyncMock(side_effect=Exception("Registration failed"))
            mock_get_integrator.return_value = mock_integrator
            
            response = client.post("/platform/components/register", json=sample_component_registration)
            
            assert response.status_code == 500
            assert "Registration failed" in response.json()["detail"]

    def test_unregister_component_success(self, client):
        """测试成功注销组件"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_integrator._unregister_component = AsyncMock()
            mock_get_integrator.return_value = mock_integrator
            
            response = client.delete("/platform/components/test_component")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["component_id"] == "test_component"

    def test_unregister_component_not_found(self, client):
        """测试注销不存在的组件"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_integrator._unregister_component = AsyncMock(side_effect=ValueError("Component not found"))
            mock_get_integrator.return_value = mock_integrator
            
            response = client.delete("/platform/components/nonexistent")
            
            assert response.status_code == 404
            assert "Component not found" in response.json()["detail"]

    def test_list_components(self, client):
        """测试列出组件"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_components = {
                "comp1": Mock(),
                "comp2": Mock()
            }
            mock_components["comp1"].component_type.value = "fine_tuning"
            mock_components["comp1"].status.value = "healthy"
            mock_components["comp1"].to_dict.return_value = {"id": "comp1", "type": "fine_tuning"}
            
            mock_components["comp2"].component_type.value = "evaluation"
            mock_components["comp2"].status.value = "healthy"
            mock_components["comp2"].to_dict.return_value = {"id": "comp2", "type": "evaluation"}
            
            mock_integrator.components = mock_components
            mock_get_integrator.return_value = mock_integrator
            
            response = client.get("/platform/components")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["total_components"] == 2
            assert "comp1" in data["components"]
            assert "comp2" in data["components"]

    def test_list_components_with_filters(self, client):
        """测试带过滤条件列出组件"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_components = {
                "comp1": Mock(),
                "comp2": Mock()
            }
            mock_components["comp1"].component_type.value = "fine_tuning"
            mock_components["comp1"].status.value = "healthy"
            mock_components["comp1"].to_dict.return_value = {"id": "comp1", "type": "fine_tuning"}
            
            mock_components["comp2"].component_type.value = "evaluation"
            mock_components["comp2"].status.value = "unhealthy"
            mock_components["comp2"].to_dict.return_value = {"id": "comp2", "type": "evaluation"}
            
            mock_integrator.components = mock_components
            mock_get_integrator.return_value = mock_integrator
            
            # 按组件类型过滤
            response = client.get("/platform/components?component_type=fine_tuning")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_components"] == 1
            assert "comp1" in data["components"]
            assert "comp2" not in data["components"]

    def test_get_component_details_success(self, client):
        """测试成功获取组件详情"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_component = Mock()
            mock_component.to_dict.return_value = {"id": "test_comp", "status": "healthy"}
            mock_integrator.components = {"test_comp": mock_component}
            mock_integrator._check_component_health = AsyncMock(return_value=True)
            mock_get_integrator.return_value = mock_integrator
            
            response = client.get("/platform/components/test_comp")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "component" in data
            assert data["component"]["current_health"] == "healthy"

    def test_get_component_details_not_found(self, client):
        """测试获取不存在组件的详情"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_integrator.components = {}
            mock_get_integrator.return_value = mock_integrator
            
            response = client.get("/platform/components/nonexistent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]


class TestWorkflowManagementAPI:
    """工作流管理API测试"""

    def test_run_workflow_success(self, client, sample_workflow_request):
        """测试成功运行工作流"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_get_integrator.return_value = mock_integrator
            
            response = client.post("/platform/workflows/run", json=sample_workflow_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "workflow_id" in data
            assert data["workflow_type"] == "full_fine_tuning"
            assert "estimated_duration" in data

    def test_run_workflow_invalid_type(self, client):
        """测试运行无效类型的工作流"""
        invalid_request = {
            "workflow_type": "invalid_type",
            "parameters": {}
        }
        
        with patch('api.v1.platform_integration.get_platform_integrator'):
            response = client.post("/platform/workflows/run", json=invalid_request)
            
            assert response.status_code == 400
            assert "Invalid workflow type" in response.json()["detail"]

    def test_get_workflow_status_success(self, client):
        """测试成功获取工作流状态"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_status = {
                "workflow_id": "test_workflow",
                "status": "running",
                "progress": 0.5
            }
            mock_integrator._get_workflow_status = AsyncMock(return_value=mock_status)
            mock_get_integrator.return_value = mock_integrator
            
            response = client.get("/platform/workflows/test_workflow/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["workflow"]["workflow_id"] == "test_workflow"

    def test_get_workflow_status_not_found(self, client):
        """测试获取不存在工作流的状态"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_integrator._get_workflow_status = AsyncMock(side_effect=ValueError("Workflow not found"))
            mock_get_integrator.return_value = mock_integrator
            
            response = client.get("/platform/workflows/nonexistent/status")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    def test_cancel_workflow(self, client):
        """测试取消工作流"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_get_integrator.return_value = mock_integrator
            
            response = client.post("/platform/workflows/test_workflow/cancel")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["workflow_id"] == "test_workflow"

    def test_list_workflows(self, client):
        """测试列出工作流"""
        response = client.get("/platform/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "workflows" in data
        assert "pagination" in data

    def test_list_workflows_with_filters(self, client):
        """测试带过滤条件列出工作流"""
        response = client.get("/platform/workflows?status=running&limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["pagination"]["limit"] == 5


class TestHealthAndMonitoringAPI:
    """健康检查和监控API测试"""

    def test_platform_health(self, client):
        """测试平台健康检查"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_health = PlatformHealthStatus(
                overall_status="healthy",
                healthy_components=3,
                total_components=3,
                components={},
                timestamp=datetime.now()
            )
            mock_integrator._check_platform_health = AsyncMock(return_value=mock_health)
            mock_get_integrator.return_value = mock_integrator
            
            response = client.get("/platform/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["overall_status"] == "healthy"
            assert data["healthy_components"] == 3
            assert data["total_components"] == 3

    def test_get_platform_metrics(self, client):
        """测试获取平台指标"""
        with patch('api.v1.platform_integration.get_monitoring_system') as mock_get_monitoring:
            mock_monitoring = Mock()
            mock_monitoring.get_metrics.return_value = b"# HELP test_metric Test metric\ntest_metric 1.0"
            mock_get_monitoring.return_value = mock_monitoring
            
            response = client.get("/platform/metrics")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_get_monitoring_report(self, client):
        """测试获取监控报告"""
        with patch('api.v1.platform_integration.get_monitoring_system') as mock_get_monitoring:
            mock_monitoring = Mock()
            mock_report = {
                "report_generated_at": datetime.now().isoformat(),
                "health_score": 95.5,
                "recommendations": ["System is operating normally"]
            }
            mock_monitoring.generate_monitoring_report = AsyncMock(return_value=mock_report)
            mock_get_monitoring.return_value = mock_monitoring
            
            response = client.get("/platform/monitoring/report")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["report"]["health_score"] == 95.5


class TestPerformanceOptimizationAPI:
    """性能优化API测试"""

    def test_run_performance_optimization(self, client):
        """测试运行性能优化"""
        with patch('api.v1.platform_integration.get_performance_optimizer') as mock_get_optimizer:
            mock_optimizer = Mock()
            mock_results = {
                "optimizations": [
                    {"optimization": "database", "results": {"status": "optimized"}}
                ],
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            mock_optimizer.optimize_system_performance = AsyncMock(return_value=mock_results)
            mock_get_optimizer.return_value = mock_optimizer
            
            response = client.post("/platform/optimization/run")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "optimization_results" in data

    def test_get_performance_metrics(self, client):
        """测试获取性能指标"""
        with patch('api.v1.platform_integration.get_performance_optimizer') as mock_get_optimizer:
            mock_optimizer = Mock()
            mock_metrics = PerformanceMetrics(
                cpu_percent=45.0,
                memory_percent=65.0,
                disk_usage={"read_bytes": 1024**3, "write_bytes": 512 * 1024**2},
                network_usage={"bytes_sent": 100 * 1024**2, "bytes_recv": 200 * 1024**2},
                bottlenecks=[],
                timestamp=datetime.now()
            )
            mock_optimizer.collect_metrics = AsyncMock(return_value=mock_metrics)
            mock_get_optimizer.return_value = mock_optimizer
            
            response = client.get("/platform/optimization/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["metrics"]["cpu_percent"] == 45.0
            assert data["metrics"]["memory_percent"] == 65.0

    def test_apply_optimization_profile_success(self, client):
        """测试成功应用优化配置文件"""
        with patch('api.v1.platform_integration.get_performance_optimizer') as mock_get_optimizer:
            mock_optimizer = Mock()
            mock_result = {
                "status": "applied",
                "profile": "high_performance",
                "configuration": {"cache_ttl": 3600}
            }
            mock_optimizer.apply_optimization_profile = AsyncMock(return_value=mock_result)
            mock_get_optimizer.return_value = mock_optimizer
            
            response = client.post("/platform/optimization/profile/high_performance")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["result"]["profile"] == "high_performance"

    def test_apply_optimization_profile_error(self, client):
        """测试应用无效优化配置文件"""
        with patch('api.v1.platform_integration.get_performance_optimizer') as mock_get_optimizer:
            mock_optimizer = Mock()
            mock_result = {
                "status": "error",
                "message": "Unknown profile"
            }
            mock_optimizer.apply_optimization_profile = AsyncMock(return_value=mock_result)
            mock_get_optimizer.return_value = mock_optimizer
            
            response = client.post("/platform/optimization/profile/invalid")
            
            assert response.status_code == 400
            assert "Unknown profile" in response.json()["detail"]

    def test_get_performance_report(self, client):
        """测试获取性能报告"""
        with patch('api.v1.platform_integration.get_performance_optimizer') as mock_get_optimizer:
            mock_optimizer = Mock()
            mock_report = {
                "timestamp": datetime.now().isoformat(),
                "performance_score": 87.5,
                "bottlenecks": [],
                "recommendations": ["System performance is good"]
            }
            mock_optimizer.generate_performance_report = AsyncMock(return_value=mock_report)
            mock_get_optimizer.return_value = mock_optimizer
            
            response = client.get("/platform/optimization/report")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["report"]["performance_score"] == 87.5


class TestDocumentationAPI:
    """文档生成API测试"""

    def test_generate_documentation(self, client):
        """测试生成文档"""
        with patch('api.v1.platform_integration.get_documentation_generator') as mock_get_generator:
            mock_generator = Mock()
            mock_get_generator.return_value = mock_generator
            
            response = client.post("/platform/documentation/generate")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "estimated_duration" in data

    def test_get_documentation_status(self, client):
        """测试获取文档生成状态"""
        response = client.get("/platform/documentation/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "generation_status" in data
        assert "available_documents" in data

    def test_generate_training_materials(self, client):
        """测试生成培训材料"""
        with patch('api.v1.platform_integration.get_documentation_generator') as mock_get_generator:
            mock_generator = Mock()
            mock_get_generator.return_value = mock_generator
            
            response = client.post("/platform/documentation/training-materials")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "estimated_duration" in data


class TestSystemConfigAPI:
    """系统配置API测试"""

    def test_get_platform_config(self, client):
        """测试获取平台配置"""
        response = client.get("/platform/config")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "config" in data
        assert "version" in data["config"]
        assert "features" in data["config"]
        assert "supported_workflow_types" in data["config"]

    def test_get_platform_stats(self, client):
        """测试获取平台统计信息"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_components = {
                "comp1": Mock(),
                "comp2": Mock()
            }
            # 设置组件属性
            for comp_id, comp in mock_components.items():
                comp.component_type.value = "fine_tuning"
                comp.status.value = "healthy"
            
            mock_integrator.components = mock_components
            mock_get_integrator.return_value = mock_integrator
            
            response = client.get("/platform/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "stats" in data
            assert "components" in data["stats"]
            assert "workflows" in data["stats"]
            assert data["stats"]["components"]["total"] == 2


class TestAPIErrorHandling:
    """API错误处理测试"""

    def test_component_registration_validation_error(self, client):
        """测试组件注册验证错误"""
        invalid_data = {
            "component_id": "",  # 空ID
            "component_type": "invalid_type",  # 无效类型
            "name": "Test"
        }
        
        response = client.post("/platform/components/register", json=invalid_data)
        
        # FastAPI会返回422验证错误
        assert response.status_code == 422

    def test_internal_server_error_handling(self, client, sample_component_registration):
        """测试内部服务器错误处理"""
        with patch('api.v1.platform_integration.get_platform_integrator') as mock_get_integrator:
            mock_integrator = Mock()
            mock_integrator._register_component_from_registration = AsyncMock(
                side_effect=Exception("Unexpected error")
            )
            mock_get_integrator.return_value = mock_integrator
            
            response = client.post("/platform/components/register", json=sample_component_registration)
            
            assert response.status_code == 500
            assert "Unexpected error" in response.json()["detail"]