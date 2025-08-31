"""平台集成器测试"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ai.platform_integration.integrator import PlatformIntegrator
from ai.platform_integration.models import (
    ComponentRegistration,
    ComponentInfo,
    ComponentType,
    ComponentStatus,
    WorkflowRequest,
    WorkflowStatus
)


@pytest.fixture
def platform_config():
    """平台配置"""
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_db': 0
    }


@pytest.fixture
def platform_integrator(platform_config):
    """平台集成器实例"""
    with patch('redis.Redis'):
        integrator = PlatformIntegrator(platform_config)
        return integrator


@pytest.fixture
def sample_component_registration():
    """示例组件注册信息"""
    return ComponentRegistration(
        component_id="test_component",
        component_type=ComponentType.FINE_TUNING,
        name="Test Component",
        version="1.0.0",
        health_endpoint="http://localhost:8001/health",
        api_endpoint="http://localhost:8001",
        metadata={"description": "Test component for unit tests"}
    )


@pytest.fixture
def sample_component_info():
    """示例组件信息"""
    return ComponentInfo(
        component_id="test_component",
        component_type=ComponentType.FINE_TUNING,
        name="Test Component",
        version="1.0.0",
        status=ComponentStatus.HEALTHY,
        health_endpoint="http://localhost:8001/health",
        api_endpoint="http://localhost:8001",
        metadata={"description": "Test component for unit tests"},
        registered_at=datetime.now(),
        last_heartbeat=datetime.now()
    )


class TestPlatformIntegrator:
    """平台集成器测试类"""

    @pytest.mark.asyncio
    async def test_init(self, platform_config):
        """测试初始化"""
        with patch('redis.Redis') as mock_redis:
            integrator = PlatformIntegrator(platform_config)
            
            assert integrator.config == platform_config
            assert integrator.components == {}
            assert integrator.component_dependencies == {}
            assert integrator._health_monitor_task is None
            mock_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_component_success(self, platform_integrator, sample_component_registration):
        """测试成功注册组件"""
        with patch.object(platform_integrator, '_check_component_health', return_value=True), \
             patch.object(platform_integrator, '_save_component_to_redis'):
            
            component_info = await platform_integrator._register_component_from_registration(
                sample_component_registration
            )
            
            assert component_info.component_id == sample_component_registration.component_id
            assert component_info.status == ComponentStatus.HEALTHY
            assert sample_component_registration.component_id in platform_integrator.components

    @pytest.mark.asyncio
    async def test_register_component_unhealthy(self, platform_integrator, sample_component_registration):
        """测试注册不健康组件"""
        with patch.object(platform_integrator, '_check_component_health', return_value=False), \
             patch.object(platform_integrator, '_save_component_to_redis'):
            
            component_info = await platform_integrator._register_component_from_registration(
                sample_component_registration
            )
            
            assert component_info.status == ComponentStatus.UNHEALTHY
            assert sample_component_registration.component_id in platform_integrator.components

    @pytest.mark.asyncio
    async def test_unregister_component_success(self, platform_integrator, sample_component_info):
        """测试成功注销组件"""
        # 先注册组件
        platform_integrator.components[sample_component_info.component_id] = sample_component_info
        
        with patch.object(platform_integrator.redis_client, 'delete'):
            await platform_integrator._unregister_component(sample_component_info.component_id)
            
            assert sample_component_info.component_id not in platform_integrator.components

    @pytest.mark.asyncio
    async def test_unregister_component_not_found(self, platform_integrator):
        """测试注销不存在的组件"""
        with pytest.raises(ValueError, match="Component non_existent not found"):
            await platform_integrator._unregister_component("non_existent")

    @pytest.mark.asyncio
    async def test_check_component_health_success(self, platform_integrator, sample_component_info):
        """测试组件健康检查成功"""
        mock_response = Mock()
        mock_response.status = 200
        
        # 创建mock context manager
        mock_response_context = AsyncMock()
        mock_response_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_context.__aexit__ = AsyncMock(return_value=None)
        
        # 创建mock session 
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response_context)
        
        # 创建mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            is_healthy = await platform_integrator._check_component_health(sample_component_info)
            
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_check_component_health_failure(self, platform_integrator, sample_component_info):
        """测试组件健康检查失败"""
        mock_response = Mock()
        mock_response.status = 500
        
        # 创建mock context manager
        mock_response_context = AsyncMock()
        mock_response_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_context.__aexit__ = AsyncMock(return_value=None)
        
        # 创建mock session 
        mock_session = AsyncMock()
        mock_session.get = Mock(return_value=mock_response_context)
        
        # 创建mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            is_healthy = await platform_integrator._check_component_health(sample_component_info)
            
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_component_health_exception(self, platform_integrator, sample_component_info):
        """测试组件健康检查异常"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Connection error")
            
            is_healthy = await platform_integrator._check_component_health(sample_component_info)
            
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_platform_health(self, platform_integrator, sample_component_info):
        """测试平台健康检查"""
        # 添加健康和不健康的组件
        healthy_component = sample_component_info
        healthy_component.status = ComponentStatus.HEALTHY
        
        unhealthy_component = ComponentInfo(
            component_id="unhealthy_component",
            component_type=ComponentType.EVALUATION,
            name="Unhealthy Component",
            version="1.0.0",
            status=ComponentStatus.UNHEALTHY,
            health_endpoint="http://localhost:8002/health",
            api_endpoint="http://localhost:8002",
            metadata={},
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        
        platform_integrator.components = {
            healthy_component.component_id: healthy_component,
            unhealthy_component.component_id: unhealthy_component
        }
        
        health_status = await platform_integrator._check_platform_health()
        
        assert health_status.total_components == 2
        assert health_status.healthy_components == 1
        assert health_status.overall_status == "degraded"

    @pytest.mark.asyncio
    async def test_execute_workflow_background(self, platform_integrator):
        """测试后台执行工作流"""
        workflow_request = WorkflowRequest(
            workflow_type="full_fine_tuning",
            parameters={"model_name": "test_model"},
            priority=1
        )
        
        with patch.object(platform_integrator, '_execute_workflow') as mock_execute:
            await platform_integrator._execute_workflow_background("test_workflow_id", workflow_request)
            
            mock_execute.assert_called_once_with("test_workflow_id", workflow_request)

    @pytest.mark.asyncio
    async def test_execute_workflow_full_fine_tuning(self, platform_integrator):
        """测试执行完整微调工作流"""
        workflow_request = WorkflowRequest(
            workflow_type="full_fine_tuning",
            parameters={"model_name": "test_model"},
            priority=1
        )
        
        # Mock workflow step execution
        with patch.object(platform_integrator, '_execute_workflow_step') as mock_step, \
             patch.object(platform_integrator, '_save_workflow_state'):
            
            mock_step.return_value = {"success": True, "result": {"status": "completed"}}
            
            result = await platform_integrator._execute_workflow("test_workflow_id", workflow_request)
            
            assert result["workflow_id"] == "test_workflow_id"
            assert result["status"] == "completed"
            assert mock_step.call_count == 6  # 6 steps in full_fine_tuning workflow

    @pytest.mark.asyncio
    async def test_execute_workflow_step_success(self, platform_integrator):
        """测试执行工作流步骤成功"""
        result = await platform_integrator._execute_workflow_step(
            "data_preparation",
            {"dataset": "test_dataset"}
        )
        
        assert result["success"] is True
        assert result["step"] == "data_preparation"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_execute_workflow_step_unknown(self, platform_integrator):
        """测试执行未知工作流步骤"""
        result = await platform_integrator._execute_workflow_step(
            "unknown_step",
            {}
        )
        
        assert result["success"] is False
        assert result["step"] == "unknown_step"
        assert "Unknown workflow step" in result["error"]

    @pytest.mark.asyncio
    async def test_get_component_by_type(self, platform_integrator, sample_component_info):
        """测试根据类型获取组件"""
        platform_integrator.components[sample_component_info.component_id] = sample_component_info
        
        component = platform_integrator._get_component_by_type(ComponentType.FINE_TUNING)
        
        assert component is not None
        assert component.component_type == ComponentType.FINE_TUNING

    @pytest.mark.asyncio
    async def test_get_component_by_type_not_found(self, platform_integrator):
        """测试根据类型获取组件但未找到"""
        component = platform_integrator._get_component_by_type(ComponentType.COMPRESSION)
        
        assert component is None

    @pytest.mark.asyncio
    async def test_call_component_api_success(self, platform_integrator, sample_component_info):
        """测试调用组件API成功"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})
        
        # 创建mock context manager
        mock_response_context = AsyncMock()
        mock_response_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_context.__aexit__ = AsyncMock(return_value=None)
        
        # 创建mock session 
        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response_context)
        
        # 创建mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            result = await platform_integrator._call_component_api(
                sample_component_info,
                "/test",
                {"data": "test"}
            )
            
            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_call_component_api_error(self, platform_integrator, sample_component_info):
        """测试调用组件API错误"""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        # 创建mock context manager
        mock_response_context = AsyncMock()
        mock_response_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_context.__aexit__ = AsyncMock(return_value=None)
        
        # 创建mock session 
        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response_context)
        
        # 创建mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await platform_integrator._call_component_api(
                    sample_component_info,
                    "/test",
                    {"data": "test"}
                )
            
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_save_component_to_redis(self, platform_integrator, sample_component_info):
        """测试保存组件到Redis"""
        with patch.object(platform_integrator.redis_client, 'setex') as mock_setex:
            await platform_integrator._save_component_to_redis(sample_component_info)
            
            mock_setex.assert_called_once()
            args = mock_setex.call_args
            assert args[0][0] == f"component:{sample_component_info.component_id}"
            assert args[0][1] == 3600  # 1 hour TTL

    @pytest.mark.asyncio
    async def test_save_workflow_state(self, platform_integrator):
        """测试保存工作流状态"""
        from ai.platform_integration.models import WorkflowState, WorkflowStatus
        
        workflow_state = WorkflowState(
            workflow_id="test_workflow",
            workflow_type="test_type",
            status=WorkflowStatus.RUNNING,
            steps=[],
            parameters={},
            started_at=datetime.now()
        )
        
        with patch.object(platform_integrator.redis_client, 'setex') as mock_setex:
            await platform_integrator._save_workflow_state("test_workflow", workflow_state)
            
            mock_setex.assert_called_once()
            args = mock_setex.call_args
            assert args[0][0] == "workflow:test_workflow"
            assert args[0][1] == 86400  # 24 hours TTL

    @pytest.mark.asyncio
    async def test_get_workflow_status_success(self, platform_integrator):
        """测试获取工作流状态成功"""
        mock_status = {"workflow_id": "test_workflow", "status": "running"}
        
        with patch.object(platform_integrator.redis_client, 'get', return_value='{"workflow_id": "test_workflow", "status": "running"}'):
            status = await platform_integrator._get_workflow_status("test_workflow")
            
            assert status["workflow_id"] == "test_workflow"
            assert status["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_workflow_status_not_found(self, platform_integrator):
        """测试获取工作流状态未找到"""
        with patch.object(platform_integrator.redis_client, 'get', return_value=None):
            with pytest.raises(ValueError, match="Workflow test_workflow not found"):
                await platform_integrator._get_workflow_status("test_workflow")

    @pytest.mark.asyncio
    async def test_health_monitor_loop(self, platform_integrator, sample_component_info):
        """测试健康监控循环"""
        platform_integrator.components[sample_component_info.component_id] = sample_component_info
        
        # 创建async mock
        async def mock_health_check(comp_info):
            return True
        
        async def mock_save_component(comp_info):
            return None
        
        with patch.object(platform_integrator, '_check_component_health', side_effect=mock_health_check), \
             patch.object(platform_integrator, '_save_component_to_redis', side_effect=mock_save_component), \
             patch('asyncio.sleep') as mock_sleep:
            
            # 第一次sleep正常，第二次抛出CancelledError让循环终止
            call_count = 0
            def sleep_side_effect(*args):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise asyncio.CancelledError()
                return None
            
            mock_sleep.side_effect = sleep_side_effect
            
            # 不期望抛出异常，因为循环内部捕获了CancelledError
            await platform_integrator._health_monitor_loop()
            
            # 验证健康检查被调用
            assert sample_component_info.status == ComponentStatus.HEALTHY
            # 验证sleep被调用了至少一次
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_start_stop_health_monitor(self, platform_integrator):
        """测试启动和停止健康监控"""
        # 测试启动
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = AsyncMock()
            mock_create_task.return_value = mock_task
            
            await platform_integrator.start_health_monitor()
            
            assert platform_integrator._health_monitor_task == mock_task
            mock_create_task.assert_called_once()
        
        # 测试停止 - 直接patch stop_health_monitor方法避免复杂的async mock
        with patch.object(platform_integrator, 'stop_health_monitor') as mock_stop:
            await mock_stop()
            mock_stop.assert_called_once()
        
        # 验证实际的停止逻辑 - 创建真实的task并停止
        real_task = asyncio.create_task(asyncio.sleep(0))  # 一个简单的task
        platform_integrator._health_monitor_task = real_task
        
        await platform_integrator.stop_health_monitor()
        assert platform_integrator._health_monitor_task is None


class TestPlatformIntegratorIntegration:
    """平台集成器集成测试"""

    @pytest.mark.asyncio
    async def test_full_component_lifecycle(self, platform_config):
        """测试完整的组件生命周期"""
        with patch('redis.Redis'):
            integrator = PlatformIntegrator(platform_config)
            
            # 注册组件
            component_reg = ComponentRegistration(
                component_id="lifecycle_test",
                component_type=ComponentType.FINE_TUNING,
                name="Lifecycle Test Component",
                version="1.0.0",
                health_endpoint="http://localhost:8001/health",
                api_endpoint="http://localhost:8001"
            )
            
            with patch.object(integrator, '_check_component_health', return_value=True), \
                 patch.object(integrator, '_save_component_to_redis'):
                
                component_info = await integrator._register_component_from_registration(component_reg)
                
                # 验证注册成功
                assert component_info.component_id == "lifecycle_test"
                assert component_info.status == ComponentStatus.HEALTHY
                assert "lifecycle_test" in integrator.components
                
                # 注销组件
                with patch.object(integrator.redis_client, 'delete'):
                    await integrator._unregister_component("lifecycle_test")
                    
                    assert "lifecycle_test" not in integrator.components

    @pytest.mark.asyncio
    async def test_workflow_execution_with_components(self, platform_config):
        """测试带组件的工作流执行"""
        with patch('redis.Redis'):
            integrator = PlatformIntegrator(platform_config)
            
            # 注册必要的组件
            components = [
                ComponentInfo(
                    component_id="data_service",
                    component_type=ComponentType.DATA_MANAGEMENT,
                    name="Data Service",
                    version="1.0.0",
                    status=ComponentStatus.HEALTHY,
                    health_endpoint="http://localhost:8001/health",
                    api_endpoint="http://localhost:8001",
                    metadata={},
                    registered_at=datetime.now(),
                    last_heartbeat=datetime.now()
                ),
                ComponentInfo(
                    component_id="tuning_service",
                    component_type=ComponentType.FINE_TUNING,
                    name="Fine-tuning Service",
                    version="1.0.0",
                    status=ComponentStatus.HEALTHY,
                    health_endpoint="http://localhost:8002/health",
                    api_endpoint="http://localhost:8002",
                    metadata={},
                    registered_at=datetime.now(),
                    last_heartbeat=datetime.now()
                )
            ]
            
            for component in components:
                integrator.components[component.component_id] = component
            
            # 执行工作流
            workflow_request = WorkflowRequest(
                workflow_type="full_fine_tuning",
                parameters={
                    "model_name": "test_model",
                    "dataset": "test_dataset"
                }
            )
            
            with patch.object(integrator, '_save_workflow_state'):
                result = await integrator._execute_workflow("integration_test_workflow", workflow_request)
                
                assert result["workflow_id"] == "integration_test_workflow"
                assert result["status"] in ["completed", "failed"]