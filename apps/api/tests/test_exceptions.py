"""
异常处理测试
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_not_found_endpoint(async_client: AsyncClient):
    """测试404错误处理"""
    response = await async_client.get("/api/v1/nonexistent")
    assert response.status_code == 404

    data = response.json()
    # FastAPI默认返回{"detail": "Not Found"}格式
    # 我们的自定义异常处理器只处理我们定义的异常类型
    assert "detail" in data or "error" in data


@pytest.mark.asyncio
async def test_method_not_allowed(async_client: AsyncClient):
    """测试405错误处理"""
    response = await async_client.post("/health")  # GET-only endpoint
    assert response.status_code == 405

    data = response.json()
    # FastAPI默认返回{"detail": "Method Not Allowed"}格式
    assert "detail" in data or "error" in data


@pytest.mark.asyncio
async def test_custom_exception_format(async_client: AsyncClient):
    """测试自定义异常的标准化格式（通过触发我们的异常处理器）"""
    # 我们需要一个会触发自定义异常的端点
    # 这里我们测试应用启动时的异常处理逻辑
    # 在实际场景中，数据库/Redis连接失败会触发我们的异常处理器
    pass  # 这个测试需要有实际触发自定义异常的端点


@pytest.mark.asyncio
async def test_request_id_in_response_header(async_client: AsyncClient):
    """测试响应中包含请求ID头部"""
    response = await async_client.get("/")
    assert response.status_code == 200

    # 请求ID应该在响应头中
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    assert isinstance(request_id, str)
    assert len(request_id) > 0
