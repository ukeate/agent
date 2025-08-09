"""
异步操作测试
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_async_database_operations(async_client: AsyncClient):
    """测试异步数据库操作端点"""
    response = await async_client.get("/api/v1/test/async-db")

    # 在测试模式下，应该返回模拟数据
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "模拟模式" in data["message"]
    assert "data" in data
    assert "current_time" in data["data"]
    assert "database_version" in data["data"]


@pytest.mark.asyncio
async def test_async_redis_operations(async_client: AsyncClient):
    """测试异步Redis操作端点"""
    response = await async_client.get("/api/v1/test/async-redis")

    # 在测试模式下，应该返回模拟数据
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "模拟模式" in data["message"]
    assert "data" in data
    assert data["data"]["set_result"] is True
    assert "Mock" in data["data"]["get_result"]


@pytest.mark.asyncio
async def test_concurrent_requests(async_client: AsyncClient):
    """测试并发请求处理"""
    response = await async_client.get("/api/v1/test/concurrent")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["task_count"] == 10
    assert isinstance(data["data"]["execution_time"], float)
    assert isinstance(data["data"]["results"], list)
    assert len(data["data"]["results"]) == 10


@pytest.mark.asyncio
async def test_mixed_async_operations(async_client: AsyncClient):
    """测试混合异步操作"""
    response = await async_client.get("/api/v1/test/mixed-async")

    # 在测试模式下，应该返回模拟数据
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "模拟模式" in data["message"]
    assert "data" in data
    assert "database_timestamp" in data["data"]
    assert data["data"]["redis_operation"] is True
    assert data["data"]["compute_completed"] is True
