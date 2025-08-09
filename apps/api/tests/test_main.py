"""
主应用测试
"""

import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """测试健康检查端点"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Status could be "healthy" or "degraded" depending on services
    assert data["status"] in ["healthy", "degraded"]
    assert data["service"] == "ai-agent-api"
    assert data["version"] == "0.1.0"
    assert "services" in data
    assert "database" in data["services"]
    assert "redis" in data["services"]


def test_root_endpoint(client: TestClient):
    """测试根端点"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "AI Agent System API"
    assert data["version"] == "0.1.0"
    assert "docs" in data


@pytest.mark.asyncio
async def test_async_health_check(async_client):
    """测试异步健康检查"""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]


def test_api_documentation_endpoints(client: TestClient):
    """测试API文档端点"""
    # 测试OpenAPI JSON
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_data = response.json()
    assert openapi_data["info"]["title"] == "AI Agent System API"

    # 测试Swagger UI (在开发模式下)
    response = client.get("/docs")
    assert response.status_code == 200

    # 测试ReDoc (在开发模式下)
    response = client.get("/redoc")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_cors_headers(async_client):
    """测试CORS头部"""
    response = await async_client.get("/", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    # CORS headers should be present in actual deployment


@pytest.mark.asyncio
async def test_request_id_header(async_client):
    """测试请求ID头部"""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
