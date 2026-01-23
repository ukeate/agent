"""
健康检查端点状态码测试
"""

from unittest.mock import AsyncMock, patch

from fastapi import status


def test_liveness_dead_returns_503(client):
    with patch(
        "src.api.v1.health.check_liveness", new_callable=AsyncMock
    ) as mock_check:
        mock_check.return_value = False
        response = client.get("/api/v1/health/live")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["status"] == "dead"


def test_readiness_not_ready_returns_503(client):
    with patch(
        "src.api.v1.health.get_health_status", new_callable=AsyncMock
    ) as mock_health:
        mock_health.return_value = {
            "status": "unhealthy",
            "components": {"database": "unhealthy"},
            "failed_components": ["database"],
        }
        response = client.get("/api/v1/health/ready")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["status"] == "not_ready"
