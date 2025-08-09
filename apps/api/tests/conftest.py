"""
pytest配置和fixtures
"""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.main import create_app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app():
    """创建测试应用实例"""
    import os

    os.environ["TESTING"] = "true"
    return create_app()


@pytest.fixture
def client(app) -> TestClient:
    """创建测试客户端"""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """创建异步测试客户端"""
    from httpx import ASGITransport

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
