import asyncio
import json
from datetime import datetime
import os
from main import app
from fastapi.testclient import TestClient

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
测试所有API端点功能
由于mutex lock问题，直接导入并测试端点函数
"""

os.environ.update({
    'DISABLE_TENSORFLOW': '1',
    'NO_TENSORFLOW': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
})

logger.info("=" * 60)
logger.info("AI Agent System - API端点测试")
logger.info("=" * 60)

# 导入main.py中的app

# 使用FastAPI的测试客户端

client = TestClient(app)

def test_endpoint(method, path, data=None, name=None):
    """测试单个端点"""
    try:
        if method == "GET":
            response = client.get(path)
        elif method == "POST":
            response = client.post(path, json=data or {})
        else:
            response = None
            
        if response:
            logger.info(f"\n✅ {name or path}")
            logger.info(f"   状态码: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)[:200]}...")
            return True
    except Exception as e:
        logger.error(f"\n❌ {name or path}")
        logger.error(f"   错误: {str(e)}")
        return False

# 测试所有端点
logger.info("\n" + "=" * 60)
logger.info("开始测试API端点")
logger.info("=" * 60)

# 基础端点
test_endpoint("GET", "/", name="根路径")
test_endpoint("GET", "/health", name="健康检查")
test_endpoint("GET", "/docs", name="API文档")

# 智能体管理
test_endpoint("GET", "/api/v1/multi-agent/agents", name="获取智能体列表")
test_endpoint("POST", "/api/v1/multi-agent/session", 
              data={"name": "test_session"}, 
              name="创建会话")
test_endpoint("POST", "/api/v1/multi-agent/chat", 
              data={"message": "Hello", "session_id": "test123"}, 
              name="智能体聊天")

# RAG系统
test_endpoint("POST", "/api/v1/rag/query", 
              data={"question": "什么是RAG?"}, 
              name="RAG查询")
test_endpoint("POST", "/api/v1/rag/index", 
              data={"content": "测试文档内容", "title": "测试"}, 
              name="RAG索引")

# 工作流
test_endpoint("GET", "/api/v1/workflows", name="获取工作流列表")
test_endpoint("POST", "/api/v1/workflows/execute", 
              data={"workflow_id": "1"}, 
              name="执行工作流")

# MCP工具
test_endpoint("GET", "/api/v1/mcp/tools", name="获取MCP工具列表")
test_endpoint("POST", "/api/v1/mcp/execute", 
              data={"tool_id": "calculator", "input": "2+2"}, 
              name="执行MCP工具")

# 实验系统
test_endpoint("GET", "/api/v1/experiments", name="获取实验列表")
test_endpoint("POST", "/api/v1/experiments/create", 
              data={"name": "新实验", "description": "测试实验"}, 
              name="创建实验")

# 监控
test_endpoint("GET", "/api/v1/monitoring/metrics", name="获取监控指标")
test_endpoint("GET", "/api/v1/health", name="API健康检查")

logger.info("\n" + "=" * 60)
logger.info("API端点测试完成")
logger.info("=" * 60)
