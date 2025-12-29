import httpx
import asyncio
import json
from typing import Dict, List, Any
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
快速高级API端点测试
测试multi_agents、async_agents、supervisor模块
"""

BASE_URL = "http://localhost:8000/api/v1"

# 高级API端点列表 (48个)
ADVANCED_ENDPOINTS = {
    "multi_agents": [
        ("POST", "/multi-agents/conversations", {"agents": ["agent1"], "topic": "test"}),
        ("GET", "/multi-agents/conversations", {}),
        ("GET", "/multi-agents/conversations/test_123", {}),
        ("POST", "/multi-agents/conversations/test_123/messages", {"sender": "user", "content": "test"}),
        ("GET", "/multi-agents/conversations/test_123/messages", {}),
        ("POST", "/multi-agents/conversations/test_123/agents", {"agent_name": "test_agent"}),
        ("DELETE", "/multi-agents/conversations/test_123/agents/test_agent", {}),
        ("GET", "/multi-agents/health", {}),
        ("GET", "/multi-agents/statistics", {}),
    ],
    "async_agents": [
        ("POST", "/async-agents/agents", {"name": "test_agent", "agent_type": "autogen"}),
        ("GET", "/async-agents/agents", {}),
        ("GET", "/async-agents/agents/test_123", {}),
        ("PUT", "/async-agents/agents/test_123", {"config": {"temp": 0.7}}),
        ("DELETE", "/async-agents/agents/test_123", {}),
        ("POST", "/async-agents/agents/test_123/tasks", {"task_name": "test", "task_data": {}}),
        ("GET", "/async-agents/agents/test_123/tasks", {}),
        ("POST", "/async-agents/workflows", {"name": "test_workflow", "agents": ["agent1"]}),
        ("GET", "/async-agents/workflows", {}),
        ("GET", "/async-agents/workflows/test_123", {}),
        ("POST", "/async-agents/workflows/test_123/execute", {"input_data": {}}),
        ("GET", "/async-agents/health", {}),
        ("GET", "/async-agents/statistics", {}),
        ("GET", "/async-agents/metrics", {}),
    ],
    "supervisor": [
        ("POST", "/supervisor/initialize?supervisor_name=test_sup", {}),
        ("POST", "/supervisor/tasks?supervisor_id=test_sup", {
            "name": "test_task", "description": "test", "task_type": "analysis", "priority": "high"
        }),
        ("GET", "/supervisor/status?supervisor_id=test_sup", {}),
        ("GET", "/supervisor/decisions?supervisor_id=test_sup&limit=10&offset=0", {}),
        ("PUT", "/supervisor/config?supervisor_id=test_sup", {"load_threshold": 0.8}),
        ("GET", "/supervisor/config?supervisor_id=test_sup", {}),
        ("POST", "/supervisor/agents/test_agent?supervisor_id=test_sup", {}),
        ("DELETE", "/supervisor/agents/test_agent?supervisor_id=test_sup", {}),
        ("POST", "/supervisor/tasks/test_123/complete?success=true&quality_score=0.8", {}),
        ("GET", "/supervisor/stats?supervisor_id=test_sup", {}),
        ("GET", "/supervisor/load-statistics?supervisor_id=test_sup", {}),
        ("GET", "/supervisor/metrics?supervisor_id=test_sup", {}),
        ("GET", "/supervisor/tasks?supervisor_id=test_sup&limit=10&offset=0", {}),
        ("GET", "/supervisor/tasks/test_123/details", {}),
        ("POST", "/supervisor/tasks/test_123/execute", {}),
        ("POST", "/supervisor/scheduler/force-execution", {}),
        ("GET", "/supervisor/scheduler/status", {}),
        ("GET", "/supervisor/health", {}),
    ]
}

async def test_endpoint(client, method: str, endpoint: str, data: dict = None):
    """测试单个端点"""
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = await client.get(url, timeout=10)
        elif method == "POST":
            response = await client.post(url, json=data, timeout=10)
        elif method == "PUT":
            response = await client.put(url, json=data, timeout=10)
        elif method == "DELETE":
            response = await client.delete(url, timeout=10)
        else:
            return {"status": "unknown_method", "method": method}
        
        return {
            "status_code": response.status_code,
            "success": response.status_code < 500,
            "response_size": len(response.content)
        }
    except Exception as e:
        return {
            "status_code": 0,
            "success": False,
            "error": str(e)[:100]
        }

async def test_module(module_name: str, endpoints: List):
    """测试一个模块的所有端点"""
    logger.info("测试模块", module=module_name.upper(), total=len(endpoints))
    logger.info("分隔线", line="-" * 50)
    
    results = {"total": 0, "passed": 0, "failed": 0, "details": []}
    
    async with httpx.AsyncClient() as client:
        for method, endpoint, data in endpoints:
            results["total"] += 1
            result = await test_endpoint(client, method, endpoint, data)
            
            endpoint_short = endpoint.split('?')[0]  # 去掉查询参数显示
            if result["success"]:
                results["passed"] += 1
                logger.info(
                    "端点通过",
                    method=method,
                    endpoint=endpoint_short,
                    status_code=result["status_code"],
                )
            else:
                results["failed"] += 1
                error_msg = result.get("error", "HTTP Error")[:50]
                logger.error(
                    "端点失败",
                    method=method,
                    endpoint=endpoint_short,
                    status_code=result["status_code"],
                    error=error_msg,
                )
            
            results["details"].append({
                "method": method,
                "endpoint": endpoint_short,
                "result": result
            })
    
    success_rate = (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0
    logger.info(
        "模块结果",
        module=module_name.upper(),
        passed=results["passed"],
        total=results["total"],
        success_rate=round(success_rate, 1),
    )
    
    return results

async def main():
    """运行所有高级API测试"""
    logger.info("高级API端点快速测试开始")
    logger.info("分隔线", line="=" * 60)
    
    all_results = {}
    total_tests = 0
    total_passed = 0
    
    # 测试每个模块
    for module_name, endpoints in ADVANCED_ENDPOINTS.items():
        results = await test_module(module_name, endpoints)
        all_results[module_name] = results
        total_tests += results["total"]
        total_passed += results["passed"]
    
    # 汇总统计
    logger.info("高级API测试综合统计")
    logger.info("分隔线", line="=" * 60)
    
    for module_name, results in all_results.items():
        success_rate = (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0
        logger.info(
            "模块统计",
            module=module_name.upper(),
            passed=results["passed"],
            total=results["total"],
            success_rate=round(success_rate, 1),
        )
    
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    logger.info(
        "总体结果",
        passed=total_passed,
        total=total_tests,
        success_rate=round(overall_success_rate, 1),
    )
    
    # 详细错误分析
    logger.info("新发现的API端点数量", total=total_tests)
    logger.info("补充测试覆盖", base_count=158, new_total=158 + total_tests)
    
    return all_results

if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
