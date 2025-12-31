from src.core.utils.timezone_utils import utc_now
import json
import time
import uuid
from typing import Dict, List, Any
import os
import requests
from fastapi.testclient import TestClient
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
AI Agent System - 完整API测试套件
按模块组织的系统化测试，覆盖所有API端点
"""

os.environ.update({
    'DISABLE_TENSORFLOW': '1',
    'NO_TENSORFLOW': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
})

logger.info("=" * 80)
logger.info("AI Agent System - 完整API测试套件")
logger.info("=" * 80)

# 测试配置
BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = str(uuid.uuid4())

class APITestSuite:
    """API测试套件"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
        
    def log_test(self, module: str, endpoint: str, method: str, status: bool, 
                 status_code: int = None, response_data: Any = None, error: str = None):
        """记录测试结果"""
        result = {
            "module": module,
            "endpoint": endpoint,
            "method": method,
            "status": "✅ 通过" if status else "❌ 失败",
            "status_code": status_code,
            "timestamp": utc_now().isoformat(),
            "error": error
        }
        self.results.append(result)
        
        # 实时输出
        logger.info(f"\n[{result['module']}] {result['method']} {endpoint}")
        logger.info(f"   状态: {result['status']}")
        if status_code:
            logger.info(f"   状态码: {status_code}")
        if response_data and status:
            logger.info(f"   响应: {json.dumps(response_data, ensure_ascii=False)[:150]}...")
        if error:
            logger.error(f"   错误: {error}")
    
    def test_request(self, module: str, method: str, endpoint: str, data: Dict = None) -> bool:
        """执行单个API测试"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data or {})
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            # 检查响应
            success = response.status_code == 200
            response_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else None
            
            self.log_test(module, endpoint, method, success, response.status_code, response_data)
            return success
            
        except Exception as e:
            self.log_test(module, endpoint, method, False, error=str(e))
            return False
    
    def test_basic_module(self):
        """测试基础模块"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 基础服务")
        logger.info('='*60)
        
        tests = [
            ("GET", "/", "根路径"),
            ("GET", "/health", "健康检查"),
            ("GET", "/docs", "API文档"),
        ]
        
        for method, endpoint, name in tests:
            self.test_request("基础服务", method, endpoint)
    
    def test_multi_agent_module(self):
        """测试多智能体系统模块"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 多智能体系统")
        logger.info('='*60)
        
        # 获取智能体列表
        self.test_request("多智能体", "GET", "/api/v1/multi-agent/agents")
        
        # 创建会话
        session_data = {"name": "test_session", "description": "API测试会话"}
        self.test_request("多智能体", "POST", "/api/v1/multi-agent/session", session_data)
        
        # 智能体对话
        chat_data = {
            "message": "你好，这是一个测试消息",
            "session_id": TEST_SESSION_ID,
            "agent": "assistant"
        }
        self.test_request("多智能体", "POST", "/api/v1/multi-agent/chat", chat_data)
    
    def test_rag_module(self):
        """测试RAG系统模块"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: RAG系统")
        logger.info('='*60)
        
        # RAG查询
        query_data = {
            "question": "什么是RAG系统？",
            "context": "测试上下文",
            "max_results": 5
        }
        self.test_request("RAG系统", "POST", "/api/v1/rag/query", query_data)
        
        # 文档索引
        index_data = {
            "content": "这是一个测试文档，包含RAG系统的相关信息。",
            "title": "测试文档",
            "metadata": {"category": "test", "source": "api_test"}
        }
        self.test_request("RAG系统", "POST", "/api/v1/rag/index", index_data)
    
    def test_workflow_module(self):
        """测试工作流系统模块"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 工作流系统")
        logger.info('='*60)
        
        # 获取工作流列表
        self.test_request("工作流", "GET", "/api/v1/workflows")
        
        # 执行工作流
        execute_data = {
            "workflow_id": "1",
            "parameters": {"input": "测试输入", "mode": "test"},
            "priority": "normal"
        }
        self.test_request("工作流", "POST", "/api/v1/workflows/execute", execute_data)
    
    def test_mcp_module(self):
        """测试MCP工具系统模块"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: MCP工具系统")
        logger.info('='*60)
        
        # 获取工具列表
        self.test_request("MCP工具", "GET", "/api/v1/mcp/tools")
        
        # 执行工具
        execute_data = {
            "tool_id": "calculator",
            "input": "2 + 2 * 3",
            "parameters": {"precision": 2}
        }
        self.test_request("MCP工具", "POST", "/api/v1/mcp/execute", execute_data)
    
    def test_experiment_module(self):
        """测试实验系统模块"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 实验系统")
        logger.info('='*60)
        
        # 获取实验列表
        self.test_request("实验系统", "GET", "/api/v1/experiments")
        
        # 创建实验
        create_data = {
            "name": "API测试实验",
            "description": "通过API创建的测试实验",
            "variants": ["control", "variant_a"],
            "traffic_split": {"control": 0.5, "variant_a": 0.5}
        }
        self.test_request("实验系统", "POST", "/api/v1/experiments/create", create_data)
    
    def test_monitoring_module(self):
        """测试监控系统模块"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 监控系统")
        logger.info('='*60)
        
        # 获取系统指标
        self.test_request("监控系统", "GET", "/api/v1/monitoring/metrics")
        
        # API健康检查
        self.test_request("监控系统", "GET", "/api/v1/health")
    
    def run_all_tests(self):
        """运行所有测试"""
        start_time = time.time()
        
        logger.info("开始执行完整API测试套件...")
        
        # 依次执行各模块测试
        self.test_basic_module()
        self.test_multi_agent_module()
        self.test_rag_module()
        self.test_workflow_module()
        self.test_mcp_module()
        self.test_experiment_module()
        self.test_monitoring_module()
        
        # 输出测试总结
        self.print_summary(time.time() - start_time)
    
    def print_summary(self, duration: float):
        """输出测试总结"""
        logger.info(f"\n{'='*80}")
        logger.info("测试总结报告")
        logger.info('='*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if "通过" in r["status"])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests} ✅")
        logger.error(f"失败测试: {failed_tests} ❌")
        logger.info(f"成功率: {(passed_tests/total_tests*100):.1f}%")
        logger.info(f"执行时间: {duration:.2f}秒")
        
        # 按模块分组统计
        module_stats = {}
        for result in self.results:
            module = result["module"]
            if module not in module_stats:
                module_stats[module] = {"total": 0, "passed": 0}
            module_stats[module]["total"] += 1
            if "通过" in result["status"]:
                module_stats[module]["passed"] += 1
        
        logger.info(f"\n{'模块统计':20} {'通过/总数':15} {'成功率':10}")
        logger.info("-" * 50)
        for module, stats in module_stats.items():
            success_rate = stats["passed"] / stats["total"] * 100
            logger.info(f"{module:20} {stats['passed']}/{stats['total']:13} {success_rate:8.1f}%")
        
        # 失败测试详情
        if failed_tests > 0:
            logger.error(f"\n失败测试详情:")
            logger.info("-" * 50)
            for result in self.results:
                if "失败" in result["status"]:
                    logger.info(f"[{result['module']}] {result['method']} {result['endpoint']}")
                    logger.error(f"   错误: {result['error']}")

def main():
    """主函数"""
    # 检查服务是否可用
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"❌ 服务不可用，状态码: {response.status_code}")
            logger.info("请确保API服务器已启动在 http://localhost:8000")
            return
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 无法连接到服务器: {e}")
        logger.info("请确保API服务器已启动在 http://localhost:8000")
        return
    
    logger.info("✅ 服务器连接正常，开始测试...")
    
    # 执行完整测试套件
    test_suite = APITestSuite(BASE_URL)
    test_suite.run_all_tests()

if __name__ == "__main__":
    setup_logging()
    main()
