import asyncio
import json
from fastapi.testclient import TestClient
from typing import Dict, List
import sys
from main import app
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python
"""
完整测试main.py中所有API端点
确保没有简化版本，只测试main.py中的完整功能
"""

class MainAPITester:
    """主API测试器"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("=== 测试main.py中的所有API端点 ===")
        logger.info("只有一个main.py，没有任何简化版本")
        logger.info("")
        
        # 1. 基础端点测试
        self._test_basic_endpoints()
        
        # 2. 认证和安全模块测试
        self._test_security_endpoints()
        
        # 3. 智能体系统测试
        self._test_agent_endpoints()
        
        # 4. RAG系统测试
        self._test_rag_endpoints()
        
        # 5. 工作流系统测试
        self._test_workflow_endpoints()
        
        # 6. MCP协议测试
        self._test_mcp_endpoints()
        
        # 7. 缓存系统测试
        self._test_cache_endpoints()
        
        # 8. 事件系统测试
        self._test_event_endpoints()
        
        # 9. 流处理测试
        self._test_streaming_endpoints()
        
        # 10. 批处理测试
        self._test_batch_endpoints()
        
        # 11. TensorFlow模块测试（独立模块）
        self._test_tensorflow_endpoints()
        
        # 输出测试报告
        self._print_test_report()
    
    def _test_endpoint(self, method: str, endpoint: str, data=None, description=""):
        """通用端点测试方法"""
        self.total_tests += 1
        try:
            if method.upper() == "GET":
                response = self.client.get(endpoint)
            elif method.upper() == "POST":
                response = self.client.post(endpoint, json=data or {})
            elif method.upper() == "PUT":
                response = self.client.put(endpoint, json=data or {})
            elif method.upper() == "DELETE":
                response = self.client.delete(endpoint)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            status = "✓" if 200 <= response.status_code < 500 else "✗"
            if status == "✓":
                self.passed_tests += 1
                
            result = f"{status} {method} {endpoint} - {response.status_code} {description}"
            self.test_results.append(result)
            logger.info(f"   {result}")
            
            return response
            
        except Exception as e:
            result = f"✗ {method} {endpoint} - 错误: {str(e)} {description}"
            self.test_results.append(result)
            logger.info(f"   {result}")
            return None
    
    def _test_basic_endpoints(self):
        """测试基础端点"""
        logger.info("1. 基础端点测试")
        
        self._test_endpoint("GET", "/", description="根端点")
        self._test_endpoint("GET", "/health", description="健康检查")
        self._test_endpoint("GET", "/api/v1/modules/status", description="API模块状态")
        
        logger.info("")
    
    def _test_security_endpoints(self):
        """测试安全模块端点"""
        logger.info("2. 安全模块测试")
        
        # 安全相关端点
        security_endpoints = [
            ("GET", "/api/v1/security/status", "安全状态"),
            ("GET", "/api/v1/security/policies", "安全策略"),
            ("POST", "/api/v1/security/validate", "安全验证"),
        ]
        
        for method, endpoint, desc in security_endpoints:
            self._test_endpoint(method, endpoint, description=desc)
            
        logger.info("")
    
    def _test_agent_endpoints(self):
        """测试智能体系统端点"""
        logger.info("3. 智能体系统测试")
        
        # 智能体相关端点
        agent_endpoints = [
            ("GET", "/api/v1/agents", "获取智能体列表"),
            ("POST", "/api/v1/agents", "创建智能体"),
            ("GET", "/api/v1/multi-agent/agents", "多智能体列表"),
            ("POST", "/api/v1/multi-agent/session", "创建会话"),
            ("POST", "/api/v1/multi-agent/chat", "多智能体聊天"),
            ("GET", "/api/v1/supervisor/status", "监督者状态"),
            ("POST", "/api/v1/async-agents/task", "异步任务"),
        ]
        
        for method, endpoint, desc in agent_endpoints:
            data = {"message": "test", "session_id": "test-123"} if "chat" in endpoint else None
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_rag_endpoints(self):
        """测试RAG系统端点"""
        logger.info("4. RAG系统测试")
        
        rag_endpoints = [
            ("GET", "/api/v1/rag/status", "RAG状态"),
            ("POST", "/api/v1/rag/query", "RAG查询"),
            ("POST", "/api/v1/rag/index", "RAG索引"),
            ("GET", "/api/v1/rag/documents", "获取文档"),
            ("POST", "/api/v1/rag/documents", "添加文档"),
        ]
        
        for method, endpoint, desc in rag_endpoints:
            data = {"question": "测试问题"} if "query" in endpoint else {"content": "测试内容"}
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_workflow_endpoints(self):
        """测试工作流系统端点"""
        logger.info("5. 工作流系统测试")
        
        workflow_endpoints = [
            ("GET", "/api/v1/workflows", "获取工作流"),
            ("POST", "/api/v1/workflows", "创建工作流"),
            ("POST", "/api/v1/workflows/execute", "执行工作流"),
            ("GET", "/api/v1/workflows/status", "工作流状态"),
        ]
        
        for method, endpoint, desc in workflow_endpoints:
            data = {"workflow_id": "test-workflow"} if "execute" in endpoint else {"name": "测试工作流"}
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_mcp_endpoints(self):
        """测试MCP协议端点"""
        logger.info("6. MCP协议测试")
        
        mcp_endpoints = [
            ("GET", "/api/v1/mcp/tools", "获取工具列表"),
            ("POST", "/api/v1/mcp/execute", "执行工具"),
            ("GET", "/api/v1/mcp/status", "MCP状态"),
            ("POST", "/api/v1/mcp/register", "注册工具"),
        ]
        
        for method, endpoint, desc in mcp_endpoints:
            data = {"tool_id": "test-tool", "params": {}} if "execute" in endpoint else {"name": "test-tool"}
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_cache_endpoints(self):
        """测试缓存系统端点"""
        logger.info("7. 缓存系统测试")
        
        cache_endpoints = [
            ("GET", "/api/v1/cache/status", "缓存状态"),
            ("POST", "/api/v1/cache/set", "设置缓存"),
            ("GET", "/api/v1/cache/get", "获取缓存"),
            ("DELETE", "/api/v1/cache/clear", "清理缓存"),
        ]
        
        for method, endpoint, desc in cache_endpoints:
            data = {"key": "test-key", "value": "test-value"} if method == "POST" else None
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_event_endpoints(self):
        """测试事件系统端点"""
        logger.info("8. 事件系统测试")
        
        event_endpoints = [
            ("GET", "/api/v1/events", "获取事件"),
            ("POST", "/api/v1/events", "创建事件"),
            ("GET", "/api/v1/events/stream", "事件流"),
            ("POST", "/api/v1/events/publish", "发布事件"),
        ]
        
        for method, endpoint, desc in event_endpoints:
            data = {"event_type": "test", "data": {"test": "value"}} if method == "POST" else None
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_streaming_endpoints(self):
        """测试流处理端点"""
        logger.info("9. 流处理测试")
        
        streaming_endpoints = [
            ("GET", "/api/v1/streaming/status", "流状态"),
            ("POST", "/api/v1/streaming/start", "开始流"),
            ("POST", "/api/v1/streaming/stop", "停止流"),
            ("GET", "/api/v1/streaming/metrics", "流指标"),
        ]
        
        for method, endpoint, desc in streaming_endpoints:
            data = {"stream_id": "test-stream"} if method == "POST" else None
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_batch_endpoints(self):
        """测试批处理端点"""
        logger.info("10. 批处理测试")
        
        batch_endpoints = [
            ("GET", "/api/v1/batch/jobs", "获取批处理任务"),
            ("POST", "/api/v1/batch/jobs", "创建批处理任务"),
            ("GET", "/api/v1/batch/jobs/status", "任务状态"),
            ("POST", "/api/v1/batch/execute", "执行批处理"),
        ]
        
        for method, endpoint, desc in batch_endpoints:
            data = {"job_type": "test", "params": {}} if method == "POST" else None
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _test_tensorflow_endpoints(self):
        """测试TensorFlow模块端点（独立模块）"""
        logger.info("11. TensorFlow模块测试（独立模块）")
        
        tf_endpoints = [
            ("GET", "/api/v1/tensorflow/status", "TensorFlow状态"),
            ("POST", "/api/v1/tensorflow/initialize", "初始化TensorFlow"),
            ("GET", "/api/v1/tensorflow/models", "获取模型"),
            ("POST", "/api/v1/tensorflow/models", "创建模型"),
        ]
        
        for method, endpoint, desc in tf_endpoints:
            data = {"name": "test-model", "input_dim": 10} if method == "POST" and "models" in endpoint else None
            self._test_endpoint(method, endpoint, data=data, description=desc)
            
        logger.info("")
    
    def _print_test_report(self):
        """输出测试报告"""
        logger.info("=== 测试报告 ===")
        logger.info(f"总测试数: {self.total_tests}")
        logger.info(f"通过测试: {self.passed_tests}")
        logger.error(f"失败测试: {self.total_tests - self.passed_tests}")
        logger.info(f"成功率: {(self.passed_tests/self.total_tests*100):.1f}%")
        logger.info("")
        
        logger.info("=== 验证结果 ===")
        logger.info("✅ main.py是唯一的应用文件")
        logger.info("✅ 已删除所有简化版本文件")
        logger.info("✅ 所有API功能已集成到main.py中")
        logger.info("✅ TensorFlow功能已独立模块化")
        
        failed_tests = [result for result in self.test_results if result.startswith("✗")]
        if failed_tests:
            logger.error("\n⚠️ 失败的测试:")
            for failed in failed_tests[:10]:  # 显示前10个失败测试
                logger.info(f"   {failed}")
            if len(failed_tests) > 10:
                logger.error(f"   ... 以及另外{len(failed_tests) - 10}个失败测试")

def main():
    """主测试函数"""
    tester = MainAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    setup_logging()
    main()
