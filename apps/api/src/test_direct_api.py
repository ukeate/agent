from src.core.utils.timezone_utils import utc_now
import os
import uuid
import json
from fastapi.testclient import TestClient
from main import app
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
直接测试main.py中的API端点（绕过lifespan mutex lock问题）
使用TestClient进行内存测试，避免网络和lifespan初始化
"""

os.environ.update({
    'DISABLE_TENSORFLOW': '1',
    'NO_TENSORFLOW': '1',
    'PYTHONDONTWRITEBYTECODE': '1',
    'TESTING': 'true',  # 激活测试模式
})

logger.info("=" * 80)
logger.info("AI Agent System - 直接API测试（绕过lifespan）")
logger.info("=" * 80)

# 导入main.py的应用
logger.info("导入main.py应用实例...")

# 创建测试客户端（不会触发lifespan）
logger.info("创建TestClient...")
client = TestClient(app)

class DirectAPITester:
    """直接API测试器"""
    
    def __init__(self):
        self.results = []
        self.test_session_id = str(uuid.uuid4())
    
    def log_test(self, module: str, endpoint: str, method: str, status: bool, 
                 status_code: int = None, response_data: any = None, error: str = None):
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
            response_str = json.dumps(response_data, ensure_ascii=False)
            logger.info(f"   响应: {response_str[:150]}...")
        if error:
            logger.error(f"   错误: {error}")
    
    def test_endpoint(self, module: str, method: str, endpoint: str, data: dict = None):
        """测试单个端点"""
        try:
            if method.upper() == "GET":
                response = client.get(endpoint)
            elif method.upper() == "POST":
                response = client.post(endpoint, json=data or {})
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            success = response.status_code == 200
            response_data = None
            
            try:
                response_data = response.json()
            except:
                response_data = response.text
                
            self.log_test(module, endpoint, method, success, response.status_code, response_data)
            return success
            
        except Exception as e:
            self.log_test(module, endpoint, method, False, error=str(e))
            return False
    
    def run_basic_tests(self):
        """测试基础端点"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 基础服务")
        logger.info('='*60)
        
        self.test_endpoint("基础服务", "GET", "/")
        self.test_endpoint("基础服务", "GET", "/health")
        # 跳过/docs，因为它会返回HTML
    
    def run_multi_agent_tests(self):
        """测试多智能体系统"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 多智能体系统")
        logger.info('='*60)
        
        self.test_endpoint("多智能体", "GET", "/api/v1/multi-agent/agents")
        
        self.test_endpoint("多智能体", "POST", "/api/v1/multi-agent/session", {
            "name": "test_session",
            "description": "直接测试会话"
        })
        
        self.test_endpoint("多智能体", "POST", "/api/v1/multi-agent/chat", {
            "message": "测试消息",
            "session_id": self.test_session_id
        })
    
    def run_rag_tests(self):
        """测试RAG系统"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: RAG系统")
        logger.info('='*60)
        
        self.test_endpoint("RAG系统", "POST", "/api/v1/rag/query", {
            "question": "什么是RAG？"
        })
        
        self.test_endpoint("RAG系统", "POST", "/api/v1/rag/index", {
            "content": "测试文档内容",
            "title": "测试文档"
        })
    
    def run_workflow_tests(self):
        """测试工作流系统"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 工作流系统")
        logger.info('='*60)
        
        self.test_endpoint("工作流", "GET", "/api/v1/workflows")
        
        self.test_endpoint("工作流", "POST", "/api/v1/workflows/execute", {
            "workflow_id": "1"
        })
    
    def run_mcp_tests(self):
        """测试MCP工具系统"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: MCP工具系统")
        logger.info('='*60)
        
        self.test_endpoint("MCP工具", "GET", "/api/v1/mcp/tools")
        
        self.test_endpoint("MCP工具", "POST", "/api/v1/mcp/execute", {
            "tool_id": "calculator",
            "input": "2+2"
        })
    
    def run_experiment_tests(self):
        """测试实验系统"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 实验系统")  
        logger.info('='*60)
        
        self.test_endpoint("实验系统", "GET", "/api/v1/experiments")
        
        self.test_endpoint("实验系统", "POST", "/api/v1/experiments/create", {
            "name": "测试实验",
            "description": "API直接测试"
        })
    
    def run_monitoring_tests(self):
        """测试监控系统"""
        logger.info(f"\n{'='*60}")
        logger.info("测试模块: 监控系统")
        logger.info('='*60)
        
        self.test_endpoint("监控系统", "GET", "/api/v1/monitoring/metrics")
        self.test_endpoint("监控系统", "GET", "/api/v1/health")
    
    def run_all_tests(self):
        """执行所有测试"""
        logger.info("开始执行所有API直接测试...")
        
        self.run_basic_tests()
        self.run_multi_agent_tests()
        self.run_rag_tests()
        self.run_workflow_tests()
        self.run_mcp_tests()
        self.run_experiment_tests()
        self.run_monitoring_tests()
        
        self.print_summary()
    
    def print_summary(self):
        """输出测试总结"""
        logger.info(f"\n{'='*80}")
        logger.info("直接API测试总结")
        logger.info('='*80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if "通过" in r["status"])
        failed = total - passed
        
        logger.info(f"总测试数: {total}")
        logger.info(f"通过: {passed} ✅")
        logger.error(f"失败: {failed} ❌")
        logger.info(f"成功率: {(passed/total*100):.1f}%")
        
        # 按模块统计
        module_stats = {}
        for result in self.results:
            module = result["module"]
            if module not in module_stats:
                module_stats[module] = {"total": 0, "passed": 0}
            module_stats[module]["total"] += 1
            if "通过" in result["status"]:
                module_stats[module]["passed"] += 1
        
        logger.info(f"\n{'模块':15} {'通过/总数':12} {'成功率':8}")
        logger.info("-" * 40)
        for module, stats in module_stats.items():
            rate = stats["passed"] / stats["total"] * 100
            logger.info(f"{module:15} {stats['passed']}/{stats['total']:10} {rate:6.1f}%")
        
        # 失败详情
        failures = [r for r in self.results if "失败" in r["status"]]
        if failures:
            logger.error(f"\n失败详情:")
            logger.info("-" * 50)
            for failure in failures:
                logger.info(f"[{failure['module']}] {failure['method']} {failure['endpoint']}")
                if failure['error']:
                    logger.info(f"   {failure['error']}")

def main():
    """主函数"""
    tester = DirectAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    setup_logging()
    main()
