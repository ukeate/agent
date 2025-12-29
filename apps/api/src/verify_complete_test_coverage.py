import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from fastapi.testclient import TestClient
from main import app
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python
"""
完整测试覆盖验证脚本
检查API代码逻辑与测试逻辑的对应关系，并补全测试覆盖
"""

class APICoverageAnalyzer:
    """API测试覆盖分析器"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.api_dir = Path("api/v1")
        self.endpoints_found = {}
        self.test_coverage = {}
        
    def analyze_all_api_modules(self):
        """分析所有API模块，提取端点信息"""
        logger.info("=== API测试覆盖分析 ===")
        logger.info("")
        
        # 分析所有成功加载的API模块
        successful_modules = [
            "security", "mcp", "test", "agents", "agent_interface", 
            "multi_agents", "async_agents", "supervisor", "workflows", 
            "rag", "cache", "events", "streaming", "batch"
        ]
        
        for module_name in successful_modules:
            self._analyze_module(module_name)
        
        self._generate_coverage_report()
        self._run_comprehensive_tests()
    
    def _analyze_module(self, module_name: str):
        """分析单个API模块"""
        module_path = self.api_dir / f"{module_name}.py"
        
        if not module_path.exists():
            logger.warning(f"⚠️  模块文件不存在: {module_path}")
            return
            
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取路由信息
            endpoints = self._extract_endpoints_from_content(content, module_name)
            self.endpoints_found[module_name] = endpoints
            
            logger.info(f"📁 {module_name}.py - 发现 {len(endpoints)} 个端点")
            for endpoint in endpoints:
                logger.info(f"   {endpoint['method']} {endpoint['path']} - {endpoint['function_name']}")
                
        except Exception as e:
            logger.error(f"✗ 分析模块 {module_name} 失败: {str(e)}")
    
    def _extract_endpoints_from_content(self, content: str, module_name: str) -> List[Dict]:
        """从模块内容中提取端点信息"""
        endpoints = []
        
        # 使用正则表达式提取路由装饰器
        route_pattern = r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\'].*?\)\s*(?:async\s+)?def\s+(\w+)'
        
        matches = re.findall(route_pattern, content, re.MULTILINE | re.DOTALL)
        
        for method, path, function_name in matches:
            # 构建完整路径
            full_path = f"/api/v1/{module_name}{path}"
            
            endpoints.append({
                "method": method.upper(),
                "path": full_path,
                "function_name": function_name,
                "module": module_name
            })
        
        return endpoints
    
    def _generate_coverage_report(self):
        """生成测试覆盖报告"""
        logger.info("\n=== 测试覆盖报告 ===")
        
        total_endpoints = 0
        for module_name, endpoints in self.endpoints_found.items():
            total_endpoints += len(endpoints)
            
        logger.info(f"📊 总端点数: {total_endpoints}")
        logger.info(f"📊 模块数: {len(self.endpoints_found)}")
        
        # 按模块统计
        for module_name, endpoints in self.endpoints_found.items():
            logger.info(f"\n🔧 {module_name} 模块:")
            logger.info(f"   端点数: {len(endpoints)}")
            
            # 按HTTP方法分组
            methods = {}
            for endpoint in endpoints:
                method = endpoint["method"]
                if method not in methods:
                    methods[method] = 0
                methods[method] += 1
            
            for method, count in methods.items():
                logger.info(f"   {method}: {count} 个")
    
    def _run_comprehensive_tests(self):
        """运行全面的端点测试"""
        logger.info("\n=== 全面端点测试 ===")
        
        total_tests = 0
        passed_tests = 0
        test_results = []
        
        for module_name, endpoints in self.endpoints_found.items():
            logger.info(f"\n🧪 测试 {module_name} 模块:")
            
            for endpoint in endpoints:
                total_tests += 1
                
                # 执行测试
                success, result = self._test_single_endpoint(endpoint)
                
                if success:
                    passed_tests += 1
                
                test_results.append(result)
                logger.info(f"   {result}")
        
        # 输出最终统计
        logger.info(f"\n=== 测试结果统计 ===")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.error(f"失败测试: {total_tests - passed_tests}")
        logger.info(f"成功率: {(passed_tests/total_tests*100):.1f}%")
        
        # 生成改进建议
        self._generate_improvement_suggestions(test_results)
    
    def _test_single_endpoint(self, endpoint: Dict) -> Tuple[bool, str]:
        """测试单个端点"""
        method = endpoint["method"]
        path = endpoint["path"]
        function_name = endpoint["function_name"]
        
        try:
            # 根据端点类型准备测试数据
            test_data = self._prepare_test_data(endpoint)
            
            # 执行HTTP请求
            if method == "GET":
                response = self.client.get(path, params=test_data.get("params"))
            elif method == "POST":
                response = self.client.post(path, json=test_data.get("json"), params=test_data.get("params"))
            elif method == "PUT":
                response = self.client.put(path, json=test_data.get("json"))
            elif method == "DELETE":
                response = self.client.delete(path)
            else:
                return False, f"✗ 不支持的HTTP方法: {method}"
            
            # 判断测试结果
            success = 200 <= response.status_code < 500
            status_symbol = "✓" if success else "✗"
            
            return success, f"{status_symbol} {method} {path} - {response.status_code} ({function_name})"
            
        except Exception as e:
            return False, f"✗ {method} {path} - 错误: {str(e)[:50]}... ({function_name})"
    
    def _prepare_test_data(self, endpoint: Dict) -> Dict:
        """为端点准备测试数据"""
        path = endpoint["path"]
        method = endpoint["method"]
        function_name = endpoint["function_name"]
        
        test_data = {"params": {}, "json": {}}
        
        # 根据端点类型准备不同的测试数据
        if "security" in path:
            # 安全相关端点通常需要认证，预期401/403
            return test_data
            
        elif "mcp" in path:
            if "tools/call" in path:
                test_data["json"] = {
                    "server_type": "filesystem",
                    "tool_name": "read_file", 
                    "arguments": {"path": "/etc/hosts"}
                }
            elif "tools/filesystem/read" in path:
                test_data["params"] = {"path": "/etc/hosts"}
            elif "tools/filesystem/list" in path:
                test_data["params"] = {"path": "/tmp"}
            elif "tools/database/query" in path:
                test_data["params"] = {"query": "SELECT 1"}
            elif "tools/system/command" in path:
                test_data["params"] = {"command": "echo test"}
                
        elif "agents" in path:
            if "sessions" in path:
                test_data["json"] = {"agent_type": "react"}
            elif "chat" in path:
                test_data["json"] = {"message": "测试消息"}
            elif "task" in path:
                test_data["json"] = {"task_description": "测试任务"}
                
        elif "test" in path:
            # 测试端点通常不需要额外数据
            return test_data
            
        elif "cache" in path:
            if "set" in path:
                test_data["json"] = {"key": "test_key", "value": "test_value"}
                
        elif "events" in path:
            if method == "POST":
                test_data["json"] = {"event_type": "test", "data": {}}
                
        elif "workflows" in path:
            if method == "POST":
                test_data["json"] = {"name": "测试工作流"}
                
        elif "rag" in path:
            if "query" in path:
                test_data["json"] = {"question": "测试问题"}
            elif "documents" in path and method == "POST":
                test_data["json"] = {"content": "测试文档"}
        
        return test_data
    
    def _generate_improvement_suggestions(self, test_results: List[str]):
        """生成改进建议"""
        logger.info(f"\n=== 测试改进建议 ===")
        
        failed_tests = [r for r in test_results if r.startswith("✗")]
        auth_failed = [r for r in failed_tests if "401" in r]
        error_tests = [r for r in failed_tests if "错误:" in r]
        
        logger.info(f"🔐 需要认证的端点: {len(auth_failed)} 个")
        logger.error(f"🐛 逻辑错误的端点: {len(error_tests)} 个")
        
        if auth_failed:
            logger.info(f"\n认证相关端点 (前3个):")
            for test in auth_failed[:3]:
                logger.info(f"   {test}")
        
        if error_tests:
            logger.info(f"\n需要修复的端点 (前3个):")
            for test in error_tests[:3]:
                logger.info(f"   {test}")
                
        logger.info(f"\n✅ 建议:")
        logger.info(f"1. 为需要认证的端点实现测试用户认证机制")
        logger.error(f"2. 修复逻辑错误的端点实现")
        logger.info(f"3. 为复杂端点添加更详细的测试用例")
        logger.info(f"4. 实现端点级别的单元测试")

def main():
    """主函数"""
    analyzer = APICoverageAnalyzer()
    analyzer.analyze_all_api_modules()

if __name__ == "__main__":
    setup_logging()
    main()
