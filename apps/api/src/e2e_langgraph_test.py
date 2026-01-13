#!/usr/bin/env python3
"""
LangGraph 0.6.5 新特性 E2E 测试脚本
全面测试Context API、durability控制、Node Caching和Pre/Post Hooks
"""
import asyncio
import json
import time
import aiohttp
import sys
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    duration_ms: float
    response: Dict[str, Any]
    error: str = ""

class LangGraphE2ETestSuite:
    """LangGraph E2E 测试套件"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results: List[TestResult] = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🚀 开始LangGraph 0.6.5新特性E2E测试...")
        
        # 测试用例列表
        test_cases = [
            ("测试Context API演示", self.test_context_api_demo),
            ("测试Context API可选参数", self.test_api_compatibility),
            ("测试Durability控制", self.test_durability_control),
            ("测试Node Caching功能", self.test_node_caching),
            ("测试Pre/Post Hooks", self.test_hooks_functionality),
            ("测试钩子状态管理", self.test_hooks_management),
            ("测试缓存统计", self.test_cache_statistics),
            ("测试完整功能演示", self.test_complete_demo),
            ("测试错误处理", self.test_error_handling),
            ("测试并发请求", self.test_concurrent_requests),
        ]
        
        async with aiohttp.ClientSession() as session:
            for test_name, test_func in test_cases:
                print(f"\n📋 执行测试: {test_name}")
                try:
                    await test_func(session)
                    print(f"✅ {test_name} - 通过")
                except Exception as e:
                    print(f"❌ {test_name} - 失败: {str(e)}")
                    self.test_results.append(TestResult(
                        name=test_name,
                        passed=False,
                        duration_ms=0.0,
                        response={},
                        error=str(e)
                    ))
        
        return self.generate_report()
    
    async def test_context_api_demo(self, session: aiohttp.ClientSession):
        """测试Context API演示"""
        start_time = time.time()
        
        # 测试新Context API
        payload = {
            "message": "测试新Context API功能",
            "user_id": "test_user_e2e",
            "session_id": "550e8400-e29b-41d4-a716-446655440001"
        }
        
        async with session.post(
            f"{self.base_url}/api/v1/langgraph/context-api/demo",
            json=payload
        ) as response:
            duration_ms = (time.time() - start_time) * 1000
            data = await response.json()
            
            # 验证响应
            assert response.status == 200, f"状态码错误: {response.status}"
            assert data["success"] is True, "API调用失败"
            assert "新Context API" in data["metadata"]["api_type"], "API类型不正确"
            assert len(data["result"]["messages"]) >= 2, "消息数量不正确"
            
            self.test_results.append(TestResult(
                name="Context API演示",
                passed=True,
                duration_ms=duration_ms,
                response=data
            ))
    
    async def test_api_compatibility(self, session: aiohttp.ClientSession):
        """测试Context API可选参数"""
        start_time = time.time()
        
        # 测试可选的conversation_id参数
        payload = {
            "message": "测试可选参数",
            "user_id": "option_user",
            "conversation_id": "550e8400-e29b-41d4-a716-446655440099",
            "session_id": "550e8400-e29b-41d4-a716-446655440002"
        }
        
        async with session.post(
            f"{self.base_url}/api/v1/langgraph/context-api/demo",
            json=payload
        ) as response:
            duration_ms = (time.time() - start_time) * 1000
            data = await response.json()
            
            assert response.status == 200, f"状态码错误: {response.status}"
            assert data["success"] is True, "可选参数测试失败"
            assert "新Context API" in data["metadata"]["api_type"], "API类型不正确"
            
            self.test_results.append(TestResult(
                name="Context API可选参数测试",
                passed=True,
                duration_ms=duration_ms,
                response=data
            ))
    
    async def test_durability_control(self, session: aiohttp.ClientSession):
        """测试Durability控制"""
        durability_modes = ["exit", "async", "sync"]
        
        for mode in durability_modes:
            start_time = time.time()
            
            payload = {
                "message": f"测试{mode}模式",
                "durability_mode": mode
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/langgraph/durability/demo",
                json=payload
            ) as response:
                duration_ms = (time.time() - start_time) * 1000
                data = await response.json()
                
                assert response.status == 200, f"Durability {mode} 状态码错误"
                assert data["success"] is True, f"Durability {mode} 失败"
                assert data["metadata"]["durability_mode"] == mode, f"Durability模式不匹配: {mode}"
                
                self.test_results.append(TestResult(
                    name=f"Durability控制-{mode}模式",
                    passed=True,
                    duration_ms=duration_ms,
                    response=data
                ))
    
    async def test_node_caching(self, session: aiohttp.ClientSession):
        """测试Node Caching功能"""
        start_time = time.time()
        
        # 测试启用缓存
        payload = {
            "message": "缓存测试消息",
            "enable_cache": True,
            "cache_ttl": 300
        }
        
        async with session.post(
            f"{self.base_url}/api/v1/langgraph/caching/demo",
            json=payload
        ) as response:
            duration_ms = (time.time() - start_time) * 1000
            data = await response.json()
            
            assert response.status == 200, "缓存测试状态码错误"
            assert data["success"] is True, "缓存测试失败"
            assert "cache_statistics" in data["metadata"], "缓存统计缺失"
            assert data["metadata"]["cache_statistics"]["cache_enabled"] is True, "缓存未启用"
            
            self.test_results.append(TestResult(
                name="Node Caching功能",
                passed=True,
                duration_ms=duration_ms,
                response=data
            ))
    
    async def test_hooks_functionality(self, session: aiohttp.ClientSession):
        """测试Pre/Post Hooks功能"""
        start_time = time.time()
        
        payload = {
            "messages": [
                {"role": "user", "content": "测试钩子功能的消息内容"}
            ],
            "enable_pre_hooks": True,
            "enable_post_hooks": True
        }
        
        async with session.post(
            f"{self.base_url}/api/v1/langgraph/hooks/demo",
            json=payload
        ) as response:
            duration_ms = (time.time() - start_time) * 1000
            data = await response.json()
            
            assert response.status == 200, "Hooks测试状态码错误"
            assert data["success"] is True, "Hooks测试失败"
            assert data["metadata"]["pre_hooks_enabled"] is True, "Pre hooks未启用"
            assert data["metadata"]["post_hooks_enabled"] is True, "Post hooks未启用"
            assert data["metadata"]["final_message_count"] > data["metadata"]["original_message_count"], "钩子未生效"
            
            self.test_results.append(TestResult(
                name="Pre/Post Hooks功能",
                passed=True,
                duration_ms=duration_ms,
                response=data
            ))
    
    async def test_hooks_management(self, session: aiohttp.ClientSession):
        """测试钩子状态管理"""
        start_time = time.time()
        
        # 获取钩子状态
        async with session.get(f"{self.base_url}/api/v1/langgraph/hooks/status") as response:
            duration_ms = (time.time() - start_time) * 1000
            data = await response.json()
            
            assert response.status == 200, "钩子状态查询失败"
            assert "pre_hooks" in data, "Pre hooks状态缺失"
            assert "post_hooks" in data, "Post hooks状态缺失"
            assert len(data["pre_hooks"]) > 0, "Pre hooks为空"
            assert len(data["post_hooks"]) > 0, "Post hooks为空"
            
            # 验证钩子信息结构
            for hook in data["pre_hooks"] + data["post_hooks"]:
                assert "name" in hook, "钩子名称缺失"
                assert "enabled" in hook, "钩子启用状态缺失"
                assert "priority" in hook, "钩子优先级缺失"
                assert "description" in hook, "钩子描述缺失"
            
            self.test_results.append(TestResult(
                name="钩子状态管理",
                passed=True,
                duration_ms=duration_ms,
                response=data
            ))
    
    async def test_cache_statistics(self, session: aiohttp.ClientSession):
        """测试缓存统计"""
        start_time = time.time()
        
        async with session.get(f"{self.base_url}/api/v1/langgraph/cache/stats") as response:
            duration_ms = (time.time() - start_time) * 1000
            data = await response.json()
            
            assert response.status == 200, "缓存统计查询失败"
            assert "cache_backend" in data, "缓存后端信息缺失"
            assert "default_policy" in data, "默认策略信息缺失"
            assert "node_policies_count" in data, "节点策略计数缺失"
            
            # 验证默认策略结构
            policy = data["default_policy"]
            assert "ttl" in policy, "TTL配置缺失"
            assert "max_size" in policy, "最大大小配置缺失"
            assert "enabled" in policy, "启用状态缺失"
            
            self.test_results.append(TestResult(
                name="缓存统计查询",
                passed=True,
                duration_ms=duration_ms,
                response=data
            ))
    
    async def test_complete_demo(self, session: aiohttp.ClientSession):
        """测试完整功能演示"""
        start_time = time.time()
        
        async with session.post(f"{self.base_url}/api/v1/langgraph/complete-demo") as response:
            duration_ms = (time.time() - start_time) * 1000
            data = await response.json()
            
            assert response.status == 200, "完整演示状态码错误"
            assert data["success"] is True, "完整演示失败"
            assert "features_demonstrated" in data["metadata"], "演示特性列表缺失"
            assert len(data["metadata"]["features_demonstrated"]) >= 3, "演示特性数量不足"
            assert data["metadata"]["workflow_type"] == "conditional_workflow", "工作流类型不正确"
            
            self.test_results.append(TestResult(
                name="完整功能演示",
                passed=True,
                duration_ms=duration_ms,
                response=data
            ))
    
    async def test_error_handling(self, session: aiohttp.ClientSession):
        """测试错误处理"""
        start_time = time.time()
        
        # 测试无效的durability模式
        payload = {
            "message": "错误测试",
            "durability_mode": "invalid_mode"
        }
        
        async with session.post(
            f"{self.base_url}/api/v1/langgraph/durability/demo",
            json=payload
        ) as response:
            duration_ms = (time.time() - start_time) * 1000
            
            # 应该返回422验证错误
            assert response.status == 422, f"错误处理状态码不正确: {response.status}"
            
            self.test_results.append(TestResult(
                name="错误处理测试",
                passed=True,
                duration_ms=duration_ms,
                response={"status": response.status}
            ))
    
    async def test_concurrent_requests(self, session: aiohttp.ClientSession):
        """测试并发请求"""
        start_time = time.time()
        
        # 创建多个并发请求
        tasks = []
        for i in range(5):
            payload = {
                "message": f"并发测试消息 {i+1}"
            }
            task = session.post(
                f"{self.base_url}/api/v1/langgraph/context-api/demo",
                json=payload
            )
            tasks.append(task)
        
        # 等待所有请求完成
        responses = await asyncio.gather(*tasks)
        duration_ms = (time.time() - start_time) * 1000
        
        success_count = 0
        for response in responses:
            if response.status == 200:
                data = await response.json()
                if data.get("success"):
                    success_count += 1
            response.close()
        
        assert success_count == 5, f"并发请求成功数量不正确: {success_count}/5"
        
        self.test_results.append(TestResult(
            name="并发请求测试",
            passed=True,
            duration_ms=duration_ms,
            response={"concurrent_requests": 5, "success_count": success_count}
        ))
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(result.duration_ms for result in self.test_results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
                "total_duration_ms": total_duration,
                "average_duration_ms": avg_duration
            },
            "test_results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "duration_ms": result.duration_ms,
                    "error": result.error
                }
                for result in self.test_results
            ],
            "features_tested": [
                "Context API (新旧兼容)",
                "Durability Control (exit/async/sync)",
                "Node Caching",
                "Pre/Post Model Hooks",
                "钩子状态管理",
                "缓存统计",
                "完整工作流演示",
                "错误处理",
                "并发处理"
            ]
        }
        
        return report

async def main():
    """主函数"""
    print("=" * 60)
    print("LangGraph 0.6.5 新特性 E2E 测试套件")
    print("=" * 60)
    
    test_suite = LangGraphE2ETestSuite()
    report = await test_suite.run_all_tests()
    
    print("\n" + "=" * 60)
    print("📊 测试报告摘要")
    print("=" * 60)
    
    summary = report["summary"]
    print(f"总计测试: {summary['total_tests']}")
    print(f"通过测试: {summary['passed']}")
    print(f"失败测试: {summary['failed']}")
    print(f"成功率: {summary['success_rate']}")
    print(f"总执行时间: {summary['total_duration_ms']:.2f}ms")
    print(f"平均执行时间: {summary['average_duration_ms']:.2f}ms")
    
    print("\n📋 测试详情:")
    for result in report["test_results"]:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {result['name']} ({result['duration_ms']:.2f}ms)")
        if result.get("error"):
            print(f"   错误: {result['error']}")
    
    print("\n🎯 已测试功能:")
    for feature in report["features_tested"]:
        print(f"• {feature}")
    
    # 保存详细报告到文件
    with open("langgraph_e2e_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细报告已保存到: langgraph_e2e_test_report.json")
    
    if summary["failed"] > 0:
        print("\n⚠️  部分测试失败，请检查错误信息")
        sys.exit(1)
    else:
        print("\n🎉 所有测试通过！LangGraph 0.6.5 新特性工作正常")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
