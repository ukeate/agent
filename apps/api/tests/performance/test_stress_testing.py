"""压力测试和稳定性测试"""

import pytest
import asyncio
import aiohttp
import time
from typing import List, Dict, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timedelta
import psutil
import gc
import tracemalloc
import random
import json


class StressTestMetrics:
    """压力测试指标"""
    def __init__(self):
        self.start_time = None
        self.checkpoints = []
        self.errors = []
        self.memory_samples = []
        self.cpu_samples = []
        self.response_times = []
        
    def add_checkpoint(self, name: str, data: Dict[str, Any]):
        """添加检查点"""
        self.checkpoints.append({
            "timestamp": utc_now(),
            "name": name,
            "data": data,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        })
    
    def add_error(self, error: str):
        """记录错误"""
        self.errors.append({
            "timestamp": utc_now(),
            "error": error
        })
    
    def sample_resources(self):
        """采样系统资源"""
        process = psutil.Process()
        self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_samples.append(process.cpu_percent())
    
    def get_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        if not self.memory_samples:
            return {"error": "No samples collected"}
        
        return {
            "duration": time.time() - self.start_time if self.start_time else 0,
            "total_errors": len(self.errors),
            "checkpoints": len(self.checkpoints),
            "avg_memory_mb": sum(self.memory_samples) / len(self.memory_samples),
            "max_memory_mb": max(self.memory_samples),
            "avg_cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            "max_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0
        }


class StressTestScenarios:
    """压力测试场景"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_long_running_stability(self):
        """长时间运行稳定性测试（简化版）"""
        metrics = StressTestMetrics()
        metrics.start_time = time.time()
        
        # 测试配置（简化版本，实际可运行24小时）
        test_duration = 60  # 60秒用于演示，实际应该是 24 * 3600
        checkpoint_interval = 10  # 每10秒一个检查点
        
        base_url = "http://localhost:8000/api/v1"
        
        async with aiohttp.ClientSession() as session:
            end_time = time.time() + test_duration
            checkpoint_time = time.time() + checkpoint_interval
            
            request_count = 0
            error_count = 0
            
            while time.time() < end_time:
                # 执行请求
                try:
                    async with session.get(f"{base_url}/health") as response:
                        await response.text()
                        request_count += 1
                        
                        if response.status != 200:
                            error_count += 1
                            metrics.add_error(f"HTTP {response.status}")
                            
                except Exception as e:
                    error_count += 1
                    metrics.add_error(str(e))
                
                # 采样资源使用
                if request_count % 100 == 0:
                    metrics.sample_resources()
                
                # 检查点
                if time.time() >= checkpoint_time:
                    metrics.add_checkpoint(
                        "periodic_check",
                        {
                            "requests": request_count,
                            "errors": error_count,
                            "error_rate": error_count / request_count if request_count > 0 else 0
                        }
                    )
                    checkpoint_time += checkpoint_interval
                    
                    # 强制垃圾回收
                    gc.collect()
                
                # 短暂延迟避免过载
                await asyncio.sleep(0.01)
        
        # 最终统计
        summary = metrics.get_summary()
        
        # 验证稳定性
        assert summary["total_errors"] / request_count < 0.01 if request_count > 0 else True  # 错误率小于1%
        assert summary["max_memory_mb"] < summary["avg_memory_mb"] * 2  # 内存没有异常增长
    
    @pytest.mark.asyncio
    async def test_extreme_load(self):
        """极限负载测试"""
        metrics = StressTestMetrics()
        metrics.start_time = time.time()
        
        base_url = "http://localhost:8000/api/v1"
        target_rps = 1000  # 目标1000 RPS
        test_duration = 10  # 10秒测试
        
        async def make_request(session):
            """发起单个请求"""
            try:
                start = time.time()
                async with session.get(f"{base_url}/health") as response:
                    await response.text()
                    response_time = time.time() - start
                    metrics.response_times.append(response_time)
                    return response.status == 200
            except Exception as e:
                metrics.add_error(str(e))
                return False
        
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, limit_per_host=0)
        ) as session:
            
            # 预热
            await make_request(session)
            
            # 极限负载测试
            start_time = time.time()
            successful_requests = 0
            total_requests = 0
            
            while time.time() - start_time < test_duration:
                # 计算需要发送的请求数以达到目标RPS
                elapsed = time.time() - start_time
                expected_requests = int(target_rps * elapsed)
                requests_to_send = expected_requests - total_requests
                
                if requests_to_send > 0:
                    # 批量发送请求
                    batch_size = min(requests_to_send, 100)
                    tasks = [make_request(session) for _ in range(batch_size)]
                    results = await asyncio.gather(*tasks)
                    
                    successful_requests += sum(results)
                    total_requests += batch_size
                    
                    # 采样资源
                    if total_requests % 100 == 0:
                        metrics.sample_resources()
                
                await asyncio.sleep(0.001)  # 微小延迟
        
        # 计算实际RPS
        actual_duration = time.time() - start_time
        actual_rps = total_requests / actual_duration if actual_duration > 0 else 0
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # 记录结果
        metrics.add_checkpoint("extreme_load_complete", {
            "target_rps": target_rps,
            "actual_rps": actual_rps,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate
        })
        
        # 验证性能
        assert actual_rps > target_rps * 0.8  # 达到目标RPS的80%
        assert success_rate > 0.95  # 95%成功率
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """内存泄漏检测测试"""
        tracemalloc.start()
        metrics = StressTestMetrics()
        metrics.start_time = time.time()
        
        base_url = "http://localhost:8000/api/v1"
        
        # 创建大量临时对象测试内存管理
        async def create_and_destroy_agents(session, batch_id):
            """创建并销毁智能体"""
            agents = []
            
            # 创建智能体
            for i in range(10):
                payload = {
                    "name": f"TempAgent_{batch_id}_{i}",
                    "system_message": "Temporary agent for memory testing" * 100,  # 大消息
                    "llm_config": {"model": "gpt-4o-mini"}
                }
                
                try:
                    async with session.post(f"{base_url}/agents", json=payload) as response:
                        if response.status in [200, 201]:
                            agent_data = await response.json()
                            agents.append(agent_data.get("id"))
                except Exception as e:
                    metrics.add_error(f"Create agent error: {e}")
            
            # 使用智能体
            for agent_id in agents:
                if agent_id:
                    try:
                        payload = {
                            "agent_id": agent_id,
                            "message": "Test message" * 100,  # 大消息
                            "conversation_id": f"test_conv_{batch_id}"
                        }
                        async with session.post(f"{base_url}/messages", json=payload) as response:
                            await response.text()
                    except Exception as e:
                        metrics.add_error(f"Message error: {e}")
            
            # 清理（模拟删除）
            agents.clear()
            
            # 强制垃圾回收
            gc.collect()
        
        async with aiohttp.ClientSession() as session:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # 执行多轮创建和销毁
            for batch in range(5):
                await create_and_destroy_agents(session, batch)
                
                # 记录内存使用
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                metrics.add_checkpoint(f"batch_{batch}", {
                    "memory_mb": current_memory,
                    "growth_mb": memory_growth
                })
                
                # 等待一段时间让系统稳定
                await asyncio.sleep(2)
                gc.collect()
        
        # 获取内存快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        tracemalloc.stop()
        
        # 分析内存增长
        checkpoints = metrics.checkpoints
        if len(checkpoints) >= 2:
            first_memory = checkpoints[0]["data"]["memory_mb"]
            last_memory = checkpoints[-1]["data"]["memory_mb"]
            memory_growth_rate = (last_memory - first_memory) / first_memory if first_memory > 0 else 0
            
            # 验证没有严重内存泄漏
            assert memory_growth_rate < 0.5  # 内存增长不超过50%
    
    @pytest.mark.asyncio
    async def test_failure_recovery(self):
        """故障恢复能力测试"""
        metrics = StressTestMetrics()
        metrics.start_time = time.time()
        
        base_url = "http://localhost:8000/api/v1"
        
        async def simulate_failure_scenario(session, scenario_type):
            """模拟故障场景"""
            if scenario_type == "network_timeout":
                # 模拟网络超时
                try:
                    async with session.get(
                        f"{base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=0.001)  # 极短超时
                    ) as response:
                        return await response.text()
                except asyncio.TimeoutError:
                    return "timeout"
                    
            elif scenario_type == "invalid_data":
                # 发送无效数据
                try:
                    async with session.post(
                        f"{base_url}/agents",
                        json={"invalid": "data"}
                    ) as response:
                        return response.status
                except Exception as e:
                    return str(e)
                    
            elif scenario_type == "rapid_requests":
                # 快速连续请求
                results = []
                for _ in range(100):
                    try:
                        async with session.get(f"{base_url}/health") as response:
                            results.append(response.status)
                    except Exception:
                        results.append("error")
                return results
        
        async with aiohttp.ClientSession() as session:
            scenarios = ["network_timeout", "invalid_data", "rapid_requests"]
            recovery_times = []
            
            for scenario in scenarios:
                # 执行故障场景
                failure_start = time.time()
                result = await simulate_failure_scenario(session, scenario)
                
                # 等待系统恢复
                await asyncio.sleep(1)
                
                # 测试恢复
                recovery_start = time.time()
                recovered = False
                
                for _ in range(10):
                    try:
                        async with session.get(f"{base_url}/health") as response:
                            if response.status == 200:
                                recovered = True
                                recovery_time = time.time() - recovery_start
                                recovery_times.append(recovery_time)
                                break
                    except Exception:
                        pass
                    
                    await asyncio.sleep(0.5)
                
                metrics.add_checkpoint(f"recovery_{scenario}", {
                    "scenario": scenario,
                    "recovered": recovered,
                    "recovery_time": recovery_time if recovered else None
                })
            
            # 验证恢复能力
            assert all(cp["data"]["recovered"] for cp in metrics.checkpoints)
            assert all(rt < 5.0 for rt in recovery_times)  # 恢复时间小于5秒
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion(self):
        """资源耗尽测试"""
        metrics = StressTestMetrics()
        metrics.start_time = time.time()
        
        base_url = "http://localhost:8000/api/v1"
        
        # 测试连接池耗尽
        async def exhaust_connections():
            """耗尽连接池"""
            connections = []
            
            try:
                # 创建大量持久连接
                for i in range(100):
                    session = aiohttp.ClientSession()
                    connections.append(session)
                    
                    # 发起请求但不关闭
                    asyncio.create_task(session.get(f"{base_url}/health"))
                
                # 等待一段时间
                await asyncio.sleep(2)
                
                # 尝试新请求
                async with aiohttp.ClientSession() as new_session:
                    async with new_session.get(
                        f"{base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        return response.status == 200
                        
            finally:
                # 清理连接
                for session in connections:
                    await session.close()
        
        # 测试CPU密集型操作
        async def cpu_intensive_operation():
            """CPU密集型操作"""
            import hashlib
            
            data = "test" * 10000
            for _ in range(1000):
                hashlib.sha256(data.encode()).hexdigest()
            
            return True
        
        # 执行资源耗尽测试
        connection_result = await exhaust_connections()
        cpu_result = await cpu_intensive_operation()
        
        # 验证系统仍然响应
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                health_check = response.status == 200
        
        metrics.add_checkpoint("resource_exhaustion", {
            "connection_test": connection_result,
            "cpu_test": cpu_result,
            "health_check": health_check
        })
        
        # 系统应该能够处理资源压力
        assert health_check