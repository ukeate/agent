#!/usr/bin/env python3
"""
完整的端到端测试 - 验证多智能体系统从前端到后端的完整流程
包括前端界面交互、WebSocket通信、智能体角色响应、OpenAI API调用
"""

import asyncio
import websockets
import json
import time
import requests
from typing import Dict, Any, List
import re

class E2EMultiAgentTest:
    """端到端多智能体测试"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3002"
        self.ws_base_url = "ws://localhost:8000/api/v1/multi-agent/ws"
        
        # 测试数据收集
        self.test_results = {
            'frontend_accessible': False,
            'backend_accessible': False,
            'agents_loaded': False,
            'websocket_connected': False,
            'conversation_created': False,
            'all_agents_responded': False,
            'responses_professional': False,
            'conversation_completed': False,
        }
        
        self.agent_responses = []
        self.speaker_changes = []
        self.all_messages = []
        
        # 期望的智能体和他们的专业关键词
        self.expected_agents = {
            'code_expert': ['代码', '实现', '函数', '类', '安全', '性能', '算法', '数据库', '接口'],
            'architect': ['架构', '设计', '模块', '系统', '微服务', '组件', '可扩展', '技术'],
            'doc_expert': ['文档', '说明', '规范', '手册', '指南', '流程', '步骤', '示例']
        }
    
    async def run_full_e2e_test(self):
        """运行完整的E2E测试"""
        print("🚀 开始完整的端到端(E2E)多智能体测试")
        print("="*80)
        
        # 1. 检查前端和后端服务
        await self.check_services()
        
        # 2. 验证后端API
        await self.verify_backend_api()
        
        # 3. 执行WebSocket对话测试
        await self.test_websocket_conversation()
        
        # 4. 分析测试结果
        await self.analyze_final_results()
        
        return self.test_results
    
    async def check_services(self):
        """检查前端和后端服务状态"""
        print("📋 1. 检查服务状态...")
        
        # 检查前端服务
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                self.test_results['frontend_accessible'] = True
                print("  ✅ 前端服务正常 (http://localhost:3002)")
            else:
                print(f"  ❌ 前端服务异常: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ 前端服务无法访问: {e}")
        
        # 检查后端服务
        try:
            response = requests.get(f"{self.backend_url}/api/v1/agent/status", timeout=5)
            if response.status_code == 200:
                self.test_results['backend_accessible'] = True
                print("  ✅ 后端服务正常 (http://localhost:8000)")
            else:
                print(f"  ❌ 后端服务异常: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ 后端服务无法访问: {e}")
    
    async def verify_backend_api(self):
        """验证后端多智能体API"""
        print("\n📋 2. 验证后端多智能体API...")
        
        try:
            # 获取智能体列表
            response = requests.get(f"{self.backend_url}/api/v1/multi-agent/agents", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('data', {}).get('agents'):
                    agents = data['data']['agents']
                    self.test_results['agents_loaded'] = True
                    print(f"  ✅ 成功加载 {len(agents)} 个智能体")
                    
                    # 验证每个智能体的配置
                    for agent in agents:
                        name = agent.get('name', 'Unknown')
                        role = agent.get('role', 'Unknown')
                        print(f"    - {name} ({role})")
                else:
                    print("  ❌ 智能体数据格式错误")
            else:
                print(f"  ❌ 智能体API调用失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ 智能体API验证失败: {e}")
    
    async def test_websocket_conversation(self):
        """测试WebSocket对话流程"""
        print("\n📋 3. 执行WebSocket对话测试...")
        
        session_id = f"e2e-test-{int(time.time())}"
        ws_url = f"{self.ws_base_url}/{session_id}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print(f"  ✅ WebSocket连接成功: {ws_url}")
                self.test_results['websocket_connected'] = True
                
                # 启动消息监听
                listen_task = asyncio.create_task(self.listen_websocket_messages(websocket))
                
                # 等待连接建立
                await asyncio.sleep(1)
                
                # 发送对话启动请求
                await self.start_conversation_via_websocket(websocket)
                
                # 等待足够长的时间让所有智能体响应
                print("  ⏳ 等待智能体响应 (最多3分钟)...")
                await self.wait_for_conversation_completion(180)  # 3分钟
                
                # 停止监听
                listen_task.cancel()
                
        except Exception as e:
            print(f"  ❌ WebSocket测试失败: {e}")
    
    async def listen_websocket_messages(self, websocket):
        """监听WebSocket消息"""
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type', 'unknown')
                self.all_messages.append(data)
                
                if msg_type == 'connection_established':
                    print("    📡 连接已确认")
                
                elif msg_type == 'conversation_created':
                    print("    🆕 对话已创建")
                    self.test_results['conversation_created'] = True
                
                elif msg_type == 'conversation_started':
                    print("    🚀 对话已启动")
                
                elif msg_type == 'speaker_change':
                    speaker = data.get('data', {}).get('current_speaker', 'Unknown')
                    round_num = data.get('data', {}).get('round', 0)
                    self.speaker_changes.append({
                        'speaker': speaker,
                        'round': round_num,
                        'timestamp': data.get('timestamp')
                    })
                    print(f"    🗣️  发言者: {speaker} (轮次: {round_num})")
                
                elif msg_type in ['new_message', 'agent_message']:
                    await self.process_agent_message(data)
                
                elif msg_type == 'conversation_completed':
                    print("    🏁 对话完成")
                    self.test_results['conversation_completed'] = True
                
                elif msg_type == 'conversation_error':
                    error = data.get('data', {}).get('error', 'Unknown error')
                    print(f"    ❌ 对话错误: {error}")
                
        except asyncio.CancelledError:
            print("    📡 停止监听WebSocket消息")
        except Exception as e:
            print(f"    ❌ WebSocket消息处理错误: {e}")
    
    async def process_agent_message(self, data):
        """处理智能体消息"""
        message_data = data.get('data', {})
        if 'message' in message_data:
            message_info = message_data['message']
        else:
            message_info = message_data
        
        sender = message_info.get('sender', 'Unknown')
        content = message_info.get('content', '')
        timestamp = message_info.get('timestamp', '')
        
        # 记录响应
        self.agent_responses.append({
            'sender': sender,
            'content': content,
            'timestamp': timestamp,
            'length': len(content)
        })
        
        print(f"    💬 {sender}: {content[:100]}...")
        
        # 检查是否所有期望的智能体都响应了
        responded_agents = set()
        for response in self.agent_responses:
            sender_name = response['sender'].lower()
            for expected_agent in self.expected_agents.keys():
                if expected_agent in sender_name:
                    responded_agents.add(expected_agent)
        
        if len(responded_agents) >= len(self.expected_agents):
            self.test_results['all_agents_responded'] = True
            print(f"    ✅ 所有期望的智能体都已响应: {responded_agents}")
    
    async def start_conversation_via_websocket(self, websocket):
        """通过WebSocket启动对话"""
        start_message = {
            "type": "start_conversation",
            "data": {
                "message": "请三位专家从各自的专业角度，简要分析设计一个电商系统需要考虑的核心要点",
                "participants": ["code_expert", "architect", "doc_expert"]
            },
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        
        print(f"    🚀 发送对话启动请求...")
        await websocket.send(json.dumps(start_message))
    
    async def wait_for_conversation_completion(self, max_wait_seconds):
        """等待对话完成"""
        start_time = time.time()
        last_response_count = 0
        stable_count = 0
        
        while time.time() - start_time < max_wait_seconds:
            current_response_count = len(self.agent_responses)
            
            # 如果响应数量没有增加，开始计算稳定时间
            if current_response_count == last_response_count:
                stable_count += 1
            else:
                stable_count = 0
                last_response_count = current_response_count
            
            # 如果已有足够响应且稳定一段时间，或者对话已标记完成
            if ((current_response_count >= 3 and stable_count >= 6) or  # 3个响应且稳定30秒
                self.test_results['conversation_completed']):
                print(f"    ✅ 对话完成，收到 {current_response_count} 个响应")
                break
            
            # 每5秒显示一次进度
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                print(f"    ⏰ 已等待 {elapsed}秒, 收到 {current_response_count} 个响应")
            
            await asyncio.sleep(5)
    
    def analyze_response_professionalism(self):
        """分析响应的专业性"""
        print("\n📋 4. 分析响应专业性...")
        
        professional_responses = 0
        
        for response in self.agent_responses:
            sender = response['sender'].lower()
            content = response['content'].lower()
            
            # 确定智能体类型
            agent_type = None
            for expected_agent in self.expected_agents.keys():
                if expected_agent in sender:
                    agent_type = expected_agent
                    break
            
            if agent_type:
                keywords = self.expected_agents[agent_type]
                found_keywords = [kw for kw in keywords if kw in content]
                
                if found_keywords:
                    professional_responses += 1
                    print(f"    ✅ {response['sender']}: 体现专业性 (关键词: {', '.join(found_keywords[:3])})")
                else:
                    print(f"    ❓ {response['sender']}: 专业性不明显")
            else:
                print(f"    ❓ {response['sender']}: 未知智能体类型")
        
        if professional_responses >= len(self.expected_agents):
            self.test_results['responses_professional'] = True
            print(f"    ✅ 专业性验证通过: {professional_responses} 个专业响应")
        else:
            print(f"    ❌ 专业性验证失败: 仅 {professional_responses} 个专业响应")
    
    async def analyze_final_results(self):
        """分析最终测试结果"""
        print("\n" + "="*80)
        print("📊 端到端测试结果分析")
        print("="*80)
        
        # 分析响应专业性
        self.analyze_response_professionalism()
        
        # 统计信息
        print(f"\n📈 统计信息:")
        print(f"  💬 总消息数: {len(self.all_messages)}")
        print(f"  🗣️  发言者变更: {len(self.speaker_changes)}")
        print(f"  🤖 智能体响应: {len(self.agent_responses)}")
        
        # 详细的智能体响应分析
        if self.agent_responses:
            print(f"\n🤖 智能体响应详情:")
            for i, response in enumerate(self.agent_responses):
                print(f"  {i+1}. {response['sender']}")
                print(f"     长度: {response['length']} 字符")
                print(f"     时间: {response['timestamp']}")
                print(f"     预览: {response['content'][:80]}...")
                print()
        
        # 测试结果汇总
        print(f"🎯 测试结果汇总:")
        
        passed_tests = 0
        total_tests = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
            if result:
                passed_tests += 1
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n📊 总体成功率: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # 最终评估
        if success_rate >= 80:
            print("🎉 E2E测试总体成功! 多智能体系统可以投入使用")
        elif success_rate >= 60:
            print("⚠️  E2E测试部分成功，建议优化后投入使用")
        else:
            print("❌ E2E测试失败，需要修复关键问题")
        
        return success_rate

async def main():
    """主测试函数"""
    tester = E2EMultiAgentTest()
    results = await tester.run_full_e2e_test()
    
    print("\n" + "="*80)
    print("🏁 E2E测试完成")
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())