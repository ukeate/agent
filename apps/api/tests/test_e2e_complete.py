#!/usr/bin/env python3
"""
完整E2E测试 - 验证所有智能体都能发出OpenAI响应并体现专业特色
"""

import asyncio
import websockets
import json
import time
import requests
from typing import Dict, Any

class CompleteE2ETest:
    """完整E2E测试 - 验证所有智能体响应"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.results = {}
        self.agent_responses = []
        self.speaker_changes = []
        
        # 专业关键词验证
        self.professional_keywords = {
            'code_expert': ['代码', '编程', '开发', '实现', '函数', '类', '算法', '性能', '优化', '调试'],
            'architect': ['架构', '设计', '系统', '模块', '组件', '可扩展', '微服务', '分层', '技术', '选型'],
            'doc_expert': ['文档', '说明', '规范', '手册', '指南', '流程', '步骤', '示例', '格式', '标准']
        }
    
    async def run_complete_test(self):
        """运行完整E2E测试"""
        print("🚀 完整E2E测试 - 验证所有智能体OpenAI响应")
        print("="*70)
        
        # 1. 前置检查
        if not await self.pre_check():
            print("❌ 前置检查失败，测试终止")
            return False
            
        # 2. 执行完整对话测试
        await self.test_full_conversation()
        
        # 3. 验证结果
        success = self.validate_all_agents_responded()
        
        return success
    
    async def pre_check(self):
        """前置检查"""
        print("\n📋 1. 前置检查...")
        
        # 检查后端服务
        try:
            response = requests.get(f"{self.backend_url}/api/v1/multi-agent/agents", timeout=5)
            if response.status_code != 200:
                print("  ❌ 后端服务异常")
                return False
                
            data = response.json()
            agents = data.get('data', {}).get('agents', [])
            
            if len(agents) < 3:
                print(f"  ❌ 智能体数量不足: {len(agents)}")
                return False
                
            print(f"  ✅ 发现 {len(agents)} 个智能体")
            for agent in agents:
                print(f"    - {agent.get('name')} ({agent.get('role')})")
                
            return True
            
        except Exception as e:
            print(f"  ❌ 前置检查失败: {e}")
            return False
    
    async def test_full_conversation(self):
        """测试完整对话流程"""
        print("\n📋 2. 执行完整对话测试...")
        
        session_id = f"complete-test-{int(time.time())}"
        ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print("  ✅ WebSocket连接成功")
                
                # 启动消息监听
                listen_task = asyncio.create_task(self.listen_all_messages(websocket))
                
                await asyncio.sleep(1)
                
                # 发送对话启动请求
                await self.send_comprehensive_request(websocket)
                
                # 等待足够时间让所有智能体响应 (2分钟)
                await self.wait_for_all_responses(120)
                
                listen_task.cancel()
                
        except Exception as e:
            print(f"  ❌ WebSocket测试失败: {e}")
    
    async def listen_all_messages(self, websocket):
        """监听所有WebSocket消息"""
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type', 'unknown')
                
                if msg_type == 'connection_established':
                    print("    📡 连接建立")
                    
                elif msg_type == 'conversation_started':
                    print("    🚀 对话启动")
                    
                elif msg_type == 'speaker_change':
                    speaker = data.get('data', {}).get('current_speaker', 'Unknown')
                    round_num = data.get('data', {}).get('round', 0)
                    self.speaker_changes.append({'speaker': speaker, 'round': round_num})
                    print(f"    🗣️  发言者变更: {speaker} (轮次: {round_num})")
                    
                elif msg_type in ['new_message', 'agent_message']:
                    await self.collect_agent_response(data)
                    
                elif msg_type == 'conversation_completed':
                    print("    🏁 对话完成")
                    
        except asyncio.CancelledError:
            print("    📡 停止监听")
        except Exception as e:
            print(f"    ❌ 消息监听错误: {e}")
    
    async def collect_agent_response(self, data):
        """收集智能体响应"""
        message_data = data.get('data', {})
        if 'message' in message_data:
            message_info = message_data['message']
        else:
            message_info = message_data
            
        sender = message_info.get('sender', 'Unknown')
        content = message_info.get('content', '')
        timestamp = message_info.get('timestamp', '')
        
        # 判断智能体类型
        agent_type = None
        for agent_key in self.professional_keywords.keys():
            if agent_key in sender.lower():
                agent_type = agent_key
                break
        
        response_data = {
            'sender': sender,
            'agent_type': agent_type,
            'content': content,
            'timestamp': timestamp,
            'length': len(content)
        }
        
        self.agent_responses.append(response_data)
        
        print(f"    💬 收到响应 #{len(self.agent_responses)}: {sender}")
        print(f"       长度: {len(content)} 字符")
        print(f"       预览: {content[:80]}...")
        
        # 检查专业性
        if agent_type and content:
            keywords = self.professional_keywords[agent_type]
            found_keywords = [kw for kw in keywords if kw in content.lower()]
            if found_keywords:
                print(f"       ✅ 专业性确认: {', '.join(found_keywords[:3])}")
            else:
                print(f"       ❓ 专业性待确认")
    
    async def send_comprehensive_request(self, websocket):
        """发送综合测试请求"""
        request = {
            "type": "start_conversation",
            "data": {
                "message": "请三位专家从各自的专业角度分析：如何设计一个高质量的在线教育平台？请每位专家重点阐述自己负责的方面。",
                "participants": ["code_expert", "architect", "doc_expert"]
            },
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        
        print("    📤 发送综合测试请求...")
        await websocket.send(json.dumps(request))
    
    async def wait_for_all_responses(self, max_seconds):
        """等待所有智能体响应"""
        print(f"  ⏳ 等待所有智能体响应 (最多{max_seconds}秒)...")
        
        start_time = time.time()
        target_responses = 3  # 期望收到3个智能体的响应
        
        while time.time() - start_time < max_seconds:
            unique_agents = set()
            for response in self.agent_responses:
                if response['agent_type']:
                    unique_agents.add(response['agent_type'])
            
            elapsed = int(time.time() - start_time)
            
            print(f"    ⏰ {elapsed}秒: 已收到 {len(self.agent_responses)} 个响应，覆盖 {len(unique_agents)} 个智能体")
            
            # 如果已收到所有期望的智能体响应，再等待一些时间以防有更多响应
            if len(unique_agents) >= target_responses:
                print(f"    ✅ 已收集到所有期望智能体的响应，再等待30秒...")
                await asyncio.sleep(30)
                break
            
            await asyncio.sleep(10)  # 每10秒检查一次
    
    def validate_all_agents_responded(self):
        """验证所有智能体都响应了"""
        print("\n📋 3. 验证智能体响应...")
        
        # 按智能体类型分组响应
        responses_by_agent = {}
        for response in self.agent_responses:
            agent_type = response['agent_type']
            if agent_type:
                if agent_type not in responses_by_agent:
                    responses_by_agent[agent_type] = []
                responses_by_agent[agent_type].append(response)
        
        print(f"  📊 响应统计:")
        print(f"    总响应数: {len(self.agent_responses)}")
        print(f"    智能体覆盖数: {len(responses_by_agent)}")
        print(f"    发言者变更: {len(self.speaker_changes)}")
        
        # 详细分析每个智能体
        print(f"\n  🤖 智能体详细分析:")
        all_agents_responded = True
        all_agents_professional = True
        
        for agent_type, expected_keywords in self.professional_keywords.items():
            agent_responses = responses_by_agent.get(agent_type, [])
            
            if agent_responses:
                response_count = len(agent_responses)
                total_length = sum(r['length'] for r in agent_responses)
                
                # 检查专业性
                all_content = ' '.join(r['content'].lower() for r in agent_responses)
                found_keywords = [kw for kw in expected_keywords if kw in all_content]
                is_professional = len(found_keywords) >= 2  # 至少包含2个专业关键词
                
                status = "✅" if is_professional else "❓"
                print(f"    {agent_type}: {response_count}个响应, {total_length}字符 {status}")
                
                if found_keywords:
                    print(f"      专业关键词: {', '.join(found_keywords[:5])}")
                    
                if not is_professional:
                    all_agents_professional = False
                    
            else:
                print(f"    {agent_type}: ❌ 无响应")
                all_agents_responded = False
        
        # 最终评估
        print(f"\n🎯 最终评估:")
        
        expected_agents = len(self.professional_keywords)
        responded_agents = len(responses_by_agent)
        
        coverage_rate = (responded_agents / expected_agents) * 100
        
        print(f"  智能体覆盖率: {responded_agents}/{expected_agents} ({coverage_rate:.1f}%)")
        print(f"  全部响应: {'✅' if all_agents_responded else '❌'}")
        print(f"  专业性合格: {'✅' if all_agents_professional else '❌'}")
        print(f"  OpenAI调用: {'✅' if len(self.agent_responses) > 0 else '❌'}")
        
        # 显示具体响应内容摘要
        if self.agent_responses:
            print(f"\n📝 响应内容摘要:")
            for i, response in enumerate(self.agent_responses[:6]):  # 最多显示6个响应
                agent_name = response['sender']
                preview = response['content'][:100] + "..." if len(response['content']) > 100 else response['content']
                print(f"  {i+1}. {agent_name}:")
                print(f"     {preview}")
                print()
        
        success = all_agents_responded and all_agents_professional and len(self.agent_responses) >= 3
        
        if success:
            print("🎉 完整E2E测试成功! 所有智能体都正常工作并体现专业特色")
        elif responded_agents >= 2:
            print("✅ 部分成功! 大部分智能体正常工作")
        else:
            print("❌ E2E测试失败! 需要检查系统配置")
        
        return success

async def main():
    """主函数"""
    tester = CompleteE2ETest()
    success = await tester.run_complete_test()
    
    print("\n" + "="*70)
    if success:
        print("🎊 完整E2E测试全部通过! 多智能体系统完美运行!")
    else:
        print("⚠️  E2E测试部分通过，建议进一步优化")
    print("="*70)
    
    return success

if __name__ == "__main__":
    asyncio.run(main())