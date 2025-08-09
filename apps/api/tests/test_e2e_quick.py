#!/usr/bin/env python3
"""
快速E2E测试 - 验证多智能体系统核心功能
"""

import asyncio
import websockets
import json
import time
import requests
from typing import Dict, Any

class QuickE2ETest:
    """快速E2E测试"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3002"
        self.results = {}
        self.agent_responses = []
        
    async def run_quick_test(self):
        """运行快速E2E测试"""
        print("🚀 快速E2E测试开始...")
        
        # 1. 服务检查
        await self.check_services()
        
        # 2. API验证
        await self.check_agents_api()
        
        # 3. WebSocket对话测试 (限时1分钟)
        await self.test_conversation_with_timeout()
        
        # 4. 结果分析
        self.analyze_results()
    
    async def check_services(self):
        """检查服务状态"""
        print("\n📋 检查服务状态...")
        
        # 前端检查
        try:
            response = requests.get(self.frontend_url, timeout=3)
            frontend_ok = response.status_code == 200
            print(f"  前端: {'✅' if frontend_ok else '❌'}")
        except:
            frontend_ok = False
            print("  前端: ❌ 无法访问")
        
        # 后端检查
        try:
            response = requests.get(f"{self.backend_url}/api/v1/agent/status", timeout=3)
            backend_ok = response.status_code == 200
            print(f"  后端: {'✅' if backend_ok else '❌'}")
        except:
            backend_ok = False
            print("  后端: ❌ 无法访问")
        
        self.results['services'] = frontend_ok and backend_ok
    
    async def check_agents_api(self):
        """检查智能体API"""
        print("\n📋 检查智能体API...")
        
        try:
            response = requests.get(f"{self.backend_url}/api/v1/multi-agent/agents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                agents = data.get('data', {}).get('agents', [])
                agent_count = len(agents)
                print(f"  智能体数量: {agent_count}")
                
                for agent in agents:
                    print(f"    - {agent.get('name')} ({agent.get('role')})")
                
                self.results['agents_api'] = agent_count >= 3
            else:
                print(f"  ❌ API错误: {response.status_code}")
                self.results['agents_api'] = False
        except Exception as e:
            print(f"  ❌ API失败: {e}")
            self.results['agents_api'] = False
    
    async def test_conversation_with_timeout(self):
        """限时对话测试"""
        print("\n📋 WebSocket对话测试 (限时60秒)...")
        
        session_id = f"quick-test-{int(time.time())}"
        ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print("  ✅ WebSocket连接成功")
                
                # 启动监听
                listen_task = asyncio.create_task(self.listen_messages(websocket))
                
                await asyncio.sleep(1)
                
                # 发送启动请求
                await self.send_start_message(websocket)
                
                # 等待60秒
                print("  ⏳ 等待智能体响应...")
                start_time = time.time()
                
                while time.time() - start_time < 60:  # 60秒限时
                    if len(self.agent_responses) >= 1:  # 至少收到1个响应就算成功
                        print(f"  ✅ 收到 {len(self.agent_responses)} 个响应")
                        break
                    await asyncio.sleep(5)
                    elapsed = int(time.time() - start_time)
                    if elapsed % 15 == 0:  # 每15秒显示进度
                        print(f"    等待中... {elapsed}秒")
                
                listen_task.cancel()
                
                self.results['conversation'] = len(self.agent_responses) >= 1
                
        except Exception as e:
            print(f"  ❌ WebSocket测试失败: {e}")
            self.results['conversation'] = False
    
    async def listen_messages(self, websocket):
        """监听消息"""
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type', 'unknown')
                
                if msg_type == 'connection_established':
                    print("    📡 连接确认")
                elif msg_type == 'conversation_started':
                    print("    🚀 对话启动")
                elif msg_type == 'speaker_change':
                    speaker = data.get('data', {}).get('current_speaker', 'Unknown')
                    print(f"    🗣️  发言者: {speaker}")
                elif msg_type in ['new_message', 'agent_message']:
                    await self.process_response(data)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"    ❌ 消息处理错误: {e}")
    
    async def process_response(self, data):
        """处理智能体响应"""
        message_data = data.get('data', {})
        if 'message' in message_data:
            message_info = message_data['message']
        else:
            message_info = message_data
            
        sender = message_info.get('sender', 'Unknown')
        content = message_info.get('content', '')
        
        self.agent_responses.append({
            'sender': sender,
            'content': content,
            'length': len(content)
        })
        
        print(f"    💬 {sender}: {len(content)}字符 - {content[:50]}...")
    
    async def send_start_message(self, websocket):
        """发送启动消息"""
        message = {
            "type": "start_conversation",
            "data": {
                "message": "请简要介绍各自的专业领域",
                "participants": ["code_expert", "architect", "doc_expert"]
            },
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        
        await websocket.send(json.dumps(message))
        print("    📤 发送启动消息")
    
    def analyze_results(self):
        """分析结果"""
        print("\n" + "="*60)
        print("📊 快速E2E测试结果")
        print("="*60)
        
        # 检查专业性
        professional_count = 0
        if self.agent_responses:
            print(f"\n🤖 智能体响应分析:")
            for i, response in enumerate(self.agent_responses):
                sender = response['sender']
                content = response['content'].lower()
                length = response['length']
                
                # 简单的专业性检查
                is_professional = any(word in content for word in [
                    '专业', '技术', '设计', '代码', '架构', '文档', '系统', '开发'
                ])
                
                if is_professional:
                    professional_count += 1
                
                status = "✅" if is_professional else "❓"
                print(f"  {i+1}. {sender}: {length}字符 {status}")
        
        # 总结
        print(f"\n🎯 测试结果:")
        services_ok = self.results.get('services', False)
        agents_ok = self.results.get('agents_api', False)
        conversation_ok = self.results.get('conversation', False)
        professional_ok = professional_count >= 1
        
        print(f"  服务可用性: {'✅' if services_ok else '❌'}")
        print(f"  智能体API: {'✅' if agents_ok else '❌'}")
        print(f"  对话功能: {'✅' if conversation_ok else '❌'}")
        print(f"  响应专业性: {'✅' if professional_ok else '❌'}")
        print(f"  响应数量: {len(self.agent_responses)}")
        
        # 最终评估
        all_passed = services_ok and agents_ok and conversation_ok and professional_ok
        
        if all_passed:
            print("\n🎉 E2E测试全部通过! 系统工作正常")
        elif conversation_ok:
            print("\n✅ 核心功能正常，部分测试通过")
        else:
            print("\n❌ E2E测试失败，需要检查系统状态")
        
        return all_passed

async def main():
    """主函数"""
    tester = QuickE2ETest()
    await tester.run_quick_test()

if __name__ == "__main__":
    asyncio.run(main())