#!/usr/bin/env python3
"""测试完整的多智能体对话流程"""

import asyncio
import websockets
import json
import time
from typing import Dict, Any

class FullConversationTest:
    """测试完整的多智能体对话流程"""
    
    def __init__(self):
        self.ws_url = "ws://localhost:8000/api/v1/multi-agent/ws/test-session-full"
        self.messages_received = []
        self.speaker_changes = []
        self.agent_responses = []
        
    async def test_full_conversation(self):
        """测试完整对话流程"""
        print("🔥 开始测试完整多智能体对话流程...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                print(f"✅ WebSocket连接成功: {self.ws_url}")
                
                # 设置消息监听
                listen_task = asyncio.create_task(self.listen_messages(websocket))
                
                # 等待连接确认
                await asyncio.sleep(1)
                
                # 启动对话
                await self.start_conversation(websocket)
                
                # 等待更长时间以观察完整对话
                print("⏱️  等待智能体响应...")
                await asyncio.sleep(15)
                
                # 取消监听任务
                listen_task.cancel()
                
                # 分析结果
                self.analyze_results()
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    async def listen_messages(self, websocket):
        """监听WebSocket消息"""
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type', 'unknown')
                
                self.messages_received.append(data)
                
                # 发言者变更
                if msg_type == 'speaker_change':
                    speaker_info = {
                        'speaker': data.get('data', {}).get('current_speaker'),
                        'round': data.get('data', {}).get('round'),
                        'timestamp': data.get('timestamp')
                    }
                    self.speaker_changes.append(speaker_info)
                    print(f"🗣️  发言者变更: {speaker_info['speaker']} (轮次: {speaker_info['round']})")
                
                # 智能体消息
                elif msg_type in ['new_message', 'agent_message']:
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
                        'timestamp': message_info.get('timestamp')
                    })
                    
                    print(f"💬 {sender}: {content[:100]}...")
                
                # 对话完成
                elif msg_type == 'conversation_completed':
                    print("🏁 对话已完成")
                
                # 其他消息
                else:
                    print(f"📨 收到消息: {msg_type}")
                    
        except asyncio.CancelledError:
            print("📡 停止监听消息")
        except Exception as e:
            print(f"❌ 监听消息失败: {e}")
    
    async def start_conversation(self, websocket):
        """启动对话"""
        start_message = {
            "type": "start_conversation",
            "data": {
                "message": "请分析和设计一个简单的用户认证系统，每个专家从自己的角度简要分析",
                "participants": ["code_expert", "architect", "doc_expert"]
            },
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        
        print(f"🚀 发送对话启动消息: 用户认证系统分析...")
        await websocket.send(json.dumps(start_message))
    
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "="*80)
        print("📊 完整对话流程测试结果")
        print("="*80)
        
        print(f"💬 总消息数: {len(self.messages_received)}")
        print(f"🗣️  发言者变更次数: {len(self.speaker_changes)}")
        print(f"🤖 智能体响应数: {len(self.agent_responses)}")
        
        # 发言者变更详情
        if self.speaker_changes:
            print(f"\n🗣️  发言者变更序列:")
            for i, change in enumerate(self.speaker_changes):
                print(f"  {i+1}. {change['speaker']} (轮次: {change['round']})")
        
        # 智能体响应详情
        if self.agent_responses:
            print(f"\n🤖 智能体响应概览:")
            for i, response in enumerate(self.agent_responses):
                content_preview = response['content'][:50] + "..." if len(response['content']) > 50 else response['content']
                print(f"  {i+1}. {response['sender']}: {content_preview}")
        
        # 验证各个智能体的专业性
        print(f"\n🎯 智能体专业性验证:")
        for response in self.agent_responses:
            sender = response['sender']
            content = response['content'].lower()
            
            if 'code_expert' in sender or '代码专家' in sender:
                if any(keyword in content for keyword in ['代码', 'code', '实现', '函数', '类', '安全', '性能']):
                    print(f"  ✅ {sender}: 体现代码专业性")
                else:
                    print(f"  ❓ {sender}: 专业性不明显")
            
            elif 'architect' in sender or '架构师' in sender:
                if any(keyword in content for keyword in ['架构', '设计', '模块', '系统', '技术', '组件']):
                    print(f"  ✅ {sender}: 体现架构专业性")
                else:
                    print(f"  ❓ {sender}: 专业性不明显")
            
            elif 'doc_expert' in sender or '文档专家' in sender:
                if any(keyword in content for keyword in ['文档', '说明', '规范', '手册', '指南']):
                    print(f"  ✅ {sender}: 体现文档专业性")
                else:
                    print(f"  ❓ {sender}: 专业性不明显")
        
        # 总体评估
        print(f"\n🎯 修复效果总结:")
        if len(self.speaker_changes) > 0:
            print(f"  ✅ 发言者状态更新功能正常")
        
        if len(self.agent_responses) > 0:
            print(f"  ✅ 智能体成功生成响应")
            if len(self.agent_responses) >= 3:
                print(f"  ✅ 多智能体协作正常")
            else:
                print(f"  ⚠️  智能体响应较少，可能对话未完全进行")
        else:
            print(f"  ❌ 智能体未生成响应")
        
        print("="*80)

async def main():
    """主测试函数"""
    tester = FullConversationTest()
    await tester.test_full_conversation()

if __name__ == "__main__":
    asyncio.run(main())