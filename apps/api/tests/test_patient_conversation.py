#!/usr/bin/env python3
"""耐心等待的多智能体对话测试"""

import asyncio
import websockets
import json
import time
from typing import Dict, Any

class PatientConversationTest:
    """耐心等待的多智能体对话测试"""
    
    def __init__(self):
        self.ws_url = "ws://localhost:8000/api/v1/multi-agent/ws/test-session-patient"
        self.messages_received = []
        self.speaker_changes = []
        self.agent_responses = []
        
    async def test_patient_conversation(self):
        """耐心等待的对话测试"""
        print("⏳ 开始耐心等待的多智能体对话测试...")
        print("💡 将等待足够长的时间以让智能体完成响应（最多2分钟）")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                print(f"✅ WebSocket连接成功: {self.ws_url}")
                
                # 设置消息监听
                listen_task = asyncio.create_task(self.listen_messages(websocket))
                
                # 等待连接确认
                await asyncio.sleep(1)
                
                # 启动对话
                await self.start_conversation(websocket)
                
                # 耐心等待 - 每10秒显示一次状态
                total_wait_time = 120  # 等待2分钟
                check_interval = 10    # 每10秒检查一次
                
                for elapsed in range(0, total_wait_time, check_interval):
                    print(f"⏰ 已等待 {elapsed}秒, 收到消息数: {len(self.messages_received)}, 智能体响应数: {len(self.agent_responses)}")
                    
                    await asyncio.sleep(check_interval)
                    
                    # 如果收到了智能体响应，再等一轮看看是否有更多
                    if len(self.agent_responses) > 0 and elapsed >= 60:
                        print("📝 已收到智能体响应，再等待30秒看是否有更多...")
                        await asyncio.sleep(30)
                        break
                
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
                    
                    print(f"💬 {sender}: {content[:150]}...")
                
                # 对话完成
                elif msg_type == 'conversation_completed':
                    print("🏁 对话已完成")
                
                # 其他消息
                elif msg_type in ['connection_established', 'conversation_created', 'conversation_started']:
                    print(f"📨 收到消息: {msg_type}")
                else:
                    print(f"📨 收到未知消息: {msg_type}")
                    
        except asyncio.CancelledError:
            print("📡 停止监听消息")
        except Exception as e:
            print(f"❌ 监听消息失败: {e}")
    
    async def start_conversation(self, websocket):
        """启动对话"""
        start_message = {
            "type": "start_conversation",
            "data": {
                "message": "请三位专家简要介绍各自的专业领域和核心职责",
                "participants": ["code_expert", "architect", "doc_expert"]
            },
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        
        print(f"🚀 发送对话启动消息: 专家自我介绍...")
        await websocket.send(json.dumps(start_message))
    
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "="*80)
        print("📊 耐心等待测试结果")
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
            print(f"\n🤖 智能体响应详情:")
            for i, response in enumerate(self.agent_responses):
                print(f"\n  📝 响应 {i+1}:")
                print(f"     发送者: {response['sender']}")
                print(f"     时间: {response['timestamp']}")
                print(f"     内容长度: {len(response['content'])}字符")
                print(f"     内容预览: {response['content'][:100]}...")
        
        # 总体评估
        print(f"\n🎯 测试结果评估:")
        if len(self.agent_responses) >= 3:
            print(f"  🎉 完美! 所有3个智能体都生成了响应")
            print(f"  ✅ 多智能体协作系统完全正常")
        elif len(self.agent_responses) >= 1:
            print(f"  ✅ 部分成功! {len(self.agent_responses)}个智能体生成了响应")
            print(f"  💡 系统基本正常，可能需要调整超时设置")
        else:
            print(f"  ❌ 测试失败! 没有收到任何智能体响应")
            print(f"  💡 需要检查后端对话执行逻辑")
            
        if len(self.speaker_changes) > 0:
            print(f"  ✅ 发言者状态更新正常")
        else:
            print(f"  ⚠️  未收到发言者变更通知")
        
        print("="*80)

async def main():
    """主测试函数"""
    tester = PatientConversationTest()
    await tester.test_patient_conversation()

if __name__ == "__main__":
    asyncio.run(main())