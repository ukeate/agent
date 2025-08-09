#!/usr/bin/env python3
"""测试修复后的多智能体系统 - 验证currentSpeaker状态更新"""

import asyncio
import websockets
import json
import time
from typing import Dict, Any

class SpeakerUpdateTest:
    """测试当前发言者状态更新"""
    
    def __init__(self):
        self.ws_url = "ws://localhost:8000/api/v1/multi-agent/ws/test-session-speaker"
        self.messages_received = []
        self.speaker_changes = []
        
    async def test_speaker_updates(self):
        """测试发言者状态更新"""
        print("🔍 开始测试多智能体发言者状态更新...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                print(f"✅ WebSocket连接成功: {self.ws_url}")
                
                # 设置消息监听
                listen_task = asyncio.create_task(self.listen_messages(websocket))
                
                # 等待连接确认
                await asyncio.sleep(1)
                
                # 启动对话
                await self.start_conversation(websocket)
                
                # 监听5秒以观察发言者变化
                await asyncio.sleep(5)
                
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
                print(f"📨 收到消息: {data.get('type', 'unknown')}")
                
                self.messages_received.append(data)
                
                # 特别关注speaker_change消息
                if data.get('type') == 'speaker_change':
                    speaker_info = {
                        'speaker': data.get('data', {}).get('current_speaker'),
                        'round': data.get('data', {}).get('round'),
                        'timestamp': data.get('timestamp')
                    }
                    self.speaker_changes.append(speaker_info)
                    print(f"🗣️  发言者变更: {speaker_info['speaker']} (轮次: {speaker_info['round']})")
                
                # 记录智能体消息
                elif data.get('type') in ['new_message', 'agent_message']:
                    message_data = data.get('data', {})
                    if 'message' in message_data:
                        message_info = message_data['message']
                    else:
                        message_info = message_data
                        
                    print(f"💬 智能体消息 - {message_info.get('sender', 'Unknown')}: {message_info.get('content', '')[:50]}...")
                    
        except asyncio.CancelledError:
            print("📡 停止监听消息")
        except Exception as e:
            print(f"❌ 监听消息失败: {e}")
    
    async def start_conversation(self, websocket):
        """启动对话"""
        start_message = {
            "type": "start_conversation",
            "data": {
                "message": "请每个智能体简要介绍自己的职责",
                "participants": ["code_expert", "architect", "doc_expert"]
            },
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        }
        
        print(f"🚀 发送对话启动消息...")
        await websocket.send(json.dumps(start_message))
    
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "="*60)
        print("📊 测试结果分析")
        print("="*60)
        
        print(f"💬 总消息数: {len(self.messages_received)}")
        print(f"🗣️  发言者变更次数: {len(self.speaker_changes)}")
        
        # 分析消息类型
        message_types = {}
        for msg in self.messages_received:
            msg_type = msg.get('type', 'unknown')
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        print(f"\n📋 消息类型统计:")
        for msg_type, count in message_types.items():
            print(f"  - {msg_type}: {count}")
        
        # 分析发言者变更
        if self.speaker_changes:
            print(f"\n🗣️  发言者变更详情:")
            for i, change in enumerate(self.speaker_changes):
                print(f"  {i+1}. {change['speaker']} (轮次: {change['round']})")
        else:
            print(f"\n⚠️  未检测到发言者变更消息")
        
        # 验证修复效果
        print(f"\n🎯 修复效果验证:")
        if len(self.speaker_changes) > 0:
            print(f"  ✅ 发言者状态更新正常 - 检测到 {len(self.speaker_changes)} 次发言者变更")
        else:
            print(f"  ❓ 需要检查后端是否发送speaker_change消息")
        
        print("="*60)

async def main():
    """主测试函数"""
    tester = SpeakerUpdateTest()
    await tester.test_speaker_updates()

if __name__ == "__main__":
    asyncio.run(main())