#!/usr/bin/env python3
"""
快速WebSocket测试 - 验证后端是否正确处理多智能体对话
"""
import asyncio
import websockets
import json
import time

async def quick_test():
    """测试WebSocket多智能体对话"""
    session_id = f"test-{int(time.time() * 1000)}"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🚀 快速测试WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 发送对话启动消息
            start_msg = {
                "type": "start_conversation",
                "data": {
                    "message": "请简单讨论什么是WebSocket？每个参与者用一句话回答。",
                    "participants": ["supervisor-1", "code_expert-1"]
                }
            }
            
            print("📤 发送对话启动消息...")
            await websocket.send(json.dumps(start_msg))
            
            # 监听响应 (只等待30秒)
            message_count = 0
            start_time = time.time()
            
            while time.time() - start_time < 30:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message = json.loads(response)
                    message_count += 1
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] 消息 #{message_count}: {msg_type}")
                    
                    if msg_type == "conversation_created":
                        print(f"   💬 对话已创建: {message['data'].get('conversation_id')[:8]}...")
                        
                    elif msg_type == "conversation_started":
                        print(f"   🚀 对话已启动")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        print(f"   🎤 当前发言者: {speaker}")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        print(f"   📝 Token: {agent} -> '{token}'")
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        content = msg_data.get('content', '')[:50]
                        print(f"   💬 完整消息: {sender} -> {content}...")
                        
                    elif msg_type == "conversation_completed":
                        print(f"   🏁 对话完成")
                        break
                        
                    # 如果收到足够消息，认为测试成功
                    if message_count >= 10:
                        print(f"   ✅ 已收到 {message_count} 条消息，测试成功！")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时 (已收到 {message_count} 条)")
                    continue
                    
            print(f"\n📊 测试结果:")
            print(f"   总消息数: {message_count}")
            print(f"   测试时长: {time.time() - start_time:.1f}秒")
            
            if message_count >= 5:
                print(f"   ✅ 后端功能正常 - 多智能体对话工作正常")
                return True
            else:
                print(f"   ❌ 后端可能存在问题 - 消息数量不足")
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(quick_test())
    exit(0 if result else 1)