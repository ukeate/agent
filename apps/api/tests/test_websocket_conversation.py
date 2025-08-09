#!/usr/bin/env python3
"""
测试WebSocket对话启动和消息推送
"""
import asyncio
import websockets
import json
import time

async def test_websocket_conversation():
    """测试WebSocket对话功能"""
    session_id = "test-conversation-001"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"连接WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送启动对话消息
            start_message = {
                "type": "start_conversation",
                "message": "测试WebSocket对话",
                "agent_roles": ["code_expert", "architect"],
                "max_rounds": 1  # 只进行1轮避免太长
            }
            
            print(f"发送启动对话消息: {start_message}")
            await websocket.send(json.dumps(start_message))
            
            # 接收并打印消息
            message_count = 0
            start_time = time.time()
            
            while message_count < 10:  # 最多等待10条消息
                try:
                    # 等待消息，每次超时5秒
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    message_count += 1
                    
                    parsed = json.loads(message)
                    msg_type = parsed.get('type', 'unknown')
                    
                    print(f"\n📨 消息 {message_count}: {msg_type}")
                    
                    if msg_type == 'conversation_created':
                        print(f"  ✅ 对话已创建")
                        print(f"  会话ID: {parsed.get('data', {}).get('conversation_id', 'N/A')}")
                    elif msg_type == 'conversation_started':
                        print(f"  ✅ 对话已启动")
                    elif msg_type == 'new_message':
                        msg_data = parsed.get('data', {}).get('message', {})
                        sender = msg_data.get('sender', 'N/A')
                        content = msg_data.get('content', 'N/A')[:150]
                        print(f"  📝 {sender}: {content}...")
                    elif msg_type == 'speaker_change':
                        speaker = parsed.get('data', {}).get('current_speaker', 'N/A')
                        round_num = parsed.get('data', {}).get('round', 'N/A')
                        print(f"  🗣️ 发言者变更: {speaker} (第{round_num}轮)")
                    elif msg_type == 'conversation_completed':
                        print(f"  🎉 对话已完成")
                        break
                    elif msg_type == 'conversation_error':
                        error = parsed.get('data', {}).get('error', 'N/A')
                        print(f"  ❌ 对话错误: {error}")
                        break
                    elif msg_type == 'error':
                        error = parsed.get('data', {}).get('message', 'N/A')
                        print(f"  ❌ 系统错误: {error}")
                        break
                    else:
                        print(f"  ❓ 未知消息类型: {msg_type}")
                    
                    # 避免无限等待
                    if time.time() - start_time > 60:  # 60秒总超时
                        print("⏰ 总测试超时")
                        break
                        
                except asyncio.TimeoutError:
                    print("⏰ 等待消息超时，可能对话已结束")
                    break
                except Exception as e:
                    print(f"❌ 接收消息错误: {e}")
                    break
            
            print(f"\n📊 测试完成，共接收 {message_count} 条消息")
            return message_count > 0
            
    except Exception as e:
        print(f"❌ WebSocket对话测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_websocket_conversation())
    if success:
        print("\n✅ WebSocket对话测试通过")
    else:
        print("\n❌ WebSocket对话测试失败")