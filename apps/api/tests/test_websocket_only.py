#!/usr/bin/env python3
"""
仅通过WebSocket测试完整对话流程
"""
import asyncio
import websockets
import json
import time

async def test_websocket_only_conversation():
    """仅通过WebSocket测试对话功能"""
    session_id = "websocket-only-test-001"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🔗 连接WebSocket: {ws_url}")
    
    messages_received = []
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送启动对话消息
            start_message = {
                "type": "start_conversation",
                "message": "WebSocket直接测试：请帮我分析React前端架构",
                "agent_roles": ["code_expert", "architect"],
                "max_rounds": 1
            }
            
            print(f"📤 发送启动消息: {start_message['message']}")
            await websocket.send(json.dumps(start_message))
            
            # 接收消息
            start_time = time.time()
            while time.time() - start_time < 60:  # 最多等待60秒
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                    parsed = json.loads(message)
                    msg_type = parsed.get('type', 'unknown')
                    messages_received.append(parsed)
                    
                    print(f"📨 收到消息: {msg_type}")
                    
                    if msg_type == 'conversation_created':
                        print(f"  ✅ 对话已创建")
                        conv_id = parsed.get('data', {}).get('conversation_id', 'N/A')
                        print(f"  会话ID: {conv_id}")
                    elif msg_type == 'conversation_started':
                        print(f"  ✅ 对话已启动")
                    elif msg_type == 'new_message':
                        msg_data = parsed.get('data', {}).get('message', {})
                        sender = msg_data.get('sender', 'N/A')
                        content = msg_data.get('content', 'N/A')[:200]
                        print(f"  💬 {sender}: {content}...")
                    elif msg_type == 'speaker_change':
                        speaker = parsed.get('data', {}).get('current_speaker', 'N/A')
                        round_num = parsed.get('data', {}).get('round', 'N/A')
                        print(f"  🗣️ 发言者变更: {speaker} (第{round_num}轮)")
                    elif msg_type == 'conversation_completed':
                        print(f"  🎉 对话完成")
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
                        
                except asyncio.TimeoutError:
                    print("⏰ 等待消息超时")
                    break
                except Exception as e:
                    print(f"❌ 接收消息错误: {e}")
                    break
            
            print(f"📊 共接收 {len(messages_received)} 条消息")
            
            # 分析结果
            message_types = [msg.get('type') for msg in messages_received]
            new_message_count = message_types.count('new_message')
            
            print(f"✨ 总消息数: {len(messages_received)}")
            print(f"💬 智能体响应消息数: {new_message_count}")
            print(f"📝 消息类型分布: {dict((t, message_types.count(t)) for t in set(message_types))}")
            
            # 判断测试是否成功
            success = (
                len(messages_received) >= 4 and  # 至少有基本的4条消息
                new_message_count >= 1 and  # 至少有1条智能体响应
                'conversation_created' in message_types and
                'conversation_started' in message_types
            )
            
            if success:
                print("\n🎉 WebSocket直接对话测试成功！")
                return True
            else:
                print("\n❌ WebSocket直接对话测试失败")
                return False
            
    except Exception as e:
        print(f"❌ WebSocket对话测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_websocket_only_conversation())
    if success:
        print("\n✅ 测试通过 - WebSocket实时对话功能正常")
    else:
        print("\n❌ 测试失败 - WebSocket实时对话功能异常")