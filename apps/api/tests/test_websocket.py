#!/usr/bin/env python3
"""
测试WebSocket实时消息推送
"""
import asyncio
import websockets
import json
import sys
import time

async def test_websocket_conversation():
    """测试WebSocket实时对话"""
    session_id = "test-session-001"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"连接WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送启动对话消息
            start_message = {
                "type": "start_conversation",
                "message": "测试WebSocket实时消息推送",
                "agent_roles": ["code_expert", "architect"],
                "max_rounds": 2
            }
            
            print(f"发送消息: {start_message}")
            await websocket.send(json.dumps(start_message))
            
            # 接收消息并打印
            message_count = 0
            max_messages = 10  # 最多接收10条消息
            timeout = 60  # 60秒超时
            
            start_time = time.time()
            
            while message_count < max_messages:
                try:
                    # 设置消息接收超时
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message_count += 1
                    
                    parsed = json.loads(message)
                    print(f"\n📨 消息 {message_count}: {parsed.get('type', 'unknown')}")
                    
                    if parsed.get('type') == 'conversation_created':
                        print(f"  会话已创建: {parsed.get('data', {}).get('conversation_id', 'N/A')}")
                    elif parsed.get('type') == 'new_message':
                        msg_data = parsed.get('data', {}).get('message', {})
                        print(f"  发送者: {msg_data.get('sender', 'N/A')}")
                        print(f"  内容: {msg_data.get('content', 'N/A')[:100]}...")
                    elif parsed.get('type') == 'speaker_change':
                        print(f"  当前发言者: {parsed.get('data', {}).get('current_speaker', 'N/A')}")
                    elif parsed.get('type') == 'conversation_completed':
                        print("  🎉 对话已完成")
                        break
                    elif parsed.get('type') == 'conversation_error':
                        print(f"  ❌ 对话错误: {parsed.get('data', {}).get('error', 'N/A')}")
                        break
                    
                    # 检查超时
                    if time.time() - start_time > timeout:
                        print(f"⏰ 测试超时 ({timeout}秒)")
                        break
                        
                except asyncio.TimeoutError:
                    print("⏰ 等待消息超时（5秒），继续等待...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 WebSocket连接已关闭")
                    break
                except Exception as e:
                    print(f"❌ 接收消息错误: {e}")
                    break
            
            print(f"\n📊 测试完成，共接收 {message_count} 条消息")
            
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")
        return False
    
    return True

async def main():
    """主测试函数"""
    print("=== WebSocket实时消息测试 ===")
    
    success = await test_websocket_conversation()
    
    if success:
        print("\n✅ WebSocket测试通过")
    else:
        print("\n❌ WebSocket测试失败")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ 用户中断测试")
    except Exception as e:
        print(f"\n💥 测试异常: {e}")