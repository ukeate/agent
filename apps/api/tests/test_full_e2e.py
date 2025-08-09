#!/usr/bin/env python3
"""
完整端到端测试：REST API + WebSocket 实时消息推送
"""
import asyncio
import websockets
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor

async def test_websocket_real_time(session_id: str):
    """测试WebSocket实时消息接收"""
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🔗 连接WebSocket: {ws_url}")
    
    messages_received = []
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送启动对话消息
            start_message = {
                "type": "start_conversation",
                "message": "完整E2E测试：请分析React前端架构",
                "agent_roles": ["code_expert", "architect"],
                "max_rounds": 1
            }
            
            print(f"📤 发送启动消息: {start_message['message']}")
            await websocket.send(json.dumps(start_message))
            
            # 接收消息
            start_time = time.time()
            while time.time() - start_time < 45:  # 最多等待45秒
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    parsed = json.loads(message)
                    msg_type = parsed.get('type', 'unknown')
                    messages_received.append(parsed)
                    
                    print(f"📨 收到消息: {msg_type}")
                    
                    if msg_type == 'new_message':
                        msg_data = parsed.get('data', {}).get('message', {})
                        sender = msg_data.get('sender', 'N/A')
                        content = msg_data.get('content', 'N/A')[:200]
                        print(f"  💬 {sender}: {content}...")
                    elif msg_type == 'conversation_completed':
                        print("  🎉 对话完成")
                        break
                    elif msg_type == 'conversation_error':
                        error = parsed.get('data', {}).get('error', 'N/A')
                        print(f"  ❌ 对话错误: {error}")
                        break
                        
                except asyncio.TimeoutError:
                    print("⏰ 等待消息超时")
                    break
                except Exception as e:
                    print(f"❌ 接收消息错误: {e}")
                    break
            
            print(f"📊 共接收 {len(messages_received)} 条消息")
            return messages_received
            
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")
        return []

def test_rest_api():
    """测试REST API对话创建"""
    url = "http://localhost:8000/api/v1/multi-agent/conversation"
    
    payload = {
        "message": "完整E2E测试：请分析React前端架构",
        "agent_roles": ["code_expert", "architect"], 
        "max_rounds": 1,
        "timeout_seconds": 60
    }
    
    print(f"🌐 发送REST API请求到: {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ REST API创建成功")
            print(f"  会话ID: {result['conversation_id']}")
            print(f"  状态: {result['status']}")
            print(f"  参与者: {[p['name'] for p in result['participants']]}")
            return result
        else:
            print(f"❌ REST API失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ REST API请求失败: {e}")
        return None

async def main():
    """主测试流程"""
    print("🚀 开始完整端到端测试")
    print("=" * 50)
    
    # 测试1: REST API对话创建
    print("\n1️⃣ 测试REST API对话创建")
    rest_result = test_rest_api()
    
    if not rest_result:
        print("❌ REST API测试失败，终止测试")
        return
    
    session_id = rest_result['conversation_id']
    
    # 等待一下让对话开始
    print("⏳ 等待对话初始化...")
    await asyncio.sleep(2)
    
    # 测试2: WebSocket实时消息接收
    print("\n2️⃣ 测试WebSocket实时消息接收")
    ws_messages = await test_websocket_real_time(session_id)
    
    # 分析结果
    print("\n" + "=" * 50)
    print("📋 测试结果分析")
    
    message_types = [msg.get('type') for msg in ws_messages]
    new_message_count = message_types.count('new_message')
    
    print(f"✨ 总消息数: {len(ws_messages)}")
    print(f"💬 智能体响应消息数: {new_message_count}")
    print(f"📝 消息类型分布: {dict((t, message_types.count(t)) for t in set(message_types))}")
    
    # 判断测试是否成功
    success = (
        len(ws_messages) >= 4 and  # 至少有基本的4条消息
        new_message_count >= 1 and  # 至少有1条智能体响应
        'conversation_created' in message_types and
        'conversation_started' in message_types
    )
    
    if success:
        print("\n🎉 完整端到端测试成功！")
        print("✅ REST API对话创建 ✓")
        print("✅ WebSocket实时连接 ✓") 
        print("✅ 智能体响应生成 ✓")
        print("✅ 实时消息推送 ✓")
    else:
        print("\n❌ 完整端到端测试失败")
        print(f"💔 问题分析: 消息数={len(ws_messages)}, 智能体响应数={new_message_count}")

if __name__ == "__main__":
    asyncio.run(main())