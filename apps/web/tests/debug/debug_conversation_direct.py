#!/usr/bin/env python3
"""
直接调用后端API触发对话，验证streaming token生成
"""
import asyncio
import websockets
import json
import requests
from datetime import datetime

async def test_conversation_trigger():
    """直接创建对话并监控WebSocket消息"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始测试对话触发...")
    
    # 1. 先创建对话通过REST API
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 创建对话...")
    conversation_data = {
        "message": "请简单介绍一下Python的异步编程",
        "agent_roles": ["code_expert"],
        "max_rounds": 2,
        "timeout_seconds": 300,
        "auto_reply": True
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/multi-agent/conversation",
            json=conversation_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            conversation_id = result["conversation_id"]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 对话创建成功: {conversation_id}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 参与者数量: {len(result['participants'])}")
            
            # 2. 连接WebSocket监控消息
            ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{conversation_id}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 连接WebSocket: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket连接已建立")
                
                # 监控消息30秒
                try:
                    await asyncio.wait_for(monitor_messages(websocket), timeout=30.0)
                except asyncio.TimeoutError:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 监控超时，结束测试")
                    
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 创建对话失败: {response.status_code}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 响应内容: {response.text}")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 测试异常: {e}")

async def monitor_messages(websocket):
    """监控WebSocket消息"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始监控WebSocket消息...")
    token_count = 0
    
    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            if msg_type == "streaming_token":
                token_count += 1
                token = data.get("data", {}).get("token", "")
                agent_name = data.get("data", {}).get("agent_name", "")
                print(f"[{timestamp}] 🎯 收到流式token #{token_count} - {agent_name}: '{token}'")
                
            elif msg_type == "connection_established":
                print(f"[{timestamp}] ✅ 连接已建立: {data.get('data', {}).get('session_id', '')}")
                
            elif msg_type == "conversation_started":
                print(f"[{timestamp}] 🚀 对话已启动")
                
            elif msg_type == "conversation_completed":
                print(f"[{timestamp}] ✅ 对话已完成，总token数: {token_count}")
                break
                
            elif msg_type == "speaker_change":
                speaker = data.get("data", {}).get("current_speaker", "")
                round_num = data.get("data", {}).get("round", "")
                print(f"[{timestamp}] 🎤 发言者切换: {speaker} (第{round_num}轮)")
                
            elif msg_type == "new_message":
                sender = data.get("data", {}).get("sender", "")
                content_preview = data.get("data", {}).get("content", "")[:50]
                print(f"[{timestamp}] 💬 新消息 - {sender}: {content_preview}...")
                
            elif msg_type == "error":
                error_msg = data.get("data", {}).get("message", "")
                print(f"[{timestamp}] ❌ 错误: {error_msg}")
                
            else:
                print(f"[{timestamp}] 📝 其他消息类型: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 无法解析消息: {message}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 处理消息异常: {e}")

if __name__ == "__main__":
    print("="*60)
    print("直接对话触发测试")
    print("="*60)
    asyncio.run(test_conversation_trigger())