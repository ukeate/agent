#!/usr/bin/env python3
"""
测试当前Playwright会话的streaming_token
"""
import asyncio
import websockets
import json
import time

async def test_current_session():
    """测试当前前端创建的会话"""
    
    # 从Playwright页面获取的当前会话ID
    session_id = "ef4eb330-bdb3-42d8-aafb-fe1e42e668f7"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🔍 测试当前会话streaming_token")
    print(f"📡 会话ID: {session_id}")
    print(f"🎯 检查后端是否发送streaming_token给前端")
    print("=" * 100)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 立即监听所有消息
            start_time = time.time()
            all_messages = []
            streaming_tokens = []
            
            print("\n🔄 监听当前会话的所有消息...")
            print("-" * 100)
            
            while time.time() - start_time < 15:  # 15秒快速测试
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    message = json.loads(response)
                    all_messages.append(message)
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话已启动！")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        print(f"   🎤 发言者: {speaker}")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        
                        streaming_tokens.append(token)
                        print(f"   📝 Token #{len(streaming_tokens)}: {agent} -> '{token}'")
                        print(f"       ✅ 后端正在发送streaming_token!")
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        print(f"   💬 新消息: {sender}")
                        
                    elif msg_type == "error":
                        error_msg = message['data'].get('message', '未知错误')
                        print(f"   ❌ 错误: {error_msg}")
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时")
                    break
                    
            # 分析结果
            print("\n" + "=" * 100)
            print("🔥 当前会话分析结果")
            print("=" * 100)
            
            print(f"📊 总消息数: {len(all_messages)}")
            print(f"📊 streaming_token数: {len(streaming_tokens)}")
            
            if len(streaming_tokens) > 0:
                print(f"\n✅ 后端正在发送streaming_token!")
                print(f"   - 收到{len(streaming_tokens)}个token")
                print(f"   - 前端应该显示这些token")
                print(f"\n❌ 前端显示bug确认:")
                print(f"   - WebSocket接收正常")
                print(f"   - 但页面仍显示'多智能体对话还未开始'")
                print(f"   - addStreamingToken函数有问题")
                return True
            else:
                print(f"\n⚠️  后端没有发送streaming_token")
                print(f"可能原因:")
                print(f"   - 对话没有真正启动")
                print(f"   - 后端处理有问题")
                print(f"   - 需要手动触发对话")
                
                # 如果没有streaming_token，尝试手动启动
                print(f"\n🔄 尝试手动启动对话...")
                trigger_msg = {
                    "type": "start_conversation",
                    "data": {
                        "message": "立即开始！请代码专家说话，必须逐token显示！",
                        "participants": ["code_expert"]
                    }
                }
                await websocket.send(json.dumps(trigger_msg))
                
                # 再等5秒
                print(f"   等待手动触发结果...")
                manual_tokens = []
                start_manual = time.time()
                
                while time.time() - start_manual < 5:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        message = json.loads(response)
                        msg_type = message.get("type")
                        
                        if msg_type == "streaming_token":
                            token = message['data'].get('token', '')
                            manual_tokens.append(token)
                            print(f"   📝 手动触发Token #{len(manual_tokens)}: '{token}'")
                            
                    except asyncio.TimeoutError:
                        break
                        
                if len(manual_tokens) > 0:
                    print(f"\n✅ 手动触发成功！收到{len(manual_tokens)}个token")
                    return True
                else:
                    print(f"\n❌ 手动触发也失败")
                    return False
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_current_session())
    print(f"\n🎯 测试结果: {'成功' if result else '失败'}")
    exit(0 if result else 1)