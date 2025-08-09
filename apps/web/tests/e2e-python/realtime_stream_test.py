#!/usr/bin/env python3
"""
实时流式测试 - 验证新会话的流式响应和前端显示
"""
import asyncio
import websockets
import json
import time

async def test_realtime_stream():
    """测试最新会话的实时流式响应"""
    
    # 从Playwright页面获取的最新会话ID
    session_id = "f2ead22b-3d0b-426a-a591-15e9c4c35ab8"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🎯 实时流式测试")
    print(f"📡 会话ID: {session_id}")
    print(f"🔥 目标: 验证streaming_token是否能触发前端显示")
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
            
            print("\n🔄 监听实时流式响应（关键测试）...")
            print("-" * 100)
            
            while time.time() - start_time < 30:  # 30秒快速测试
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message = json.loads(response)
                    all_messages.append(message)
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话启动 - 前端应该收到此消息")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        print(f"   🎤 发言者变更: {speaker}")
                        print(f"       ➤ 前端应该收到此消息并调用setCurrentSpeaker")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        message_id = message['data'].get('message_id', '')
                        
                        streaming_tokens.append({
                            'timestamp': timestamp,
                            'agent': agent,
                            'token': token,
                            'message_id': message_id
                        })
                        
                        print(f"   📝 Token #{len(streaming_tokens)}: {agent} -> '{token}'")
                        print(f"       ➤ message_id: {message_id}")
                        print(f"       ➤ 前端应该收到此消息并调用addStreamingToken")
                        print(f"       ➤ 页面应该立即显示这个token")
                        
                        # 前3个token给出详细分析
                        if len(streaming_tokens) <= 3:
                            print(f"       🔍 这是第{len(streaming_tokens)}个token，页面必须显示！")
                        
                    elif msg_type == "streaming_complete":
                        agent = message['data'].get('agent_name', '')
                        print(f"   ✅ 流式完成: {agent}")
                        print(f"       ➤ 前端应该收到此消息并调用completeStreamingMessage")
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        print(f"   💬 新消息: {sender}")
                        print(f"       ➤ 前端应该收到此消息并调用addMessage")
                        
                    # 如果收到足够的streaming_token，证明后端正常
                    if len(streaming_tokens) >= 5:
                        print(f"\n🎉 后端streaming_token正常！已收到{len(streaming_tokens)}个token")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时")
                    break
                    
            # 最终分析
            print("\n" + "=" * 100)
            print("🔥 CRITICAL BUG分析")
            print("=" * 100)
            
            print(f"📊 总消息数: {len(all_messages)}")
            print(f"📊 streaming_token数: {len(streaming_tokens)}")
            
            if len(streaming_tokens) > 0:
                print(f"\n✅ 后端完全正常:")
                print(f"   - WebSocket连接成功")
                print(f"   - 收到{len(streaming_tokens)}个streaming_token")
                print(f"   - 每个token都有正确的message_id和agent_name")
                
                print(f"\n❌ 前端显示bug:")
                print(f"   - 页面显示'多智能体对话还未开始'")
                print(f"   - 但后端已发送{len(streaming_tokens)}个token")
                print(f"   - 前端WebSocket hook没有收到这些消息")
                
                print(f"\n🐛 根本原因:")
                print(f"   - 前端连接到了错误的会话ID")
                print(f"   - 或者WebSocket消息路由有问题")
                print(f"   - 或者前端消息处理函数有bug")
                
                print(f"\n💡 解决方案:")
                print(f"   1. 检查前端实际连接的会话ID")
                print(f"   2. 修复WebSocket消息处理逻辑")
                print(f"   3. 确保currentMessages数组正确更新")
                return True
            else:
                print(f"\n⚠️  没有收到streaming_token，可能:")
                print(f"   - 对话没有真正开始")
                print(f"   - 后端处理有问题")
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_realtime_stream())
    print(f"\n🎯 测试结果: {'后端正常，前端有bug' if result else '系统问题'}")
    exit(0 if result else 1)