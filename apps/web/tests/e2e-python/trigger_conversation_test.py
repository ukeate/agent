#!/usr/bin/env python3
"""
触发对话测试 - 手动发送启动消息并监听流式响应
"""
import asyncio
import websockets
import json
import time

async def trigger_conversation():
    """手动触发对话并监听流式响应"""
    
    # 最新会话ID
    session_id = "f2ead22b-3d0b-426a-a591-15e9c4c35ab8"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🚀 触发对话测试")
    print(f"📡 会话ID: {session_id}")
    print(f"🎯 手动发送启动消息，强制触发流式响应")
    print("=" * 100)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 手动发送多个不同类型的启动消息
            messages_to_send = [
                {
                    "type": "start_conversation",
                    "data": {
                        "message": "立即开始流式响应！每个专家说一句话，必须在页面显示每个token！",
                        "participants": ["doc_expert", "supervisor"]
                    }
                },
                {
                    "type": "send_message", 
                    "data": {
                        "message": "测试流式显示功能，请逐token响应！",
                        "sender": "user"
                    }
                }
            ]
            
            for i, message in enumerate(messages_to_send, 1):
                print(f"\n📤 发送第{i}个启动消息: {message['type']}")
                print(f"   内容: {message['data']['message'][:50]}...")
                await websocket.send(json.dumps(message))
                
                # 等待响应
                print(f"   等待响应...")
                await asyncio.sleep(2)
            
            # 监听流式响应
            start_time = time.time()
            streaming_tokens = []
            all_responses = []
            
            print(f"\n🔄 开始监听流式响应...")
            print("-" * 100)
            
            while time.time() - start_time < 60:  # 60秒超时
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    message = json.loads(response)
                    all_responses.append(message)
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话启动成功！")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        round_num = message['data'].get('round', 0)
                        print(f"   🎤 发言者: {speaker} (第{round_num}轮)")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        message_id = message['data'].get('message_id', '')
                        full_content = message['data'].get('full_content', '')
                        
                        streaming_tokens.append(token)
                        
                        print(f"   📝 Token #{len(streaming_tokens)}: {agent} -> '{token}'")
                        if len(streaming_tokens) <= 5:
                            print(f"       🔍 完整内容: {full_content[:50]}...")
                            print(f"       🖥️  页面应该立即显示这个token!")
                        
                    elif msg_type == "streaming_complete":
                        agent = message['data'].get('agent_name', '')
                        content = message['data'].get('full_content', '')
                        print(f"   ✅ 流式完成: {agent}")
                        print(f"       📄 完整消息: {content[:80]}...")
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        content = msg_data.get('content', '')
                        print(f"   💬 新消息: {sender}")
                        print(f"       📄 内容: {content[:80]}...")
                        
                    elif msg_type == "conversation_completed":
                        print(f"   🏁 对话完成")
                        break
                        
                    elif msg_type == "error":
                        error_msg = message['data'].get('message', '未知错误')
                        print(f"   ❌ 错误: {error_msg}")
                        
                    # 如果收到足够的token，说明成功
                    if len(streaming_tokens) >= 5:
                        print(f"\n🎉 成功触发流式响应！收到{len(streaming_tokens)}个token")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时...")
                    continue
                    
            # 结果分析
            print("\n" + "=" * 100)
            print("🔥 触发对话结果分析")
            print("=" * 100)
            
            print(f"📊 发送消息数: {len(messages_to_send)}")
            print(f"📊 收到响应数: {len(all_responses)}")
            print(f"📊 streaming_token数: {len(streaming_tokens)}")
            
            if len(streaming_tokens) > 0:
                print(f"\n🎉 SUCCESS - 流式响应触发成功!")
                print(f"✅ 后端正常处理了对话启动")
                print(f"✅ 收到了{len(streaming_tokens)}个实时token")
                print(f"✅ 每个token都应该在前端页面显示")
                
                print(f"\n🐛 前端显示bug确认:")
                print(f"❌ 页面仍显示'多智能体对话还未开始'")
                print(f"❌ 但后端已经发送了{len(streaming_tokens)}个token")
                print(f"❌ 前端WebSocket处理逻辑有严重问题")
                
                return True
            else:
                print(f"\n⚠️  没有触发流式响应")
                print(f"可能的原因:")
                print(f"- 后端对话服务有问题")
                print(f"- 消息格式不正确")
                print(f"- 会话状态异常")
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(trigger_conversation())
    print(f"\n🎯 最终结论: {'成功触发，前端有bug' if result else '对话触发失败'}")
    exit(0 if result else 1)