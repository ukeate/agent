#!/usr/bin/env python3
"""
调试消息问题 - 验证WebSocket消息是否正确添加到前端store
"""
import asyncio
import websockets
import json
import time

async def debug_message_issue():
    """调试为什么WebSocket消息没有添加到currentMessages数组"""
    
    # 最新会话ID
    session_id = "7526dea6-819c-4878-96a6-6d6b2bbe1c66"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🐛 调试消息问题")
    print(f"📡 会话ID: {session_id}")
    print(f"🔍 问题: WebSocket收到消息但前端currentMessages为空")
    print("=" * 100)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 立即发送新消息触发流式响应
            trigger_message = {
                "type": "start_conversation",
                "data": {
                    "message": "立即开始流式响应测试！请每个专家说一句话，必须在页面显示！",
                    "participants": ["doc_expert", "supervisor"]
                }
            }
            
            print(f"📤 发送触发消息...")
            await websocket.send(json.dumps(trigger_message))
            
            # 监听消息详细信息
            start_time = time.time()
            streaming_tokens = []
            new_messages = []
            speaker_changes = []
            
            print("\n🔄 监听并分析所有WebSocket消息...")
            print("-" * 100)
            
            while time.time() - start_time < 60:  # 60秒超时
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    message = json.loads(response)
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话启动 - 前端应开始显示消息")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        round_num = message['data'].get('round', 0)
                        speaker_changes.append({
                            'speaker': speaker, 
                            'round': round_num,
                            'timestamp': timestamp
                        })
                        print(f"   🎤 发言者: {speaker} (第{round_num}轮)")
                        print(f"       ➤ 前端应调用setCurrentSpeaker('{speaker}')")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        message_id = message['data'].get('message_id', '')
                        full_content = message['data'].get('full_content', '')
                        
                        streaming_tokens.append({
                            'message_id': message_id,
                            'agent': agent,
                            'token': token,
                            'full_content': full_content,
                            'timestamp': timestamp
                        })
                        
                        print(f"   📝 Token #{len(streaming_tokens)}: {agent} -> '{token}'")
                        print(f"       ➤ message_id: {message_id}")
                        print(f"       ➤ full_content: {full_content[:30]}...")
                        print(f"       ➤ 前端应调用addStreamingToken('{message_id}', {{...}})")
                        
                    elif msg_type == "streaming_complete":
                        agent = message['data'].get('agent_name', '')
                        message_id = message['data'].get('message_id', '')
                        full_content = message['data'].get('full_content', '')
                        print(f"   ✅ 流式完成: {agent}")
                        print(f"       ➤ message_id: {message_id}")
                        print(f"       ➤ 前端应调用completeStreamingMessage('{message_id}', {{...}})")
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        content = msg_data.get('content', '')
                        msg_id = msg_data.get('id', '')
                        
                        new_messages.append({
                            'id': msg_id,
                            'sender': sender,
                            'content': content,
                            'timestamp': timestamp
                        })
                        
                        print(f"   💬 新消息: {sender}")
                        print(f"       ➤ id: {msg_id}")
                        print(f"       ➤ content: {content[:50]}...")
                        print(f"       ➤ 前端应调用addMessage({{...}})")
                        
                    # 如果收到足够的消息，分析问题
                    if len(streaming_tokens) >= 3 or len(new_messages) >= 1:
                        print(f"\n🔍 已收到足够消息，分析前端显示问题...")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时...")
                    break
                    
            # 问题分析
            print("\n" + "=" * 100)
            print("🐛 前端消息显示问题分析")
            print("=" * 100)
            
            print(f"📊 收到的streaming_token数: {len(streaming_tokens)}")
            print(f"📊 收到的new_message数: {len(new_messages)}")
            print(f"📊 收到的speaker_change数: {len(speaker_changes)}")
            
            if len(streaming_tokens) > 0:
                print(f"\n📝 streaming_token消息详情:")
                for i, token in enumerate(streaming_tokens[:5], 1):
                    print(f"  {i}. ID:{token['message_id'][:8]}... Agent:{token['agent']} Token:'{token['token']}'")
                    
            if len(new_messages) > 0:
                print(f"\n💬 new_message消息详情:")
                for i, msg in enumerate(new_messages, 1):
                    print(f"  {i}. ID:{msg['id'][:8]}... Sender:{msg['sender']}")
                    
            # 潜在问题识别
            print(f"\n🔍 潜在问题识别:")
            if len(streaming_tokens) > 0:
                print(f"✅ 后端正常发送streaming_token消息")
                print(f"❌ 前端addStreamingToken可能没有正确执行")
                print(f"🐛 可能原因:")
                print(f"   1. addStreamingToken函数有bug")
                print(f"   2. message_id格式问题")
                print(f"   3. store状态更新失败")
                print(f"   4. React组件没有订阅currentMessages变化")
                
                # 检查message_id格式
                message_ids = [t['message_id'] for t in streaming_tokens]
                print(f"\n🔍 message_id格式检查:")
                for mid in set(message_ids):
                    print(f"   - {mid}")
                    
            else:
                print(f"❌ 后端没有发送streaming_token消息")
                print(f"🐛 可能后端对话没有启动")
                
            return len(streaming_tokens) > 0 or len(new_messages) > 0
                
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_message_issue())
    exit(0 if result else 1)