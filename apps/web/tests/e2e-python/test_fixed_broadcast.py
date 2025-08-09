#!/usr/bin/env python3
"""
测试修复后的WebSocket广播功能
"""
import asyncio
import websockets
import json
import time
import uuid

async def test_fixed_broadcast():
    """测试修复后的双连接广播"""
    
    # 生成新的会话ID测试
    session_id = str(uuid.uuid4())
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🔧 测试修复后的WebSocket广播功能")
    print(f"📡 新会话ID: {session_id}")
    print(f"🎯 验证两个连接是否都能收到streaming_token")
    print("=" * 100)
    
    # 连接1和连接2的消息接收记录
    conn1_tokens = []
    conn2_tokens = []
    
    async def handle_connection(connection_id, websocket, token_list):
        """处理单个WebSocket连接"""
        print(f"✅ 连接{connection_id}建立成功")
        
        # 接收连接确认
        response = await websocket.recv()
        msg = json.loads(response)
        print(f"📨 连接{connection_id}确认: {msg.get('type')}")
        
        # 监听消息
        start_time = time.time()
        while time.time() - start_time < 15:  # 15秒测试
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                message = json.loads(response)
                msg_type = message.get("type")
                
                if msg_type == "streaming_token":
                    token = message['data'].get('token', '')
                    token_list.append(token)
                    print(f"📝 连接{connection_id} Token #{len(token_list)}: '{token}'")
                    
                    # 前5个token显示详细信息
                    if len(token_list) <= 5:
                        print(f"    ✅ 连接{connection_id}成功接收streaming_token!")
                    
                elif msg_type in ["conversation_created", "conversation_started", "speaker_change", "new_message"]:
                    print(f"📨 连接{connection_id} {msg_type}")
                    
            except asyncio.TimeoutError:
                if len(token_list) > 0:
                    break  # 如果已经收到token就停止等待
                continue
                
    try:
        # 建立两个并发连接
        async with websockets.connect(ws_url) as ws1:
            async with websockets.connect(ws_url) as ws2:
                
                print(f"\n🚀 两个连接已建立，发送启动消息...")
                
                # 从连接1发送启动消息
                trigger_msg = {
                    "type": "start_conversation", 
                    "data": {
                        "message": "修复测试！两个连接都应该收到streaming_token！",
                        "participants": ["code_expert"]
                    }
                }
                
                await ws1.send(json.dumps(trigger_msg))
                print(f"📤 启动消息已发送")
                
                # 并发处理两个连接
                await asyncio.gather(
                    handle_connection(1, ws1, conn1_tokens),
                    handle_connection(2, ws2, conn2_tokens)
                )
                
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False
        
    # 分析结果
    print("\n" + "=" * 100)
    print("🔥 修复后的广播功能测试结果")
    print("=" * 100)
    
    print(f"📊 连接1收到token数: {len(conn1_tokens)}")
    print(f"📊 连接2收到token数: {len(conn2_tokens)}")
    
    if len(conn1_tokens) > 0 and len(conn2_tokens) > 0:
        print(f"\n🎉 修复成功！后端现在正确广播到两个连接")
        print(f"   ✅ 连接1收到{len(conn1_tokens)}个token")
        print(f"   ✅ 连接2收到{len(conn2_tokens)}个token")
        print(f"   ✅ 前端WebSocket现在应该能收到streaming_token")
        
        # 检查token是否相同（应该一致）
        min_tokens = min(len(conn1_tokens), len(conn2_tokens))
        if min_tokens > 0:
            tokens_match = conn1_tokens[:min_tokens] == conn2_tokens[:min_tokens] 
            print(f"   ✅ Token内容一致性: {'完全一致' if tokens_match else '有差异'}")
        
        return True
        
    elif len(conn1_tokens) > 0 or len(conn2_tokens) > 0:
        print(f"\n⚠️  部分修复：只有一个连接收到消息")
        print(f"   - 连接1: {len(conn1_tokens)}个token")
        print(f"   - 连接2: {len(conn2_tokens)}个token")
        print(f"   - 可能还有其他问题需要解决")
        return False
        
    else:
        print(f"\n❌ 修复失败：两个连接都没收到token")
        print(f"   - 可能对话没有启动")
        print(f"   - 或者还有其他问题")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_fixed_broadcast())
    print(f"\n🎯 修复结果: {'成功' if result else '失败'}")
    exit(0 if result else 1)