#!/usr/bin/env python3
"""
测试双WebSocket连接 - 验证后端消息路由问题
"""
import asyncio
import websockets
import json
import time

async def test_dual_connections():
    """测试两个WebSocket连接同时接收消息"""
    
    session_id = "ef4eb330-bdb3-42d8-aafb-fe1e42e668f7"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🔍 测试双WebSocket连接消息路由")
    print(f"📡 会话ID: {session_id}")
    print(f"🎯 验证后端是否向所有连接广播消息")
    print("=" * 100)
    
    # 连接1和连接2的消息接收计录
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
        while time.time() - start_time < 10:  # 10秒测试
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                message = json.loads(response)
                msg_type = message.get("type")
                
                if msg_type == "streaming_token":
                    token = message['data'].get('token', '')
                    token_list.append(token)
                    print(f"📝 连接{connection_id} Token #{len(token_list)}: '{token}'")
                    
                elif msg_type in ["conversation_started", "speaker_change", "new_message"]:
                    print(f"📨 连接{connection_id} {msg_type}")
                    
            except asyncio.TimeoutError:
                break
                
    try:
        # 建立两个并发连接
        async with websockets.connect(ws_url) as ws1:
            async with websockets.connect(ws_url) as ws2:
                
                # 等待一秒让连接稳定
                await asyncio.sleep(1)
                
                # 从连接1发送启动消息
                trigger_msg = {
                    "type": "start_conversation", 
                    "data": {
                        "message": "双连接测试！检查消息路由！",
                        "participants": ["code_expert"]
                    }
                }
                
                print(f"\n📤 从连接1发送启动消息...")
                await ws1.send(json.dumps(trigger_msg))
                
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
    print("🔥 双连接消息路由分析")
    print("=" * 100)
    
    print(f"📊 连接1收到token数: {len(conn1_tokens)}")
    print(f"📊 连接2收到token数: {len(conn2_tokens)}")
    
    if len(conn1_tokens) > 0 and len(conn2_tokens) > 0:
        print(f"\n✅ 后端正确广播消息到两个连接")
        print(f"   - 连接1收到{len(conn1_tokens)}个token")
        print(f"   - 连接2收到{len(conn2_tokens)}个token")
        print(f"   - 前端WebSocket处理逻辑有其他bug")
        return True
        
    elif len(conn1_tokens) > 0 and len(conn2_tokens) == 0:
        print(f"\n❌ 后端只向第一个连接发送消息!")
        print(f"   - 连接1(发送者)收到{len(conn1_tokens)}个token")
        print(f"   - 连接2(监听者)收到0个token")
        print(f"   - 这解释了为什么前端收不到消息")
        print(f"\n🐛 根本原因: 后端消息路由只发送给触发连接")
        return False
        
    elif len(conn1_tokens) == 0 and len(conn2_tokens) > 0:
        print(f"\n🤔 奇怪：只有连接2收到消息")
        return False
        
    else:
        print(f"\n⚠️  两个连接都没收到token，可能对话没启动")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_dual_connections())
    print(f"\n🎯 结论: {'后端广播正常' if result else '后端消息路由有bug'}")
    exit(0 if result else 1)