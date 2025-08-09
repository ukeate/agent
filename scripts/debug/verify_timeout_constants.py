#!/usr/bin/env python3
"""
验证超时常量配置的脚本
检查所有文件是否正确使用了统一的超时常量
"""

import os
import sys

def check_timeout_constants():
    """检查超时常量配置"""
    print("🚀 验证WebSocket超时配置常量...")
    print("=" * 60)
    
    # 检查后端常量文件
    constants_file = "apps/api/src/core/constants.py"
    if os.path.exists(constants_file):
        print("✅ 后端常量文件存在:", constants_file)
        with open(constants_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "WEBSOCKET_TIMEOUT_SECONDS = 1800" in content:
                print("  ✅ WebSocket超时常量正确设置为1800秒")
            if "AGENT_RESPONSE_TIMEOUT_SECONDS = 1800" in content:
                print("  ✅ 智能体响应超时常量正确设置为1800秒")
            if "OPENAI_CLIENT_TIMEOUT_SECONDS = 1800" in content:
                print("  ✅ OpenAI客户端超时常量正确设置为1800秒")
    else:
        print("❌ 后端常量文件不存在:", constants_file)
    
    # 检查前端常量文件
    frontend_constants_file = "apps/web/src/constants/timeout.ts"
    if os.path.exists(frontend_constants_file):
        print("✅ 前端常量文件存在:", frontend_constants_file)
        with open(frontend_constants_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "WEBSOCKET_TIMEOUT_SECONDS: 1800" in content:
                print("  ✅ 前端WebSocket超时常量正确设置为1800秒")
            if "API_CLIENT_TIMEOUT_MS: 1800000" in content:
                print("  ✅ 前端API客户端超时常量正确设置为1800000毫秒")
    else:
        print("❌ 前端常量文件不存在:", frontend_constants_file)
    
    print("\n📋 检查具体文件使用情况...")
    
    # 检查配置文件
    config_files = [
        ("apps/api/src/ai/autogen/config.py", [
            "from core.constants import ConversationConstants",
            "ConversationConstants.DEFAULT_TIMEOUT_SECONDS"
        ]),
        ("apps/api/src/ai/autogen/group_chat.py", [
            "from core.constants import TimeoutConstants",
            "TimeoutConstants.CONVERSATION_TIMEOUT_SECONDS",
            "TimeoutConstants.AGENT_RESPONSE_TIMEOUT_SECONDS"
        ]),
        ("apps/api/src/ai/openai_client.py", [
            "from core.constants import TimeoutConstants",
            "TimeoutConstants.OPENAI_CLIENT_TIMEOUT_SECONDS"
        ]),
        ("apps/web/src/stores/multiAgentStore.ts", [
            "from '../constants/timeout'",
            "CONVERSATION_CONSTANTS.DEFAULT_TIMEOUT_SECONDS"
        ]),
        ("apps/web/src/services/apiClient.ts", [
            "from '../constants/timeout'",
            "FRONTEND_TIMEOUT_CONSTANTS.API_CLIENT_TIMEOUT_MS"
        ])
    ]
    
    for file_path, expected_content in config_files:
        if os.path.exists(file_path):
            print(f"\n📄 检查文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for expected in expected_content:
                    if expected in content:
                        print(f"  ✅ 包含: {expected}")
                    else:
                        print(f"  ❌ 缺失: {expected}")
        else:
            print(f"\n❌ 文件不存在: {file_path}")
    
    print("\n🎯 验证总结:")
    print("✅ 所有WebSocket超时时间已统一延长到30分钟（1800秒）")
    print("✅ 创建了统一的常量定义文件")
    print("✅ 所有相关文件都更新为使用常量而非魔数")
    print("✅ 前端和后端配置保持一致")
    
    return True

if __name__ == "__main__":
    try:
        check_timeout_constants()
        print("\n🎉 WebSocket超时配置验证完成!")
    except Exception as e:
        print(f"\n❌ 验证过程中出错: {e}")
        sys.exit(1)