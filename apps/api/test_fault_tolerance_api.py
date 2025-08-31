#!/usr/bin/env python3
"""
测试故障容错API功能的简单脚本
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from fastapi.testclient import TestClient

def test_fault_tolerance_api():
    """测试故障容错API基本功能"""
    try:
        # 导入依赖
        from core.dependencies import initialize_fault_tolerance_system
        from api.v1.fault_tolerance import router
        from fastapi import FastAPI
        
        # 初始化故障容错系统
        initialize_fault_tolerance_system()
        print("✓ 故障容错系统初始化成功")
        
        # 创建测试应用
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        print("✓ 故障容错路由注册成功")
        
        # 创建测试客户端
        client = TestClient(app)
        
        # 测试系统状态API
        print("\n测试系统状态API...")
        response = client.get("/api/v1/fault-tolerance/status")
        if response.status_code == 200:
            print("✓ /fault-tolerance/status API 正常")
            data = response.json()
            print(f"  系统状态: {data.get('status', 'unknown')}")
        else:
            print(f"✗ /fault-tolerance/status API 失败: {response.status_code}")
            print(f"  响应: {response.text}")
        
        # 测试恢复统计API
        print("\n测试恢复统计API...")
        response = client.get("/api/v1/fault-tolerance/recovery/statistics")
        if response.status_code == 200:
            print("✓ /fault-tolerance/recovery/statistics API 正常")
            data = response.json()
            print(f"  成功率: {data.get('success_rate', 'unknown')}")
        else:
            print(f"✗ /fault-tolerance/recovery/statistics API 失败: {response.status_code}")
            print(f"  响应: {response.text}")
        
        # 测试备份统计API
        print("\n测试备份统计API...")
        response = client.get("/api/v1/fault-tolerance/backup/statistics")
        if response.status_code == 200:
            print("✓ /fault-tolerance/backup/statistics API 正常")
            data = response.json()
            print(f"  成功数量: {data.get('successful_backups', 'unknown')}")
        else:
            print(f"✗ /fault-tolerance/backup/statistics API 失败: {response.status_code}")
            print(f"  响应: {response.text}")
        
        # 测试一致性统计API
        print("\n测试一致性统计API...")
        response = client.get("/api/v1/fault-tolerance/consistency/statistics")
        if response.status_code == 200:
            print("✓ /fault-tolerance/consistency/statistics API 正常")
            data = response.json()
            print(f"  一致性率: {data.get('consistency_rate', 'unknown')}")
        else:
            print(f"✗ /fault-tolerance/consistency/statistics API 失败: {response.status_code}")
            print(f"  响应: {response.text}")
        
        print("\n✓ 故障容错API测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 故障容错API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fault_tolerance_api()
    sys.exit(0 if success else 1)