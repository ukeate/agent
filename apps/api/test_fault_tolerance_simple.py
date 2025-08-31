#!/usr/bin/env python3
"""
简单的故障容错API测试脚本
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fault_tolerance_system():
    """测试故障容错系统基本功能"""
    try:
        # 测试依赖项初始化
        from core.dependencies import initialize_fault_tolerance_system, get_fault_tolerance_system
        
        # 初始化
        initialize_fault_tolerance_system()
        print("✓ 故障容错系统初始化成功")
        
        # 测试获取系统实例
        import asyncio
        
        async def test_system():
            system = await get_fault_tolerance_system()
            print("✓ 故障容错系统实例获取成功")
            
            # 测试基本方法
            try:
                status = await system.get_system_status()
                print("✓ 系统状态获取成功")
                print(f"  状态: {status}")
            except Exception as e:
                print(f"⚠ 系统状态获取有问题: {e}")
            
            try:
                metrics = await system.get_system_metrics()
                print("✓ 系统指标获取成功")
                print(f"  指标: {metrics}")
            except Exception as e:
                print(f"⚠ 系统指标获取有问题: {e}")
        
        # 运行异步测试
        asyncio.run(test_system())
        
        print("\n✓ 故障容错系统基本功能测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 故障容错系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fault_tolerance_system()
    sys.exit(0 if success else 1)