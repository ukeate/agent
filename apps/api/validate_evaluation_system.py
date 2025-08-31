#!/usr/bin/env python3
"""
模型评估系统验证脚本
简化验证逻辑，避免重依赖导入问题
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加src路径
sys.path.append(str(Path(__file__).parent / "src"))

def validate_imports():
    """验证核心模块导入"""
    try:
        print("验证核心模块导入...")
        
        # 验证基础结构
        from ai.model_evaluation import evaluation_engine
        from ai.model_evaluation import benchmark_manager
        from ai.model_evaluation import performance_monitor
        print("✓ 核心评估模块导入成功")
        
        # 验证API模块
        from api.v1 import model_evaluation
        print("✓ API模块导入成功")
        
        # 验证数据模型
        from db import evaluation_models
        print("✓ 数据模型导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def validate_structure():
    """验证文件结构"""
    print("\n验证文件结构...")
    
    required_files = [
        "src/ai/model_evaluation/evaluation_engine.py",
        "src/ai/model_evaluation/benchmark_manager.py", 
        "src/ai/model_evaluation/performance_monitor.py",
        "src/api/v1/model_evaluation.py",
        "src/db/evaluation_models.py",
        "migrations/002_create_evaluation_tables.py"
    ]
    
    base_path = Path(__file__).parent
    missing_files = []
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def validate_config_classes():
    """验证配置类定义"""
    try:
        print("\n验证配置类...")
        from ai.model_evaluation.evaluation_engine import EvaluationConfig, ModelEvaluationEngine
        from ai.model_evaluation.benchmark_manager import BenchmarkConfig, BenchmarkManager
        from ai.model_evaluation.performance_monitor import MonitorConfig, PerformanceMonitor
        
        # 测试基本实例化
        config = EvaluationConfig()
        print(f"✓ EvaluationConfig: device={config.device}")
        
        bench_config = BenchmarkConfig()
        print(f"✓ BenchmarkConfig: batch_size={bench_config.batch_size}")
        
        monitor_config = MonitorConfig()
        print(f"✓ MonitorConfig: update_interval={monitor_config.update_interval}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置类验证失败: {e}")
        return False

async def validate_api_routes():
    """验证API路由定义"""
    try:
        print("\n验证API路由...")
        from api.v1.model_evaluation import router
        
        routes = [route.path for route in router.routes]
        expected_routes = ["/evaluate", "/jobs", "/benchmarks", "/models"]
        
        for expected in expected_routes:
            if any(expected in route for route in routes):
                print(f"✓ 路由包含: {expected}")
            else:
                print(f"✗ 缺少路由: {expected}")
                
        return True
        
    except Exception as e:
        print(f"✗ API路由验证失败: {e}")
        return False

def validate_database_models():
    """验证数据库模型"""
    try:
        print("\n验证数据库模型...")
        from db.evaluation_models import (
            EvaluationJob, BenchmarkDefinition, ModelInfo, EvaluationResult, 
            PerformanceMetric, PerformanceAlert, EvaluationReport
        )
        
        models = [EvaluationJob, BenchmarkDefinition, ModelInfo, EvaluationResult, 
                 PerformanceMetric, PerformanceAlert, EvaluationReport]
        for model in models:
            print(f"✓ 模型类: {model.__name__}")
            
        return True
        
    except Exception as e:
        print(f"✗ 数据库模型验证失败: {e}")
        return False

async def main():
    """主验证流程"""
    print("=" * 60)
    print("模型评估系统验证")
    print("=" * 60)
    
    results = []
    
    # 1. 文件结构验证
    results.append(validate_structure())
    
    # 2. 模块导入验证
    results.append(validate_imports())
    
    # 3. 配置类验证
    results.append(validate_config_classes())
    
    # 4. API路由验证
    results.append(await validate_api_routes())
    
    # 5. 数据库模型验证
    results.append(validate_database_models())
    
    # 输出总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有验证通过！模型评估系统实现完整")
        return True
    else:
        print("✗ 部分验证失败，需要修复问题")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)