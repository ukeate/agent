#!/usr/bin/env python3
"""
模型压缩功能验证脚本

快速验证模型压缩系统的核心功能
"""

import torch
import torch.nn as nn
from src.ai.model_compression import (
    QuantizationEngine,
    QuantizationConfig,
    QuantizationMethod,
    PrecisionType,
    CompressionPipeline,
    CompressionJob,
    CompressionMethod
)


class SimpleModel(nn.Module):
    """简单测试模型"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


def test_quantization_engine():
    """测试量化引擎"""
    print("🔧 测试量化引擎...")
    
    try:
        # 创建量化引擎
        engine = QuantizationEngine()
        print(f"  ✅ 量化引擎初始化成功")
        
        # 创建简单模型
        model = SimpleModel()
        print(f"  ✅ 测试模型创建成功")
        
        # 创建量化配置
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8
        )
        print(f"  ✅ 量化配置创建成功")
        
        # 执行量化
        quantized_model, stats = engine.quantize_model(model, config)
        print(f"  ✅ 量化执行成功")
        print(f"     - 压缩比: {stats.get('compression_ratio', 'N/A')}")
        print(f"     - 原始参数数: {stats.get('original_params', 'N/A')}")
        
        # 测试量化后的模型
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = quantized_model(test_input)
            print(f"  ✅ 量化模型推理成功, 输出形状: {output.shape}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ 量化引擎测试失败: {e}")
        return False


def test_compression_pipeline():
    """测试压缩流水线"""
    print("\n📦 测试压缩流水线...")
    
    try:
        # 创建流水线
        pipeline = CompressionPipeline()
        print(f"  ✅ 压缩流水线初始化成功")
        
        # 创建压缩任务
        job = CompressionJob(
            job_name="test_job",
            model_path="test_model.pth",
            compression_method=CompressionMethod.QUANTIZATION,
            quantization_config=QuantizationConfig(
                method=QuantizationMethod.PTQ,
                precision=PrecisionType.INT8
            )
        )
        print(f"  ✅ 压缩任务创建成功")
        
        # 提交任务
        job_id = pipeline.submit_job(job)
        print(f"  ✅ 任务提交成功, Job ID: {job_id}")
        
        # 查询任务状态
        status = pipeline.get_job_status(job_id)
        print(f"  ✅ 任务状态查询成功: {status}")
        
        # 获取统计信息
        stats = pipeline.get_pipeline_statistics()
        print(f"  ✅ 流水线统计信息获取成功")
        print(f"     - 总任务数: {stats.get('total_jobs', 0)}")
        print(f"     - 活跃任务数: {stats.get('active_jobs', 0)}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ 压缩流水线测试失败: {e}")
        return False


def test_api_integration():
    """测试API集成"""
    print("\n🌐 测试API集成...")
    
    try:
        # 检查API路由器是否正确导入
        from src.api.v1 import v1_router
        print(f"  ✅ API路由器导入成功")
        
        # 检查模型压缩路由是否注册
        routes = [route.path for route in v1_router.routes]
        compression_routes = [r for r in routes if 'model_compression' in r]
        print(f"  ✅ 发现 {len(compression_routes)} 个模型压缩API端点")
        
        # 显示部分路由
        for route in compression_routes[:5]:
            print(f"     - {route}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ API集成测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始模型压缩系统验证...\n")
    
    results = []
    
    # 运行各项测试
    results.append(test_quantization_engine())
    results.append(test_compression_pipeline())
    results.append(test_api_integration())
    
    # 总结结果
    print(f"\n📊 验证结果总结:")
    passed = sum(results)
    total = len(results)
    
    print(f"   通过: {passed}/{total}")
    print(f"   成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有核心功能验证通过！")
        return 0
    else:
        print("⚠️  部分功能存在问题，需要检查。")
        return 1


if __name__ == "__main__":
    exit(main())