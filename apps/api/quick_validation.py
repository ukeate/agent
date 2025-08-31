#!/usr/bin/env python3
"""
快速验证模型压缩核心功能
"""

import torch
import torch.nn as nn

# 测试量化引擎基础功能
def test_basic_quantization():
    print("🔧 测试基础量化功能...")
    
    try:
        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 2)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        model.eval()
        
        # 使用PyTorch原生量化
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # 测试推理
        test_input = torch.randn(1, 5)
        
        with torch.no_grad():
            original_output = model(test_input)
            quantized_output = quantized_model(test_input)
        
        print(f"  ✅ 原始模型输出形状: {original_output.shape}")
        print(f"  ✅ 量化模型输出形状: {quantized_output.shape}")
        print(f"  ✅ 输出差异: {torch.mean(torch.abs(original_output - quantized_output)).item():.6f}")
        
        # 计算模型大小
        import pickle
        original_size = len(pickle.dumps(model.state_dict()))
        quantized_size = len(pickle.dumps(quantized_model.state_dict()))
        compression_ratio = original_size / quantized_size
        
        print(f"  ✅ 压缩比: {compression_ratio:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 量化测试失败: {e}")
        return False


# 测试数据模型
def test_data_models():
    print("\n📊 测试数据模型...")
    
    try:
        from src.ai.model_compression.models import (
            QuantizationConfig,
            QuantizationMethod, 
            PrecisionType,
            CompressionJob,
            CompressionMethod
        )
        
        # 创建量化配置
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8,
            calibration_dataset_size=100
        )
        print(f"  ✅ 量化配置创建成功: {config.method}, {config.precision}")
        
        # 创建压缩任务
        job = CompressionJob(
            job_name="test_job",
            model_path="test.pth",
            compression_method=CompressionMethod.QUANTIZATION,
            quantization_config=config
        )
        print(f"  ✅ 压缩任务创建成功: {job.job_name}")
        
        # 测试配置转换
        config_dict = config.to_dict()
        print(f"  ✅ 配置字典转换成功: {len(config_dict)} 个字段")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据模型测试失败: {e}")
        return False


# 测试量化引擎
def test_quantization_engine():
    print("\n⚙️ 测试量化引擎...")
    
    try:
        from src.ai.model_compression.quantization_engine import QuantizationEngine
        from src.ai.model_compression.models import QuantizationConfig, QuantizationMethod, PrecisionType
        
        # 创建引擎
        engine = QuantizationEngine()
        print(f"  ✅ 量化引擎初始化成功")
        print(f"     支持的方法: {list(engine.supported_methods.keys())}")
        
        # 创建简单模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(5, 2)
            
            def forward(self, x):
                return self.linear2(self.relu(self.linear1(x)))
        
        model = TestModel()
        
        # 测试模型大小计算
        model_size = engine._get_model_size(model)
        param_count = engine._count_parameters(model)
        print(f"  ✅ 模型大小: {model_size} bytes")
        print(f"  ✅ 参数数量: {param_count}")
        
        # 配置验证
        config = QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8
        )
        
        is_valid = engine.validate_config(config)
        print(f"  ✅ 配置验证: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 量化引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 开始快速验证...\n")
    
    tests = [
        test_basic_quantization,
        test_data_models, 
        test_quantization_engine
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # 总结
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 验证结果:")
    print(f"   通过: {passed}/{total}")
    print(f"   成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 核心功能验证通过！")
        return 0
    else:
        print("⚠️  部分功能需要修复。")
        return 1


if __name__ == "__main__":
    exit(main())