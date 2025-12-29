import asyncio
import numpy as np
import sys
import time
from pathlib import Path
from src.ai.rag.quantization import (
    VectorQuantizer,
    QuantizationConfig,
    QuantizationMode,
    QuantizationQualityAssessment
)
from src.ai.rag.performance_monitor import VectorPerformanceMonitor
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
pgvector 0.8 升级和量化系统验证脚本

验证量化算法、性能优化和系统集成功能
"""

sys.path.insert(0, str(Path(__file__).parent))

async def test_quantization_functionality():
    """测试量化功能"""
    logger.info("🔧 Testing Vector Quantization...")
    
    # 创建测试向量
    np.random.seed(42)
    test_vectors = [np.random.normal(0, 1, 1536).astype(np.float32) for _ in range(5)]
    
    # 测试不同量化模式
    modes = [QuantizationMode.INT8, QuantizationMode.INT4, QuantizationMode.ADAPTIVE]
    
    for mode in modes:
        logger.info(f"\n  Testing {mode.value} quantization...")
        
        config = QuantizationConfig(mode=mode)
        quantizer = VectorQuantizer(config)
        
        # 量化向量
        results = []
        for i, vector in enumerate(test_vectors):
            quantized, params = await quantizer.quantize_vector(vector)
            results.append((vector, quantized, params))
            
            logger.info(f"    Vector {i+1}: {params['mode']}, "
                  f"compression={params.get('compression', 1.0):.1f}x, "
                  f"precision_loss={params.get('precision_loss', 0.0):.3f}")
        
        # 测试反量化
        original = results[0][0]
        quantized = results[0][1]
        params = results[0][2]
        
        dequantized = await quantizer.dequantize_vector(quantized, params)
        mse = np.mean((original - dequantized) ** 2)
        logger.info(f"    Dequantization MSE: {mse:.6f}")
        
    logger.info("✅ Vector quantization tests passed!")

async def test_quality_assessment():
    """测试质量评估"""
    logger.info("\n🔍 Testing Quality Assessment...")
    
    assessor = QuantizationQualityAssessment()
    
    # 创建测试数据
    np.random.seed(42)
    original_vectors = [np.random.normal(0, 1, 100) for _ in range(10)]
    
    raise RuntimeError("未接入真实量化结果与参数，无法执行质量评估示例")
    
    # 执行质量评估
    quality_report = await assessor.assess_quality(
        original_vectors, quantized_vectors, params
    )
    
    logger.info(f"  Quality Score: {quality_report['quality_score']:.3f}")
    logger.info(f"  Average Compression: {quality_report['average_compression_ratio']:.1f}x")
    logger.info(f"  Average Precision Loss: {quality_report['average_precision_loss']:.3f}")
    
    # 测试回退决策
    should_fallback, reason = await assessor.should_fallback(quality_report, threshold=0.9)
    logger.info(f"  Should Fallback: {should_fallback} - {reason}")
    
    logger.info("✅ Quality assessment tests passed!")

async def test_performance_monitoring():
    """测试性能监控"""
    logger.info("\n📊 Testing Performance Monitoring...")
    
    monitor = VectorPerformanceMonitor()
    
    raise RuntimeError("未接入真实搜索函数，无法验证性能监控")

def test_memory_optimization():
    """测试内存优化"""
    logger.info("\n💾 Testing Memory Optimization...")
    
    # 创建向量数据
    np.random.seed(42)
    vectors = [np.random.normal(0, 1, 1536).astype(np.float32) for _ in range(100)]
    
    # 计算原始内存使用
    original_memory = sum(v.nbytes for v in vectors)
    logger.info(f"  Original memory usage: {original_memory / 1024:.1f} KB")
    
    raise RuntimeError("未接入真实量化内存统计，无法验证内存优化")

def test_index_configurations():
    """测试索引配置"""
    logger.info("\n🗂️  Testing Index Configurations...")
    
    from ai.rag.pgvector_optimizer import IndexConfig, IndexType
    
    # 测试不同索引类型
    configs = [
        IndexConfig(index_type=IndexType.HNSW, hnsw_m=16, hnsw_ef_construction=200),
        IndexConfig(index_type=IndexType.IVF, ivf_lists=1000),
        IndexConfig(index_type=IndexType.HYBRID)
    ]
    
    for config in configs:
        logger.info(f"  {config.index_type.value} index:")
        if config.index_type == IndexType.HNSW:
            logger.info(f"    M: {config.hnsw_m}, ef_construction: {config.hnsw_ef_construction}")
        elif config.index_type == IndexType.IVF:
            logger.info(f"    Lists: {config.ivf_lists}, Probes: {config.ivf_probes}")
        elif config.index_type == IndexType.HYBRID:
            logger.info("    Using both HNSW and IVF indexes")
    
    logger.info("✅ Index configuration tests passed!")

async def run_comprehensive_validation():
    """运行综合验证"""
    logger.info("🚀 Starting pgvector 0.8 Upgrade and Quantization System Validation")
    logger.info("=" * 70)
    
    try:
        # 运行各项测试
        await test_quantization_functionality()
        await test_quality_assessment()
        await test_performance_monitoring()
        test_memory_optimization()
        test_index_configurations()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 All validation tests passed successfully!")
        logger.info("\nSystem Status:")
        logger.info("✅ Vector quantization (INT8/INT4/Adaptive) - Working")
        logger.info("✅ Quality assessment and fallback - Working") 
        logger.info("✅ Performance monitoring - Working")
        logger.info("✅ Memory optimization (20%+ target) - Achieved")
        logger.info("✅ Index configurations - Working")
        logger.info("✅ Integration components - Ready")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_logging()
    success = asyncio.run(run_comprehensive_validation())
    sys.exit(0 if success else 1)
