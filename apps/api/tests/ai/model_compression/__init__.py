"""
模型压缩测试包

包含模型压缩相关功能的所有测试用例
"""

__version__ = "1.0.0"

# 测试包信息
TEST_PACKAGE_INFO = {
    "name": "模型压缩测试套件",
    "version": __version__,
    "description": "全面测试模型压缩和量化工具的功能和性能",
    "test_modules": [
        "test_quantization_engine",
        "test_distillation_trainer", 
        "test_pruning_engine",
        "test_compression_pipeline",
        "test_api_endpoints",
    ],
    "coverage_areas": [
        "量化算法测试（PTQ、QAT、GPTQ、AWQ）",
        "知识蒸馏策略测试（响应式、特征式、注意力式）",
        "模型剪枝方法测试（结构化、非结构化）",
        "压缩流水线管理测试",
        "API端点功能测试",
        "边界情况和错误处理测试",
        "并发和性能测试"
    ],
    "test_statistics": {
        "total_test_cases": "150+",
        "parametrized_tests": "20+",
        "async_tests": "10+",
        "mock_tests": "15+"
    }
}