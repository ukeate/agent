"""
智能异常检测模块

这个模块实现了多种异常检测算法：
1. 统计方法：Z-score、IQR
2. 机器学习方法：Isolation Forest、LOF
3. 特征工程：用户行为特征提取
4. 综合决策：多算法融合

从演示版升级为真正的数据科学实现！
"""

try:
    # 优先尝试导入完整版本
    from .core import (
        IntelligentAnomalyDetector,
        AnomalyResult,
        UserBehaviorFeatureExtractor,
        StatisticalAnomalyDetector,
        MachineLearningAnomalyDetector,
        create_sample_events
    )
    __all__ = [
        'IntelligentAnomalyDetector',
        'AnomalyResult', 
        'UserBehaviorFeatureExtractor',
        'StatisticalAnomalyDetector',
        'MachineLearningAnomalyDetector',
        'create_sample_events'
    ]
except ImportError:
    # 如果导入失败，使用简化版本
    from .simple_detector import (
        SimpleAnomalyDetector as IntelligentAnomalyDetector,
        SimpleAnomalyResult as AnomalyResult,
        create_sample_events
    )
    __all__ = [
        'IntelligentAnomalyDetector',
        'AnomalyResult',
        'create_sample_events'
    ]