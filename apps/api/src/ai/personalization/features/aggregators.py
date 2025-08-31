import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import logging

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """特征聚合器"""
    
    def __init__(self, aggregation_type: str):
        self.aggregation_type = aggregation_type
        self.aggregation_funcs = {
            "count": self._aggregate_count,
            "sum": self._aggregate_sum,
            "average": self._aggregate_average,
            "max": self._aggregate_max,
            "min": self._aggregate_min,
            "standard_deviation": self._aggregate_std,
            "percentile": self._aggregate_percentile,
            "weighted_average": self._aggregate_weighted_average,
            "exponential_decay": self._aggregate_exponential_decay,
            "moving_average": self._aggregate_moving_average
        }
        
    async def aggregate_features(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """聚合特征
        
        Args:
            features: 特征列表
            weights: 权重列表（可选）
            **kwargs: 额外参数
            
        Returns:
            Dict[str, float]: 聚合后的特征
        """
        if not features:
            return {}
            
        aggregation_func = self.aggregation_funcs.get(self.aggregation_type)
        if not aggregation_func:
            logger.warning(f"未知的聚合类型: {self.aggregation_type}")
            return {}
            
        return aggregation_func(features, weights, **kwargs)
    
    def _aggregate_count(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """计数聚合"""
        result = {}
        
        for feature_dict in features:
            for key in feature_dict:
                if key not in result:
                    result[key] = 0
                result[key] += 1
                
        return result
    
    def _aggregate_sum(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """求和聚合"""
        result = defaultdict(float)
        
        if weights:
            for feature_dict, weight in zip(features, weights):
                for key, value in feature_dict.items():
                    result[key] += value * weight
        else:
            for feature_dict in features:
                for key, value in feature_dict.items():
                    result[key] += value
                    
        return dict(result)
    
    def _aggregate_average(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """平均值聚合"""
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)
        
        for feature_dict in features:
            for key, value in feature_dict.items():
                sum_dict[key] += value
                count_dict[key] += 1
        
        result = {}
        for key in sum_dict:
            result[key] = sum_dict[key] / count_dict[key] if count_dict[key] > 0 else 0.0
            
        return result
    
    def _aggregate_max(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """最大值聚合"""
        result = {}
        
        for feature_dict in features:
            for key, value in feature_dict.items():
                if key not in result:
                    result[key] = value
                else:
                    result[key] = max(result[key], value)
                    
        return result
    
    def _aggregate_min(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """最小值聚合"""
        result = {}
        
        for feature_dict in features:
            for key, value in feature_dict.items():
                if key not in result:
                    result[key] = value
                else:
                    result[key] = min(result[key], value)
                    
        return result
    
    def _aggregate_std(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """标准差聚合"""
        values_dict = defaultdict(list)
        
        for feature_dict in features:
            for key, value in feature_dict.items():
                values_dict[key].append(value)
        
        result = {}
        for key, values in values_dict.items():
            result[key] = np.std(values) if len(values) > 1 else 0.0
            
        return result
    
    def _aggregate_percentile(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        percentile: float = 50,
        **kwargs
    ) -> Dict[str, float]:
        """百分位数聚合"""
        values_dict = defaultdict(list)
        
        for feature_dict in features:
            for key, value in feature_dict.items():
                values_dict[key].append(value)
        
        result = {}
        for key, values in values_dict.items():
            if values:
                result[key] = np.percentile(values, percentile)
            else:
                result[key] = 0.0
                
        return result
    
    def _aggregate_weighted_average(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """加权平均聚合"""
        if not weights:
            # 如果没有提供权重，使用均匀权重
            weights = [1.0 / len(features)] * len(features)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        result = defaultdict(float)
        
        for feature_dict, weight in zip(features, weights):
            for key, value in feature_dict.items():
                result[key] += value * weight
                
        return dict(result)
    
    def _aggregate_exponential_decay(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        decay_rate: float = 0.9,
        **kwargs
    ) -> Dict[str, float]:
        """指数衰减聚合（最新的特征权重更高）"""
        result = defaultdict(float)
        
        # 计算指数衰减权重
        decay_weights = [decay_rate ** i for i in range(len(features))]
        total_weight = sum(decay_weights)
        decay_weights = [w / total_weight for w in decay_weights]
        
        for feature_dict, weight in zip(features, decay_weights):
            for key, value in feature_dict.items():
                result[key] += value * weight
                
        return dict(result)
    
    def _aggregate_moving_average(
        self,
        features: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        window_size: int = 5,
        **kwargs
    ) -> Dict[str, float]:
        """移动平均聚合"""
        result = {}
        
        # 只取最近的window_size个特征
        recent_features = features[-window_size:] if len(features) > window_size else features
        
        # 计算移动平均
        sum_dict = defaultdict(float)
        count_dict = defaultdict(int)
        
        for feature_dict in recent_features:
            for key, value in feature_dict.items():
                sum_dict[key] += value
                count_dict[key] += 1
        
        for key in sum_dict:
            result[key] = sum_dict[key] / count_dict[key] if count_dict[key] > 0 else 0.0
            
        return result


class TemporalAggregator:
    """时间序列特征聚合器"""
    
    def __init__(self):
        self.time_windows = [60, 300, 1800, 3600, 86400]  # 1分钟, 5分钟, 30分钟, 1小时, 1天
        
    async def aggregate_temporal_features(
        self,
        time_series_data: List[Dict[str, Any]],
        current_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """聚合时间序列特征
        
        Args:
            time_series_data: 时间序列数据列表，每个元素包含timestamp和value
            current_time: 当前时间（用于计算时间窗口）
            
        Returns:
            Dict[str, float]: 时间窗口聚合特征
        """
        if not time_series_data:
            return {}
            
        if current_time is None:
            current_time = utc_now()
            
        aggregated_features = {}
        
        for window_seconds in self.time_windows:
            window_data = []
            cutoff_time = current_time - timedelta(seconds=window_seconds)
            
            # 筛选时间窗口内的数据
            for data_point in time_series_data:
                if "timestamp" in data_point:
                    timestamp = data_point["timestamp"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    
                    if timestamp >= cutoff_time:
                        window_data.append(data_point.get("value", 0))
            
            # 计算窗口内的统计特征
            if window_data:
                prefix = f"window_{window_seconds}s"
                aggregated_features[f"{prefix}_count"] = len(window_data)
                aggregated_features[f"{prefix}_sum"] = np.sum(window_data)
                aggregated_features[f"{prefix}_mean"] = np.mean(window_data)
                aggregated_features[f"{prefix}_std"] = np.std(window_data) if len(window_data) > 1 else 0.0
                aggregated_features[f"{prefix}_min"] = np.min(window_data)
                aggregated_features[f"{prefix}_max"] = np.max(window_data)
                aggregated_features[f"{prefix}_median"] = np.median(window_data)
                
                # 计算趋势特征
                if len(window_data) > 1:
                    # 简单线性趋势
                    x = np.arange(len(window_data))
                    slope, _ = np.polyfit(x, window_data, 1)
                    aggregated_features[f"{prefix}_trend"] = slope
                    
                    # 变化率
                    changes = np.diff(window_data)
                    aggregated_features[f"{prefix}_avg_change"] = np.mean(changes)
                    aggregated_features[f"{prefix}_volatility"] = np.std(changes)
        
        return aggregated_features


class CrossFeatureAggregator:
    """交叉特征聚合器"""
    
    def __init__(self):
        self.interaction_types = ["multiply", "add", "subtract", "divide", "polynomial"]
        
    async def create_cross_features(
        self,
        features: Dict[str, float],
        feature_pairs: Optional[List[tuple]] = None,
        max_features: int = 50
    ) -> Dict[str, float]:
        """创建交叉特征
        
        Args:
            features: 原始特征字典
            feature_pairs: 指定的特征对列表（可选）
            max_features: 最大交叉特征数
            
        Returns:
            Dict[str, float]: 交叉特征
        """
        cross_features = {}
        
        # 如果没有指定特征对，自动选择重要的特征进行交叉
        if feature_pairs is None:
            feature_keys = list(features.keys())
            # 限制特征数量以避免组合爆炸
            if len(feature_keys) > 10:
                # 选择值较大的特征（假设它们更重要）
                sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
                feature_keys = [k for k, v in sorted_features[:10]]
            
            # 生成特征对
            feature_pairs = []
            for i, key1 in enumerate(feature_keys):
                for key2 in feature_keys[i+1:]:
                    feature_pairs.append((key1, key2))
                    if len(feature_pairs) >= max_features // 2:
                        break
                if len(feature_pairs) >= max_features // 2:
                    break
        
        # 创建交叉特征
        for key1, key2 in feature_pairs:
            if key1 in features and key2 in features:
                val1 = features[key1]
                val2 = features[key2]
                
                # 乘积特征
                cross_features[f"{key1}_x_{key2}"] = val1 * val2
                
                # 和特征
                cross_features[f"{key1}_plus_{key2}"] = val1 + val2
                
                # 差特征
                cross_features[f"{key1}_minus_{key2}"] = abs(val1 - val2)
                
                # 比率特征（避免除零）
                if abs(val2) > 1e-6:
                    cross_features[f"{key1}_div_{key2}"] = val1 / val2
                
                # 多项式特征
                cross_features[f"{key1}_sq_plus_{key2}_sq"] = val1**2 + val2**2
                
                if len(cross_features) >= max_features:
                    break
        
        return cross_features


class HierarchicalAggregator:
    """层次化特征聚合器"""
    
    def __init__(self):
        self.hierarchy_levels = ["user", "session", "context", "global"]
        
    async def aggregate_hierarchical_features(
        self,
        features_by_level: Dict[str, Dict[str, float]],
        level_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """层次化聚合特征
        
        Args:
            features_by_level: 各层次的特征字典
            level_weights: 层次权重
            
        Returns:
            Dict[str, float]: 层次化聚合后的特征
        """
        if not level_weights:
            # 默认权重：用户级别最高，全局级别最低
            level_weights = {
                "user": 0.4,
                "session": 0.3,
                "context": 0.2,
                "global": 0.1
            }
        
        # 归一化权重
        total_weight = sum(level_weights.values())
        if total_weight > 0:
            level_weights = {k: v/total_weight for k, v in level_weights.items()}
        
        aggregated_features = defaultdict(float)
        
        # 按层次聚合特征
        for level, features in features_by_level.items():
            weight = level_weights.get(level, 0.1)
            
            for feature_name, feature_value in features.items():
                # 添加层次前缀
                hierarchical_name = f"{level}_{feature_name}"
                aggregated_features[hierarchical_name] = feature_value * weight
                
                # 也添加到通用特征中
                aggregated_features[feature_name] += feature_value * weight
        
        return dict(aggregated_features)