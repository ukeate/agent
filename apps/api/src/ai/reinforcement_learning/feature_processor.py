"""
上下文特征处理器

处理用户特征和物品特征，为上下文感知的多臂老虎机算法提供标准化的特征向量。
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass
import json


@dataclass
class FeatureConfig:
    """特征配置"""
    name: str
    feature_type: str  # 'numeric', 'categorical', 'text', 'temporal'
    required: bool = False
    default_value: Any = None
    normalization: str = 'none'  # 'none', 'min_max', 'z_score', 'log'
    encoding: str = 'none'  # 'none', 'one_hot', 'label', 'embedding'
    max_categories: int = 100  # 最大分类数


class ContextFeatureProcessor:
    """上下文特征处理器"""
    
    def __init__(self, config: Optional[List[FeatureConfig]] = None):
        """
        初始化特征处理器
        
        Args:
            config: 特征配置列表
        """
        self.config = config or []
        self.feature_stats = {}  # 特征统计信息（用于标准化）
        self.category_mappings = {}  # 分类特征映射
        self.is_fitted = False
        self.feature_dimension = 0
        self.feature_names = []
    
    def fit(self, samples: List[Dict[str, Any]]) -> 'ContextFeatureProcessor':
        """
        拟合特征处理器
        
        Args:
            samples: 特征样本列表
            
        Returns:
            self
        """
        if not samples:
            raise ValueError("需要至少一个样本进行拟合")
        
        # 自动推断特征配置（如果未提供）
        if not self.config:
            self.config = self._infer_feature_config(samples)
        
        # 计算特征统计信息
        self._compute_feature_stats(samples)
        
        # 构建分类特征映射
        self._build_category_mappings(samples)
        
        # 计算特征维度
        self.feature_dimension = self._calculate_feature_dimension()
        
        # 生成特征名称
        self.feature_names = self._generate_feature_names()
        
        self.is_fitted = True
        return self
    
    def transform(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        转换单个样本为特征向量
        
        Args:
            sample: 输入样本
            
        Returns:
            特征向量
        """
        if not self.is_fitted:
            raise ValueError("特征处理器未拟合，请先调用fit()方法")
        
        feature_vector = []
        
        for feature_config in self.config:
            feature_value = sample.get(feature_config.name, feature_config.default_value)
            transformed_features = self._transform_single_feature(feature_value, feature_config)
            feature_vector.extend(transformed_features)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def transform_batch(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """
        批量转换样本为特征矩阵
        
        Args:
            samples: 样本列表
            
        Returns:
            特征矩阵 (n_samples, n_features)
        """
        if not samples:
            return np.array([]).reshape(0, self.feature_dimension)
        
        feature_matrix = []
        for sample in samples:
            feature_vector = self.transform(sample)
            feature_matrix.append(feature_vector)
        
        return np.array(feature_matrix, dtype=np.float32)
    
    def fit_transform(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """
        拟合并转换样本
        
        Args:
            samples: 样本列表
            
        Returns:
            特征矩阵
        """
        self.fit(samples)
        return self.transform_batch(samples)
    
    def _infer_feature_config(self, samples: List[Dict[str, Any]]) -> List[FeatureConfig]:
        """推断特征配置"""
        feature_configs = []
        
        # 获取所有特征名称
        all_features = set()
        for sample in samples:
            all_features.update(sample.keys())
        
        for feature_name in sorted(all_features):
            # 分析特征值类型
            values = [sample.get(feature_name) for sample in samples if feature_name in sample]
            values = [v for v in values if v is not None]
            
            if not values:
                continue
            
            feature_type = self._infer_feature_type(values)
            
            config = FeatureConfig(
                name=feature_name,
                feature_type=feature_type,
                required=True,  # 默认都是必需的
                normalization='min_max' if feature_type == 'numeric' else 'none',
                encoding='one_hot' if feature_type == 'categorical' else 'none'
            )
            
            feature_configs.append(config)
        
        return feature_configs
    
    def _infer_feature_type(self, values: List[Any]) -> str:
        """推断特征类型"""
        if not values:
            return 'numeric'
        
        # 检查是否为数值型
        numeric_count = sum(1 for v in values if isinstance(v, (int, float)))
        if numeric_count > len(values) * 0.8:
            return 'numeric'
        
        # 检查是否为时间型
        temporal_count = sum(1 for v in values if isinstance(v, (datetime, str)) and self._is_temporal_string(str(v)))
        if temporal_count > len(values) * 0.5:
            return 'temporal'
        
        # 检查分类数量
        unique_values = len(set(str(v) for v in values))
        if unique_values <= min(len(values) * 0.5, 50):
            return 'categorical'
        
        # 默认为文本
        return 'text'
    
    def _is_temporal_string(self, value: str) -> bool:
        """检查字符串是否为时间格式"""
        temporal_patterns = [
            '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
            '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d'
        ]
        
        for pattern in temporal_patterns:
            try:
                datetime.strptime(value, pattern)
                return True
            except ValueError:
                continue
        
        return False
    
    def _compute_feature_stats(self, samples: List[Dict[str, Any]]):
        """计算特征统计信息"""
        for feature_config in self.config:
            feature_name = feature_config.name
            values = []
            
            for sample in samples:
                value = sample.get(feature_name, feature_config.default_value)
                if value is not None and feature_config.feature_type == 'numeric':
                    try:
                        values.append(float(value))
                    except (ValueError, TypeError):
                        pass
            
            if values and feature_config.feature_type == 'numeric':
                self.feature_stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                self.feature_stats[feature_name] = {'count': len(values)}
    
    def _build_category_mappings(self, samples: List[Dict[str, Any]]):
        """构建分类特征映射"""
        for feature_config in self.config:
            if feature_config.feature_type != 'categorical':
                continue
            
            feature_name = feature_config.name
            categories = set()
            
            for sample in samples:
                value = sample.get(feature_name, feature_config.default_value)
                if value is not None:
                    categories.add(str(value))
            
            # 限制分类数量
            sorted_categories = sorted(categories)
            if len(sorted_categories) > feature_config.max_categories:
                sorted_categories = sorted_categories[:feature_config.max_categories]
            
            self.category_mappings[feature_name] = {
                'categories': sorted_categories,
                'category_to_index': {cat: i for i, cat in enumerate(sorted_categories)},
                'default_index': len(sorted_categories)  # 未知分类的索引
            }
    
    def _calculate_feature_dimension(self) -> int:
        """计算特征维度"""
        total_dim = 0
        
        for feature_config in self.config:
            if feature_config.feature_type == 'numeric':
                total_dim += 1
            elif feature_config.feature_type == 'categorical':
                if feature_config.encoding == 'one_hot':
                    # One-hot编码：分类数+1（未知分类）
                    num_categories = len(self.category_mappings.get(feature_config.name, {}).get('categories', []))
                    total_dim += num_categories + 1
                else:
                    # 标签编码：1维
                    total_dim += 1
            elif feature_config.feature_type == 'temporal':
                total_dim += 4  # 年、月、日、小时
            elif feature_config.feature_type == 'text':
                total_dim += 1  # 简化为文本长度
            else:
                total_dim += 1
        
        return total_dim
    
    def _generate_feature_names(self) -> List[str]:
        """生成特征名称列表"""
        feature_names = []
        
        for feature_config in self.config:
            if feature_config.feature_type == 'numeric':
                feature_names.append(f"{feature_config.name}")
            elif feature_config.feature_type == 'categorical' and feature_config.encoding == 'one_hot':
                categories = self.category_mappings.get(feature_config.name, {}).get('categories', [])
                for cat in categories:
                    feature_names.append(f"{feature_config.name}_{cat}")
                feature_names.append(f"{feature_config.name}_unknown")
            elif feature_config.feature_type == 'temporal':
                feature_names.extend([
                    f"{feature_config.name}_year",
                    f"{feature_config.name}_month", 
                    f"{feature_config.name}_day",
                    f"{feature_config.name}_hour"
                ])
            else:
                feature_names.append(f"{feature_config.name}")
        
        return feature_names
    
    def _transform_single_feature(self, value: Any, config: FeatureConfig) -> List[float]:
        """转换单个特征"""
        if value is None:
            value = config.default_value
        
        if config.feature_type == 'numeric':
            return self._transform_numeric_feature(value, config)
        elif config.feature_type == 'categorical':
            return self._transform_categorical_feature(value, config)
        elif config.feature_type == 'temporal':
            return self._transform_temporal_feature(value, config)
        elif config.feature_type == 'text':
            return self._transform_text_feature(value, config)
        else:
            return [0.0]
    
    def _transform_numeric_feature(self, value: Any, config: FeatureConfig) -> List[float]:
        """转换数值特征"""
        try:
            numeric_value = float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            numeric_value = 0.0
        
        # 应用标准化
        if config.normalization == 'min_max':
            stats = self.feature_stats.get(config.name, {})
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0)
            if max_val > min_val:
                numeric_value = (numeric_value - min_val) / (max_val - min_val)
            else:
                numeric_value = 0.0
        elif config.normalization == 'z_score':
            stats = self.feature_stats.get(config.name, {})
            mean = stats.get('mean', 0.0)
            std = stats.get('std', 1.0)
            if std > 0:
                numeric_value = (numeric_value - mean) / std
            else:
                numeric_value = 0.0
        elif config.normalization == 'log':
            numeric_value = np.log(max(numeric_value, 1e-8))
        
        return [numeric_value]
    
    def _transform_categorical_feature(self, value: Any, config: FeatureConfig) -> List[float]:
        """转换分类特征"""
        str_value = str(value) if value is not None else 'unknown'
        mapping = self.category_mappings.get(config.name, {})
        
        if config.encoding == 'one_hot':
            # One-hot编码
            categories = mapping.get('categories', [])
            one_hot = [0.0] * (len(categories) + 1)  # +1 for unknown
            
            if str_value in mapping.get('category_to_index', {}):
                index = mapping['category_to_index'][str_value]
                one_hot[index] = 1.0
            else:
                one_hot[-1] = 1.0  # unknown category
            
            return one_hot
        else:
            # 标签编码
            if str_value in mapping.get('category_to_index', {}):
                index = mapping['category_to_index'][str_value]
            else:
                index = mapping.get('default_index', 0)
            
            return [float(index)]
    
    def _transform_temporal_feature(self, value: Any, config: FeatureConfig) -> List[float]:
        """转换时间特征"""
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                # 尝试多种时间格式解析
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        dt = datetime.strptime(value, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    dt = utc_now()
            except:
                dt = utc_now()
        else:
            dt = utc_now()
        
        # 提取时间特征
        year_normalized = (dt.year - 2000) / 50.0  # 假设年份范围在2000-2050
        month_normalized = (dt.month - 1) / 11.0  # 0-11标准化
        day_normalized = (dt.day - 1) / 30.0  # 0-30标准化
        hour_normalized = dt.hour / 23.0  # 0-23标准化
        
        return [year_normalized, month_normalized, day_normalized, hour_normalized]
    
    def _transform_text_feature(self, value: Any, config: FeatureConfig) -> List[float]:
        """转换文本特征（简化版本）"""
        text = str(value) if value is not None else ''
        
        # 简单的文本特征：长度标准化
        text_length = len(text)
        normalized_length = min(text_length / 100.0, 1.0)  # 假设最大长度100
        
        return [normalized_length]
    
    def get_feature_importance(self, feature_weights: np.ndarray) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            feature_weights: 特征权重向量
            
        Returns:
            特征重要性字典
        """
        if len(feature_weights) != self.feature_dimension:
            raise ValueError(f"权重向量长度({len(feature_weights)})不匹配特征维度({self.feature_dimension})")
        
        importance = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(feature_weights):
                importance[feature_name] = float(np.abs(feature_weights[i]))
        
        return importance
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "num_features": len(self.config),
            "feature_dimension": self.feature_dimension,
            "is_fitted": self.is_fitted,
            "feature_types": [config.feature_type for config in self.config],
            "feature_names_sample": self.feature_names[:10] if self.feature_names else [],
            "category_counts": {
                name: len(mapping.get('categories', []))
                for name, mapping in self.category_mappings.items()
            }
        }


class UserFeatureProcessor(ContextFeatureProcessor):
    """用户特征处理器"""
    
    def __init__(self):
        """初始化用户特征处理器"""
        user_features = [
            FeatureConfig("age", "numeric", normalization="min_max"),
            FeatureConfig("gender", "categorical", encoding="one_hot"),
            FeatureConfig("location", "categorical", encoding="one_hot"),
            FeatureConfig("signup_date", "temporal"),
            FeatureConfig("total_purchases", "numeric", normalization="log"),
            FeatureConfig("avg_session_duration", "numeric", normalization="z_score"),
            FeatureConfig("preferred_categories", "categorical", encoding="one_hot"),
            FeatureConfig("device_type", "categorical", encoding="one_hot"),
        ]
        super().__init__(user_features)


class ItemFeatureProcessor(ContextFeatureProcessor):
    """物品特征处理器"""
    
    def __init__(self):
        """初始化物品特征处理器"""
        item_features = [
            FeatureConfig("category", "categorical", encoding="one_hot"),
            FeatureConfig("price", "numeric", normalization="log"),
            FeatureConfig("brand", "categorical", encoding="one_hot"),
            FeatureConfig("rating", "numeric", normalization="min_max"),
            FeatureConfig("num_reviews", "numeric", normalization="log"),
            FeatureConfig("created_date", "temporal"),
            FeatureConfig("popularity_score", "numeric", normalization="z_score"),
            FeatureConfig("tags", "text"),
        ]
        super().__init__(item_features)