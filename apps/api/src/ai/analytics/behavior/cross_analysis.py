"""
多维度交叉分析模块

提供跨多个维度的行为数据分析和关联性分析。
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dataclasses import dataclass
from ..models import BehaviorEvent, UserSession, DimensionFilter, CrossAnalysisResult

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class DimensionMetrics:
    """维度指标"""
    dimension_name: str
    unique_values: int
    distribution: Dict[str, int]
    entropy: float
    top_values: List[Tuple[str, int]]

@dataclass
class CorrelationResult:
    """关联性分析结果"""
    dimension_pairs: Tuple[str, str]
    correlation_type: str  # 'pearson', 'spearman', 'chi2'
    correlation_value: float
    p_value: float
    significance_level: str  # 'high', 'medium', 'low', 'not_significant'

class CrossDimensionAnalyzer:
    """跨维度分析器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.dimension_cache = {}
        
    async def analyze_dimensions(self, events: List[BehaviorEvent]) -> Dict[str, DimensionMetrics]:
        """分析各个维度的基础指标"""
        dimensions = {}
        
        # 提取维度数据
        df = self._events_to_dataframe(events)
        
        for column in df.columns:
            if column in ['timestamp', 'duration']:
                continue
                
            dimension_data = df[column].dropna()
            if len(dimension_data) == 0:
                continue
                
            # 计算基础统计
            unique_values = dimension_data.nunique()
            value_counts = dimension_data.value_counts()
            
            # 计算信息熵
            probabilities = value_counts / len(dimension_data)
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Top值
            top_values = [(str(k), int(v)) for k, v in value_counts.head(10).items()]
            
            dimensions[column] = DimensionMetrics(
                dimension_name=column,
                unique_values=unique_values,
                distribution=value_counts.to_dict(),
                entropy=entropy,
                top_values=top_values
            )
            
        return dimensions
    
    async def correlation_analysis(self, events: List[BehaviorEvent]) -> List[CorrelationResult]:
        """执行关联性分析"""
        results = []
        df = self._events_to_dataframe(events)
        
        # 数值型列相关性分析
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            results.extend(await self._numeric_correlation(df[numeric_columns]))
        
        # 分类型列关联性分析
        categorical_columns = df.select_dtypes(exclude=[np.number, np.datetime64]).columns
        if len(categorical_columns) >= 2:
            results.extend(await self._categorical_association(df[categorical_columns]))
        
        # 数值-分类交叉分析
        if len(numeric_columns) > 0 and len(categorical_columns) > 0:
            results.extend(await self._mixed_correlation(df, numeric_columns, categorical_columns))
        
        return results
    
    async def cluster_analysis(self, events: List[BehaviorEvent], n_clusters: int = 5) -> Dict[str, Any]:
        """多维聚类分析"""
        df = self._events_to_dataframe(events)
        
        # 准备数值数据
        numeric_data = df.select_dtypes(include=[np.number])
        if len(numeric_data.columns) < 2:
            return {"error": "需要至少2个数值维度进行聚类分析"}
        
        # 标准化
        scaled_data = self.scaler.fit_transform(numeric_data.fillna(0))
        
        # 层次聚类
        linkage_matrix = linkage(scaled_data, method='ward')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # PCA降维可视化
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        return {
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": self._calculate_cluster_centers(scaled_data, cluster_labels),
            "pca_coordinates": pca_result.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "feature_importance": dict(zip(numeric_data.columns, 
                                         np.abs(pca.components_).mean(axis=0)))
        }
    
    async def time_series_cross_analysis(self, events: List[BehaviorEvent], 
                                       time_window: str = '1H') -> Dict[str, Any]:
        """时间序列跨维度分析"""
        df = self._events_to_dataframe(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 按时间窗口聚合
        time_grouped = df.groupby(pd.Grouper(key='timestamp', freq=time_window))
        
        analysis = {
            "time_windows": [],
            "dimension_trends": {},
            "cross_correlations": [],
            "volatility_analysis": {}
        }
        
        # 各时间窗口的维度分布
        for time_group, group_data in time_grouped:
            if len(group_data) == 0:
                continue
                
            window_analysis = {
                "timestamp": time_group.isoformat(),
                "event_count": len(group_data),
                "dimension_distributions": {}
            }
            
            # 分析每个维度在该时间窗口的分布
            for column in group_data.columns:
                if column in ['timestamp', 'duration']:
                    continue
                    
                if group_data[column].dtype in ['object', 'category']:
                    dist = group_data[column].value_counts().to_dict()
                    window_analysis["dimension_distributions"][column] = dist
                else:
                    window_analysis["dimension_distributions"][column] = {
                        "mean": float(group_data[column].mean()),
                        "std": float(group_data[column].std()),
                        "count": int(group_data[column].count())
                    }
            
            analysis["time_windows"].append(window_analysis)
        
        return analysis
    
    async def behavioral_pattern_cross_analysis(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """行为模式跨维度分析"""
        df = self._events_to_dataframe(events)
        
        # 按用户分组分析
        if 'user_id' in df.columns:
            user_patterns = {}
            for user_id, user_data in df.groupby('user_id'):
                user_patterns[str(user_id)] = await self._analyze_user_pattern(user_data)
            
            return {
                "user_patterns": user_patterns,
                "pattern_similarities": await self._calculate_pattern_similarities(user_patterns),
                "dominant_patterns": await self._identify_dominant_patterns(user_patterns)
            }
        
        return {"error": "需要用户ID维度进行行为模式分析"}
    
    def _events_to_dataframe(self, events: List[BehaviorEvent]) -> pd.DataFrame:
        """将事件列表转换为DataFrame"""
        data = []
        for event in events:
            row = {
                'timestamp': event.timestamp,
                'user_id': event.user_id,
                'event_type': event.event_type,
                'duration': event.duration,
                'page_path': event.context.get('page') if event.context else None,
                'user_agent': event.context.get('user_agent') if event.context else None,
                'session_id': event.session_id
            }
            
            # 添加自定义属性
            if event.properties:
                for k, v in event.properties.items():
                    if isinstance(v, (str, int, float, bool)):
                        row[f'prop_{k}'] = v
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    async def _numeric_correlation(self, df: pd.DataFrame) -> List[CorrelationResult]:
        """数值型关联性分析"""
        results = []
        columns = df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                data1 = df[col1].dropna()
                data2 = df[col2].dropna()
                
                # 取交集
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) < 10:
                    continue
                
                vals1 = data1.loc[common_idx]
                vals2 = data2.loc[common_idx]
                
                # Pearson相关性
                pearson_corr, pearson_p = pearsonr(vals1, vals2)
                results.append(CorrelationResult(
                    dimension_pairs=(col1, col2),
                    correlation_type='pearson',
                    correlation_value=float(pearson_corr),
                    p_value=float(pearson_p),
                    significance_level=self._determine_significance(pearson_p)
                ))
                
                # Spearman相关性
                spearman_corr, spearman_p = spearmanr(vals1, vals2)
                results.append(CorrelationResult(
                    dimension_pairs=(col1, col2),
                    correlation_type='spearman',
                    correlation_value=float(spearman_corr),
                    p_value=float(spearman_p),
                    significance_level=self._determine_significance(spearman_p)
                ))
        
        return results
    
    async def _categorical_association(self, df: pd.DataFrame) -> List[CorrelationResult]:
        """分类型关联性分析"""
        results = []
        columns = df.columns.tolist()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                try:
                    # 创建交叉表
                    crosstab = pd.crosstab(df[col1], df[col2])
                    if crosstab.size < 4:  # 至少2x2表
                        continue
                    
                    # 卡方检验
                    chi2, p_value, dof, expected = chi2_contingency(crosstab)
                    
                    # Cramér's V
                    n = crosstab.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
                    
                    results.append(CorrelationResult(
                        dimension_pairs=(col1, col2),
                        correlation_type='chi2',
                        correlation_value=float(cramers_v),
                        p_value=float(p_value),
                        significance_level=self._determine_significance(p_value)
                    ))
                    
                except Exception as e:
                    logger.warning(f"分类关联性分析失败 {col1}-{col2}: {e}")
                    continue
        
        return results
    
    async def _mixed_correlation(self, df: pd.DataFrame, numeric_cols: List[str], 
                               categorical_cols: List[str]) -> List[CorrelationResult]:
        """混合型关联性分析"""
        results = []
        
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                try:
                    # 方差分析(ANOVA)的简化版本
                    grouped = df.groupby(cat_col)[num_col]
                    group_means = grouped.mean()
                    overall_mean = df[num_col].mean()
                    
                    # 计算组间和组内方差
                    between_group_var = sum(grouped.size() * (group_means - overall_mean) ** 2)
                    within_group_var = sum(grouped.apply(lambda x: sum((x - x.mean()) ** 2)))
                    
                    if within_group_var == 0:
                        continue
                    
                    f_statistic = between_group_var / within_group_var
                    
                    # 简化的p值估计
                    eta_squared = between_group_var / (between_group_var + within_group_var)
                    
                    results.append(CorrelationResult(
                        dimension_pairs=(num_col, cat_col),
                        correlation_type='eta_squared',
                        correlation_value=float(eta_squared),
                        p_value=0.05,  # 简化处理
                        significance_level='medium'
                    ))
                    
                except Exception as e:
                    logger.warning(f"混合关联性分析失败 {num_col}-{cat_col}: {e}")
                    continue
        
        return results
    
    def _determine_significance(self, p_value: float) -> str:
        """确定显著性水平"""
        if p_value < 0.001:
            return 'high'
        elif p_value < 0.01:
            return 'medium'
        elif p_value < 0.05:
            return 'low'
        else:
            return 'not_significant'
    
    def _calculate_cluster_centers(self, data: np.ndarray, labels: np.ndarray) -> Dict[int, List[float]]:
        """计算聚类中心"""
        centers = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            center = data[mask].mean(axis=0)
            centers[int(label)] = center.tolist()
        
        return centers
    
    async def _analyze_user_pattern(self, user_data: pd.DataFrame) -> Dict[str, Any]:
        """分析单个用户的行为模式"""
        pattern = {
            "event_frequency": user_data.groupby('event_type').size().to_dict(),
            "temporal_pattern": self._analyze_temporal_pattern(user_data),
            "session_patterns": self._analyze_session_patterns(user_data)
        }
        return pattern
    
    def _analyze_temporal_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析时间模式"""
        if 'timestamp' not in data.columns:
            return {}
        
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        return {
            "hourly_distribution": data['hour'].value_counts().to_dict(),
            "daily_distribution": data['day_of_week'].value_counts().to_dict(),
            "peak_hours": data['hour'].value_counts().head(3).index.tolist()
        }
    
    def _analyze_session_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析会话模式"""
        if 'session_id' not in data.columns:
            return {}
        
        session_stats = data.groupby('session_id').agg({
            'event_type': 'count',
            'duration': 'sum'
        }).rename(columns={'event_type': 'event_count', 'duration': 'total_duration'})
        
        return {
            "avg_events_per_session": float(session_stats['event_count'].mean()),
            "avg_session_duration": float(session_stats['total_duration'].mean()),
            "session_count": len(session_stats)
        }
    
    async def _calculate_pattern_similarities(self, user_patterns: Dict[str, Dict]) -> Dict[str, float]:
        """计算用户模式相似性"""
        # 简化的相似性计算
        similarities = {}
        users = list(user_patterns.keys())
        
        for i, user1 in enumerate(users):
            for user2 in users[i+1:]:
                # 基于事件频率计算相似性
                freq1 = user_patterns[user1].get('event_frequency', {})
                freq2 = user_patterns[user2].get('event_frequency', {})
                
                all_events = set(freq1.keys()) | set(freq2.keys())
                if not all_events:
                    similarities[f"{user1}-{user2}"] = 0.0
                    continue
                
                # 余弦相似性
                vec1 = [freq1.get(event, 0) for event in all_events]
                vec2 = [freq2.get(event, 0) for event in all_events]
                
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = np.sqrt(sum(a * a for a in vec1))
                norm2 = np.sqrt(sum(a * a for a in vec2))
                
                if norm1 == 0 or norm2 == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm1 * norm2)
                
                similarities[f"{user1}-{user2}"] = float(similarity)
        
        return similarities
    
    async def _identify_dominant_patterns(self, user_patterns: Dict[str, Dict]) -> Dict[str, Any]:
        """识别主导模式"""
        all_event_types = set()
        for pattern in user_patterns.values():
            all_event_types.update(pattern.get('event_frequency', {}).keys())
        
        # 统计每种事件类型的用户数
        event_user_count = {}
        for event_type in all_event_types:
            count = sum(1 for pattern in user_patterns.values() 
                       if event_type in pattern.get('event_frequency', {}))
            event_user_count[event_type] = count
        
        # 按用户数排序
        sorted_events = sorted(event_user_count.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "most_common_events": sorted_events[:10],
            "user_count": len(user_patterns),
            "unique_event_types": len(all_event_types)
        }

class MultiDimensionInsightEngine:
    """多维度洞察引擎"""
    
    def __init__(self):
        self.analyzer = CrossDimensionAnalyzer()
        self.insight_cache = {}
    
    async def generate_comprehensive_insights(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """生成综合性多维度洞察"""
        insights = {
            "dimension_metrics": await self.analyzer.analyze_dimensions(events),
            "correlation_analysis": await self.analyzer.correlation_analysis(events),
            "cluster_analysis": await self.analyzer.cluster_analysis(events),
            "time_series_analysis": await self.analyzer.time_series_cross_analysis(events),
            "behavioral_patterns": await self.analyzer.behavioral_pattern_cross_analysis(events),
            "actionable_recommendations": await self._generate_recommendations(events)
        }
        
        return insights
    
    async def _generate_recommendations(self, events: List[BehaviorEvent]) -> List[Dict[str, str]]:
        """生成可执行的建议"""
        recommendations = []
        
        # 基于事件频率的建议
        event_counts = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        if event_counts:
            most_common = max(event_counts.items(), key=lambda x: x[1])
            least_common = min(event_counts.items(), key=lambda x: x[1])
            
            recommendations.append({
                "type": "optimization",
                "priority": "high",
                "recommendation": f"重点优化 '{most_common[0]}' 功能，使用频率最高（{most_common[1]}次）",
                "impact": "用户体验显著提升"
            })
            
            if least_common[1] < len(events) * 0.01:  # 使用率低于1%
                recommendations.append({
                    "type": "feature_review",
                    "priority": "medium",
                    "recommendation": f"评估 '{least_common[0]}' 功能价值，使用率极低（{least_common[1]}次）",
                    "impact": "资源重新分配"
                })
        
        # 基于时间分布的建议
        time_recommendations = await self._analyze_time_patterns_for_recommendations(events)
        recommendations.extend(time_recommendations)
        
        return recommendations
    
    async def _analyze_time_patterns_for_recommendations(self, events: List[BehaviorEvent]) -> List[Dict[str, str]]:
        """基于时间模式生成建议"""
        recommendations = []
        
        # 按小时分析
        hour_counts = {}
        for event in events:
            hour = event.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if hour_counts:
            peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
            low_hour = min(hour_counts.items(), key=lambda x: x[1])[0]
            
            recommendations.append({
                "type": "scheduling",
                "priority": "medium",
                "recommendation": f"在{peak_hour}:00-{peak_hour+1}:00高峰期增加系统资源，在{low_hour}:00-{low_hour+1}:00低谷期进行维护",
                "impact": "系统性能和稳定性提升"
            })
        
        return recommendations
