"""
异常检测服务

实现多种异常检测算法用于实验监控
"""

from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from collections import deque
import json
from ..core.database import get_db_session
from ..core.config import get_settings

class AnomalyType(str, Enum):
    """异常类型"""
    METRIC_SPIKE = "metric_spike"  # 指标突增
    METRIC_DROP = "metric_drop"  # 指标突降
    SAMPLE_RATIO_MISMATCH = "sample_ratio_mismatch"  # 样本比例不匹配
    DATA_QUALITY = "data_quality"  # 数据质量问题
    SEASONALITY = "seasonality"  # 季节性异常
    TREND_CHANGE = "trend_change"  # 趋势变化
    OUTLIER = "outlier"  # 离群值
    VARIANCE_CHANGE = "variance_change"  # 方差变化
    DISTRIBUTION_SHIFT = "distribution_shift"  # 分布偏移
    CORRELATION_BREAK = "correlation_break"  # 相关性破坏

class DetectionMethod(str, Enum):
    """检测方法"""
    Z_SCORE = "z_score"
    IQR = "iqr"  # 四分位距
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    DBSCAN = "dbscan"
    STATISTICAL_PROCESS_CONTROL = "spc"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    CUSUM = "cusum"  # 累积和控制图
    EWMA = "ewma"  # 指数加权移动平均

@dataclass
class Anomaly:
    """异常记录"""
    timestamp: datetime
    type: AnomalyType
    severity: str  # low, medium, high, critical
    metric_name: str
    experiment_id: str
    variant: Optional[str]
    observed_value: float
    expected_value: Optional[float]
    deviation: float
    detection_method: DetectionMethod
    confidence: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectionConfig:
    """检测配置"""
    methods: List[DetectionMethod] = field(
        default_factory=lambda: [DetectionMethod.Z_SCORE, DetectionMethod.IQR]
    )
    sensitivity: float = 0.95  # 灵敏度
    window_size: int = 100  # 滑动窗口大小
    min_samples: int = 30  # 最小样本数
    z_threshold: float = 3.0  # Z-score阈值
    iqr_multiplier: float = 1.5  # IQR乘数
    enable_seasonal: bool = True  # 启用季节性检测
    enable_trend: bool = True  # 启用趋势检测

class AnomalyDetectionService:
    """异常检测服务"""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.anomaly_history: deque = deque(maxlen=1000)
        self.metric_baselines: Dict[str, Dict[str, Any]] = {}
        
    async def detect_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]] = None,
        variant: Optional[str] = None
    ) -> List[Anomaly]:
        """
        检测异常
        
        Args:
            experiment_id: 实验ID
            metric_name: 指标名称
            values: 指标值列表
            timestamps: 时间戳列表
            variant: 变体名称
            
        Returns:
            检测到的异常列表
        """
        if len(values) < self.config.min_samples:
            return []
            
        anomalies = []
        
        # 使用多种方法检测
        for method in self.config.methods:
            if method == DetectionMethod.Z_SCORE:
                detected = await self._detect_zscore_anomalies(
                    experiment_id, metric_name, values, timestamps, variant
                )
                anomalies.extend(detected)
                
            elif method == DetectionMethod.IQR:
                detected = await self._detect_iqr_anomalies(
                    experiment_id, metric_name, values, timestamps, variant
                )
                anomalies.extend(detected)
                
            elif method == DetectionMethod.STATISTICAL_PROCESS_CONTROL:
                detected = await self._detect_spc_anomalies(
                    experiment_id, metric_name, values, timestamps, variant
                )
                anomalies.extend(detected)
                
            elif method == DetectionMethod.CUSUM:
                detected = await self._detect_cusum_anomalies(
                    experiment_id, metric_name, values, timestamps, variant
                )
                anomalies.extend(detected)
                
            elif method == DetectionMethod.EWMA:
                detected = await self._detect_ewma_anomalies(
                    experiment_id, metric_name, values, timestamps, variant
                )
                anomalies.extend(detected)
                
        # 检测特定类型的异常
        if self.config.enable_seasonal:
            seasonal_anomalies = await self._detect_seasonal_anomalies(
                experiment_id, metric_name, values, timestamps, variant
            )
            anomalies.extend(seasonal_anomalies)
            
        if self.config.enable_trend:
            trend_anomalies = await self._detect_trend_anomalies(
                experiment_id, metric_name, values, timestamps, variant
            )
            anomalies.extend(trend_anomalies)
            
        # 去重和排序
        anomalies = self._deduplicate_anomalies(anomalies)
        anomalies.sort(key=lambda x: (x.timestamp, -self._severity_score(x.severity)))
        
        # 保存到历史
        self.anomaly_history.extend(anomalies)
        
        return anomalies
        
    async def _detect_zscore_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]],
        variant: Optional[str]
    ) -> List[Anomaly]:
        """Z-score异常检测"""
        anomalies = []
        values_array = np.array(values)
        
        # 计算均值和标准差
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return anomalies
            
        # 计算Z-scores
        z_scores = np.abs((values_array - mean) / std)
        
        # 找出异常点
        anomaly_indices = np.where(z_scores > self.config.z_threshold)[0]
        
        for idx in anomaly_indices:
            anomaly_type = (
                AnomalyType.METRIC_SPIKE 
                if values[idx] > mean 
                else AnomalyType.METRIC_DROP
            )
            
            anomalies.append(Anomaly(
                timestamp=timestamps[idx] if timestamps else utc_now(),
                type=anomaly_type,
                severity=self._calculate_severity(z_scores[idx], method=DetectionMethod.Z_SCORE),
                metric_name=metric_name,
                experiment_id=experiment_id,
                variant=variant,
                observed_value=values[idx],
                expected_value=mean,
                deviation=z_scores[idx],
                detection_method=DetectionMethod.Z_SCORE,
                confidence=self._calculate_confidence(z_scores[idx], DetectionMethod.Z_SCORE),
                description=f"值 {values[idx]:.2f} 偏离均值 {mean:.2f} 达 {z_scores[idx]:.2f} 个标准差",
                metadata={"z_score": float(z_scores[idx]), "mean": mean, "std": std}
            ))
            
        return anomalies
        
    async def _detect_iqr_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]],
        variant: Optional[str]
    ) -> List[Anomaly]:
        """IQR异常检测"""
        anomalies = []
        values_array = np.array(values)
        
        # 计算四分位数
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        
        # 计算界限
        lower_bound = q1 - self.config.iqr_multiplier * iqr
        upper_bound = q3 + self.config.iqr_multiplier * iqr
        
        # 找出异常点
        for idx, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                anomaly_type = (
                    AnomalyType.METRIC_SPIKE 
                    if value > upper_bound 
                    else AnomalyType.METRIC_DROP
                )
                
                deviation = (
                    (value - upper_bound) / iqr 
                    if value > upper_bound 
                    else (lower_bound - value) / iqr
                )
                
                anomalies.append(Anomaly(
                    timestamp=timestamps[idx] if timestamps else utc_now(),
                    type=anomaly_type,
                    severity=self._calculate_severity(deviation, method=DetectionMethod.IQR),
                    metric_name=metric_name,
                    experiment_id=experiment_id,
                    variant=variant,
                    observed_value=value,
                    expected_value=(q1 + q3) / 2,
                    deviation=deviation,
                    detection_method=DetectionMethod.IQR,
                    confidence=self._calculate_confidence(deviation, DetectionMethod.IQR),
                    description=f"值 {value:.2f} 超出IQR范围 [{lower_bound:.2f}, {upper_bound:.2f}]",
                    metadata={
                        "q1": q1, "q3": q3, "iqr": iqr,
                        "lower_bound": lower_bound, "upper_bound": upper_bound
                    }
                ))
                
        return anomalies
        
    async def _detect_spc_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]],
        variant: Optional[str]
    ) -> List[Anomaly]:
        """统计过程控制异常检测"""
        anomalies = []
        values_array = np.array(values)
        
        # 计算控制限
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        ucl = mean + 3 * std  # 上控制限
        lcl = mean - 3 * std  # 下控制限
        uwl = mean + 2 * std  # 上警告限
        lwl = mean - 2 * std  # 下警告限
        
        # Western Electric规则
        consecutive_above_uwl = 0
        consecutive_below_lwl = 0
        consecutive_trend = 0
        last_value = None
        
        for idx, value in enumerate(values):
            # 规则1: 单点超出控制限
            if value > ucl or value < lcl:
                anomaly_type = (
                    AnomalyType.METRIC_SPIKE if value > ucl else AnomalyType.METRIC_DROP
                )
                anomalies.append(self._create_anomaly(
                    timestamp=timestamps[idx] if timestamps else utc_now(),
                    type=anomaly_type,
                    metric_name=metric_name,
                    experiment_id=experiment_id,
                    variant=variant,
                    value=value,
                    expected=mean,
                    method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                    description="单点超出3σ控制限"
                ))
                
            # 规则2: 连续2个点超出警告限
            if value > uwl:
                consecutive_above_uwl += 1
                consecutive_below_lwl = 0
                if consecutive_above_uwl >= 2:
                    anomalies.append(self._create_anomaly(
                        timestamp=timestamps[idx] if timestamps else utc_now(),
                        type=AnomalyType.TREND_CHANGE,
                        metric_name=metric_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        value=value,
                        expected=mean,
                        method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                        description="连续2个点超出2σ上警告限"
                    ))
            elif value < lwl:
                consecutive_below_lwl += 1
                consecutive_above_uwl = 0
                if consecutive_below_lwl >= 2:
                    anomalies.append(self._create_anomaly(
                        timestamp=timestamps[idx] if timestamps else utc_now(),
                        type=AnomalyType.TREND_CHANGE,
                        metric_name=metric_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        value=value,
                        expected=mean,
                        method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                        description="连续2个点超出2σ下警告限"
                    ))
            else:
                consecutive_above_uwl = 0
                consecutive_below_lwl = 0
                
            # 规则3: 连续7个点递增或递减
            if last_value is not None:
                if value > last_value:
                    if consecutive_trend >= 0:
                        consecutive_trend += 1
                    else:
                        consecutive_trend = 1
                elif value < last_value:
                    if consecutive_trend <= 0:
                        consecutive_trend -= 1
                    else:
                        consecutive_trend = -1
                        
                if abs(consecutive_trend) >= 7:
                    anomalies.append(self._create_anomaly(
                        timestamp=timestamps[idx] if timestamps else utc_now(),
                        type=AnomalyType.TREND_CHANGE,
                        metric_name=metric_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        value=value,
                        expected=mean,
                        method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                        description=f"连续7个点{'递增' if consecutive_trend > 0 else '递减'}"
                    ))
                    consecutive_trend = 0
                    
            last_value = value
            
        return anomalies
        
    async def _detect_cusum_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]],
        variant: Optional[str]
    ) -> List[Anomaly]:
        """CUSUM累积和控制图异常检测"""
        anomalies = []
        values_array = np.array(values)
        
        # 参数设置
        target = np.mean(values_array)
        std = np.std(values_array)
        k = 0.5 * std  # 参考值
        h = 5 * std  # 决策区间
        
        # 初始化CUSUM
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        
        for i in range(1, len(values)):
            # 正向CUSUM
            cusum_pos[i] = max(0, values[i] - target - k + cusum_pos[i-1])
            # 负向CUSUM
            cusum_neg[i] = max(0, target - k - values[i] + cusum_neg[i-1])
            
            # 检测异常
            if cusum_pos[i] > h:
                anomalies.append(Anomaly(
                    timestamp=timestamps[i] if timestamps else utc_now(),
                    type=AnomalyType.METRIC_SPIKE,
                    severity="high",
                    metric_name=metric_name,
                    experiment_id=experiment_id,
                    variant=variant,
                    observed_value=values[i],
                    expected_value=target,
                    deviation=cusum_pos[i] / std,
                    detection_method=DetectionMethod.CUSUM,
                    confidence=0.95,
                    description=f"CUSUM检测到持续上升趋势",
                    metadata={"cusum_value": cusum_pos[i], "threshold": h}
                ))
                cusum_pos[i] = 0  # 重置
                
            if cusum_neg[i] > h:
                anomalies.append(Anomaly(
                    timestamp=timestamps[i] if timestamps else utc_now(),
                    type=AnomalyType.METRIC_DROP,
                    severity="high",
                    metric_name=metric_name,
                    experiment_id=experiment_id,
                    variant=variant,
                    observed_value=values[i],
                    expected_value=target,
                    deviation=cusum_neg[i] / std,
                    detection_method=DetectionMethod.CUSUM,
                    confidence=0.95,
                    description=f"CUSUM检测到持续下降趋势",
                    metadata={"cusum_value": cusum_neg[i], "threshold": h}
                ))
                cusum_neg[i] = 0  # 重置
                
        return anomalies
        
    async def _detect_ewma_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]],
        variant: Optional[str]
    ) -> List[Anomaly]:
        """EWMA指数加权移动平均异常检测"""
        anomalies = []
        values_array = np.array(values)
        
        # 参数设置
        lambda_param = 0.2  # 平滑参数
        L = 3  # 控制限宽度
        
        # 初始化
        ewma = np.zeros(len(values))
        ewma[0] = values[0]
        
        # 计算EWMA
        for i in range(1, len(values)):
            ewma[i] = lambda_param * values[i] + (1 - lambda_param) * ewma[i-1]
            
        # 计算控制限
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        for i in range(len(values)):
            # 计算当前控制限
            var_ewma = (lambda_param / (2 - lambda_param)) * (
                1 - (1 - lambda_param) ** (2 * (i + 1))
            )
            ucl = mean + L * std * np.sqrt(var_ewma)
            lcl = mean - L * std * np.sqrt(var_ewma)
            
            # 检测异常
            if ewma[i] > ucl or ewma[i] < lcl:
                anomaly_type = (
                    AnomalyType.METRIC_SPIKE if ewma[i] > ucl else AnomalyType.METRIC_DROP
                )
                
                anomalies.append(Anomaly(
                    timestamp=timestamps[i] if timestamps else utc_now(),
                    type=anomaly_type,
                    severity="medium",
                    metric_name=metric_name,
                    experiment_id=experiment_id,
                    variant=variant,
                    observed_value=values[i],
                    expected_value=mean,
                    deviation=(ewma[i] - mean) / std,
                    detection_method=DetectionMethod.EWMA,
                    confidence=0.90,
                    description=f"EWMA值 {ewma[i]:.2f} 超出控制限",
                    metadata={"ewma": ewma[i], "ucl": ucl, "lcl": lcl}
                ))
                
        return anomalies
        
    async def _detect_seasonal_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]],
        variant: Optional[str]
    ) -> List[Anomaly]:
        """季节性异常检测"""
        anomalies = []
        
        if not timestamps or len(values) < 14:  # 至少需要两周数据
            return anomalies
            
        # 简化的季节性检测：比较同一星期几的数据
        weekday_data = {}
        for i, (ts, val) in enumerate(zip(timestamps, values)):
            weekday = ts.weekday()
            if weekday not in weekday_data:
                weekday_data[weekday] = []
            weekday_data[weekday].append((i, val))
            
        # 检测每个星期几的异常
        for weekday, day_values in weekday_data.items():
            if len(day_values) < 2:
                continue
                
            indices, vals = zip(*day_values)
            vals_array = np.array(vals)
            mean = np.mean(vals_array)
            std = np.std(vals_array)
            
            if std == 0:
                continue
                
            for idx, val in zip(indices, vals):
                z_score = abs((val - mean) / std)
                if z_score > 2.5:
                    anomalies.append(Anomaly(
                        timestamp=timestamps[idx],
                        type=AnomalyType.SEASONALITY,
                        severity=self._calculate_severity(z_score, DetectionMethod.Z_SCORE),
                        metric_name=metric_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        observed_value=val,
                        expected_value=mean,
                        deviation=z_score,
                        detection_method=DetectionMethod.Z_SCORE,
                        confidence=0.85,
                        description=f"星期{weekday}的值异常: {val:.2f} (平均: {mean:.2f})",
                        metadata={"weekday": weekday, "z_score": z_score}
                    ))
                    
        return anomalies
        
    async def _detect_trend_anomalies(
        self,
        experiment_id: str,
        metric_name: str,
        values: List[float],
        timestamps: Optional[List[datetime]],
        variant: Optional[str]
    ) -> List[Anomaly]:
        """趋势异常检测"""
        anomalies = []
        
        if len(values) < self.config.window_size:
            return anomalies
            
        # 使用滑动窗口检测趋势变化
        window_size = min(self.config.window_size, len(values) // 2)
        
        for i in range(window_size, len(values) - window_size):
            prev_window = values[i-window_size:i]
            curr_window = values[i:i+window_size]
            
            # Mann-Kendall趋势检验
            z_stat, p_value = self._mann_kendall_test(curr_window)
            
            if p_value < 0.05:  # 显著趋势
                # 比较前后窗口的均值
                prev_mean = np.mean(prev_window)
                curr_mean = np.mean(curr_window)
                change_rate = (curr_mean - prev_mean) / prev_mean if prev_mean != 0 else 0
                
                if abs(change_rate) > 0.2:  # 20%的变化
                    anomalies.append(Anomaly(
                        timestamp=timestamps[i] if timestamps else utc_now(),
                        type=AnomalyType.TREND_CHANGE,
                        severity="medium" if abs(change_rate) < 0.5 else "high",
                        metric_name=metric_name,
                        experiment_id=experiment_id,
                        variant=variant,
                        observed_value=curr_mean,
                        expected_value=prev_mean,
                        deviation=change_rate,
                        detection_method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                        confidence=1 - p_value,
                        description=f"检测到趋势变化: {change_rate*100:.1f}%",
                        metadata={
                            "z_statistic": z_stat,
                            "p_value": p_value,
                            "change_rate": change_rate
                        }
                    ))
                    
        return anomalies
        
    def _mann_kendall_test(self, values: List[float]) -> Tuple[float, float]:
        """Mann-Kendall趋势检验"""
        n = len(values)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(values[j] - values[i])
                
        # 计算方差
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # 计算Z统计量
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
            
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return z, p_value
        
    async def detect_sample_ratio_mismatch(
        self,
        experiment_id: str,
        control_count: int,
        treatment_count: int,
        expected_ratio: float = 0.5
    ) -> Optional[Anomaly]:
        """检测样本比例不匹配(SRM)"""
        total = control_count + treatment_count
        if total == 0:
            return None
            
        observed_ratio = control_count / total
        
        # 二项检验
        p_value = stats.binomtest(control_count, total, expected_ratio).pvalue
        
        if p_value < 0.01:  # 严格的阈值
            return Anomaly(
                timestamp=utc_now(),
                type=AnomalyType.SAMPLE_RATIO_MISMATCH,
                severity="critical",
                metric_name="sample_ratio",
                experiment_id=experiment_id,
                variant=None,
                observed_value=observed_ratio,
                expected_value=expected_ratio,
                deviation=abs(observed_ratio - expected_ratio),
                detection_method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                confidence=1 - p_value,
                description=f"样本比例 {observed_ratio:.3f} 显著偏离预期 {expected_ratio:.3f}",
                metadata={
                    "control_count": control_count,
                    "treatment_count": treatment_count,
                    "p_value": p_value
                }
            )
            
        return None
        
    async def detect_data_quality_issues(
        self,
        experiment_id: str,
        data: Dict[str, Any]
    ) -> List[Anomaly]:
        """检测数据质量问题"""
        anomalies = []
        
        # 检查缺失值
        missing_rate = data.get("missing_rate", 0)
        if missing_rate > 0.05:  # 5%以上缺失
            anomalies.append(Anomaly(
                timestamp=utc_now(),
                type=AnomalyType.DATA_QUALITY,
                severity="medium" if missing_rate < 0.1 else "high",
                metric_name="data_quality",
                experiment_id=experiment_id,
                variant=None,
                observed_value=missing_rate,
                expected_value=0,
                deviation=missing_rate,
                detection_method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                confidence=1.0,
                description=f"数据缺失率过高: {missing_rate*100:.1f}%",
                metadata={"missing_rate": missing_rate}
            ))
            
        # 检查重复值
        duplicate_rate = data.get("duplicate_rate", 0)
        if duplicate_rate > 0.01:  # 1%以上重复
            anomalies.append(Anomaly(
                timestamp=utc_now(),
                type=AnomalyType.DATA_QUALITY,
                severity="low" if duplicate_rate < 0.05 else "medium",
                metric_name="data_quality",
                experiment_id=experiment_id,
                variant=None,
                observed_value=duplicate_rate,
                expected_value=0,
                deviation=duplicate_rate,
                detection_method=DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                confidence=1.0,
                description=f"数据重复率异常: {duplicate_rate*100:.1f}%",
                metadata={"duplicate_rate": duplicate_rate}
            ))
            
        return anomalies
        
    def _create_anomaly(
        self,
        timestamp: datetime,
        type: AnomalyType,
        metric_name: str,
        experiment_id: str,
        variant: Optional[str],
        value: float,
        expected: float,
        method: DetectionMethod,
        description: str
    ) -> Anomaly:
        """创建异常记录"""
        deviation = abs(value - expected) / expected if expected != 0 else float('inf')
        
        return Anomaly(
            timestamp=timestamp,
            type=type,
            severity=self._calculate_severity(deviation, method),
            metric_name=metric_name,
            experiment_id=experiment_id,
            variant=variant,
            observed_value=value,
            expected_value=expected,
            deviation=deviation,
            detection_method=method,
            confidence=self._calculate_confidence(deviation, method),
            description=description,
            metadata={}
        )
        
    def _calculate_severity(self, deviation: float, method: DetectionMethod) -> str:
        """计算严重程度"""
        if method == DetectionMethod.Z_SCORE:
            if deviation < 3:
                return "low"
            elif deviation < 4:
                return "medium"
            elif deviation < 5:
                return "high"
            else:
                return "critical"
        elif method == DetectionMethod.IQR:
            if deviation < 2:
                return "low"
            elif deviation < 3:
                return "medium"
            elif deviation < 4:
                return "high"
            else:
                return "critical"
        else:
            # 默认基于百分比偏差
            if deviation < 0.1:
                return "low"
            elif deviation < 0.3:
                return "medium"
            elif deviation < 0.5:
                return "high"
            else:
                return "critical"
                
    def _calculate_confidence(self, deviation: float, method: DetectionMethod) -> float:
        """计算置信度"""
        if method == DetectionMethod.Z_SCORE:
            # 基于正态分布的概率
            return min(2 * (1 - stats.norm.cdf(deviation)), 1.0)
        elif method == DetectionMethod.IQR:
            # 基于经验规则
            return min(0.5 + deviation * 0.1, 1.0)
        else:
            # 默认置信度
            return 0.8
            
    def _severity_score(self, severity: str) -> int:
        """严重程度评分"""
        scores = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }
        return scores.get(severity, 0)
        
    def _deduplicate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """去重异常"""
        seen = set()
        unique_anomalies = []
        
        for anomaly in anomalies:
            key = (
                anomaly.metric_name,
                anomaly.type,
                anomaly.variant,
                anomaly.timestamp.replace(microsecond=0)  # 忽略毫秒
            )
            
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)
                
        return unique_anomalies
        
    async def get_anomaly_summary(
        self,
        experiment_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取异常摘要"""
        # 过滤历史异常
        filtered_anomalies = [
            a for a in self.anomaly_history
            if a.experiment_id == experiment_id
        ]
        
        if start_time:
            filtered_anomalies = [
                a for a in filtered_anomalies
                if a.timestamp >= start_time
            ]
            
        if end_time:
            filtered_anomalies = [
                a for a in filtered_anomalies
                if a.timestamp <= end_time
            ]
            
        # 统计
        summary = {
            "total_anomalies": len(filtered_anomalies),
            "by_type": {},
            "by_severity": {},
            "by_metric": {},
            "recent_anomalies": []
        }
        
        for anomaly in filtered_anomalies:
            # 按类型统计
            if anomaly.type not in summary["by_type"]:
                summary["by_type"][anomaly.type] = 0
            summary["by_type"][anomaly.type] += 1
            
            # 按严重程度统计
            if anomaly.severity not in summary["by_severity"]:
                summary["by_severity"][anomaly.severity] = 0
            summary["by_severity"][anomaly.severity] += 1
            
            # 按指标统计
            if anomaly.metric_name not in summary["by_metric"]:
                summary["by_metric"][anomaly.metric_name] = 0
            summary["by_metric"][anomaly.metric_name] += 1
            
        # 最近的异常
        summary["recent_anomalies"] = [
            {
                "timestamp": a.timestamp.isoformat(),
                "type": a.type,
                "severity": a.severity,
                "metric": a.metric_name,
                "description": a.description
            }
            for a in sorted(filtered_anomalies, key=lambda x: x.timestamp, reverse=True)[:10]
        ]
        
        return summary
