"""
情感状态时间序列建模和分析

处理情感轨迹的时间序列分析、趋势检测和模式发现
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .models import EmotionState, EmotionStatistics
from .space_mapper import EmotionSpaceMapper

logger = logging.getLogger(__name__)


class TemporalEmotionAnalyzer:
    """情感时间序列分析器"""
    
    def __init__(self):
        self.space_mapper = EmotionSpaceMapper()
    
    def create_emotion_timeseries(
        self, 
        states: List[EmotionState],
        time_resolution: str = '1H'  # 时间分辨率: 1H, 30min, 15min等
    ) -> pd.DataFrame:
        """
        创建情感状态时间序列数据框
        
        Args:
            states: 情感状态列表
            time_resolution: 时间分辨率
            
        Returns:
            时间序列数据框
        """
        if not states:
            return pd.DataFrame()
        
        try:
            # 创建基础数据框
            data = []
            for state in states:
                vad = self.space_mapper.map_state_to_space(state)
                data.append({
                    'timestamp': state.timestamp,
                    'emotion': state.emotion,
                    'intensity': state.intensity,
                    'valence': vad[0],
                    'arousal': vad[1],
                    'dominance': vad[2],
                    'confidence': state.confidence,
                    'source': state.source,
                    'session_id': state.session_id
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # 按时间分辨率重采样
            if time_resolution != 'raw':
                # 数值列使用加权平均（基于confidence）
                numeric_cols = ['intensity', 'valence', 'arousal', 'dominance']
                
                def weighted_mean(group):
                    weights = group['confidence'] if 'confidence' in group else 1
                    result = {}
                    for col in numeric_cols:
                        if col in group:
                            result[col] = np.average(group[col], weights=weights)
                    # 选择置信度最高的情感类型
                    if len(group) > 0:
                        max_conf_idx = group['confidence'].idxmax()
                        result['emotion'] = group.loc[max_conf_idx, 'emotion']
                        result['confidence'] = group['confidence'].mean()
                    return pd.Series(result)
                
                df = df.resample(time_resolution).apply(weighted_mean)
            
            return df
            
        except Exception as e:
            logger.error(f"创建时间序列失败: {e}")
            return pd.DataFrame()
    
    def detect_emotion_trends(
        self, 
        states: List[EmotionState],
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        检测情感趋势
        
        Args:
            states: 情感状态列表
            window_size: 滑动窗口大小
            
        Returns:
            趋势分析结果
        """
        if len(states) < window_size:
            return {'error': '数据不足，无法分析趋势'}
        
        try:
            df = self.create_emotion_timeseries(states)
            if df.empty:
                return {'error': '无法创建时间序列'}
            
            trends = {
                'valence_trend': self._calculate_trend(df['valence'].values),
                'arousal_trend': self._calculate_trend(df['arousal'].values),
                'dominance_trend': self._calculate_trend(df['dominance'].values),
                'intensity_trend': self._calculate_trend(df['intensity'].values),
                'trend_strength': 0.0,
                'trend_direction': 'stable',
                'change_points': []
            }
            
            # 计算总体趋势强度
            trend_values = [
                abs(trends['valence_trend']),
                abs(trends['arousal_trend']),
                abs(trends['dominance_trend']),
                abs(trends['intensity_trend'])
            ]
            trends['trend_strength'] = float(np.mean(trend_values))
            
            # 确定总体趋势方向
            valence_trend = trends['valence_trend']
            if valence_trend > 0.1:
                trends['trend_direction'] = 'improving'
            elif valence_trend < -0.1:
                trends['trend_direction'] = 'declining'
            else:
                trends['trend_direction'] = 'stable'
            
            # 检测变化点
            trends['change_points'] = self._detect_change_points(df['valence'].values)
            
            return trends
            
        except Exception as e:
            logger.error(f"检测情感趋势失败: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """使用线性回归计算趋势斜率"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = values
        
        # 移除NaN值
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # 计算斜率
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        return float(slope)
    
    def _detect_change_points(
        self, 
        values: np.ndarray,
        min_size: int = 5,
        threshold: float = 0.3
    ) -> List[int]:
        """检测时间序列中的变化点"""
        if len(values) < min_size * 2:
            return []
        
        change_points = []
        
        try:
            # 简单的变化点检测：比较前后窗口的均值差异
            for i in range(min_size, len(values) - min_size):
                before_mean = np.nanmean(values[max(0, i-min_size):i])
                after_mean = np.nanmean(values[i:min(len(values), i+min_size)])
                
                if abs(after_mean - before_mean) > threshold:
                    change_points.append(i)
            
            return change_points
            
        except Exception as e:
            logger.error(f"检测变化点失败: {e}")
            return []
    
    def analyze_temporal_patterns(
        self, 
        states: List[EmotionState]
    ) -> Dict[str, Any]:
        """
        分析时间模式（小时、星期模式等）
        
        Args:
            states: 情感状态列表
            
        Returns:
            时间模式分析结果
        """
        if not states:
            return {}
        
        try:
            df = self.create_emotion_timeseries(states, time_resolution='raw')
            if df.empty:
                return {}
            
            patterns = {
                'hourly_patterns': {},
                'daily_patterns': {},
                'weekly_patterns': {},
                'monthly_patterns': {},
                'seasonal_effects': {}
            }
            
            # 小时模式分析
            df['hour'] = df.index.hour
            hourly_stats = df.groupby('hour').agg({
                'valence': ['mean', 'std'],
                'arousal': ['mean', 'std'],
                'dominance': ['mean', 'std'],
                'intensity': ['mean', 'std']
            })
            
            for hour in range(24):
                if hour in hourly_stats.index:
                    patterns['hourly_patterns'][hour] = {
                        'valence': float(hourly_stats.loc[hour, ('valence', 'mean')]),
                        'arousal': float(hourly_stats.loc[hour, ('arousal', 'mean')]),
                        'dominance': float(hourly_stats.loc[hour, ('dominance', 'mean')]),
                        'intensity': float(hourly_stats.loc[hour, ('intensity', 'mean')]),
                        'count': int(hourly_stats.loc[hour, ('valence', 'count')])
                    }
            
            # 星期模式分析
            df['weekday'] = df.index.dayofweek
            weekly_stats = df.groupby('weekday').agg({
                'valence': ['mean', 'std'],
                'arousal': ['mean', 'std'],
                'intensity': ['mean', 'std']
            })
            
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday']
            
            for weekday in range(7):
                if weekday in weekly_stats.index:
                    patterns['weekly_patterns'][weekday_names[weekday]] = {
                        'valence': float(weekly_stats.loc[weekday, ('valence', 'mean')]),
                        'arousal': float(weekly_stats.loc[weekday, ('arousal', 'mean')]),
                        'intensity': float(weekly_stats.loc[weekday, ('intensity', 'mean')])
                    }
            
            # 检测最佳和最差时间段
            patterns['best_hours'] = self._find_peak_hours(patterns['hourly_patterns'], 'valence')
            patterns['worst_hours'] = self._find_peak_hours(patterns['hourly_patterns'], 'valence', ascending=True)
            
            return patterns
            
        except Exception as e:
            logger.error(f"分析时间模式失败: {e}")
            return {}
    
    def _find_peak_hours(
        self, 
        hourly_patterns: Dict, 
        metric: str,
        top_k: int = 3,
        ascending: bool = False
    ) -> List[Tuple[int, float]]:
        """找出指定指标的峰值时间段"""
        if not hourly_patterns:
            return []
        
        hour_values = [
            (hour, data.get(metric, 0)) 
            for hour, data in hourly_patterns.items()
        ]
        
        hour_values.sort(key=lambda x: x[1], reverse=not ascending)
        return hour_values[:top_k]
    
    def detect_emotion_cycles(
        self, 
        states: List[EmotionState],
        min_period: int = 4,
        max_period: int = 48
    ) -> Dict[str, Any]:
        """
        检测情感周期性模式
        
        Args:
            states: 情感状态列表
            min_period: 最小周期长度（小时）
            max_period: 最大周期长度（小时）
            
        Returns:
            周期检测结果
        """
        if len(states) < max_period * 2:
            return {'error': '数据不足，无法检测周期'}
        
        try:
            df = self.create_emotion_timeseries(states, time_resolution='1H')
            if df.empty:
                return {}
            
            cycles = {
                'detected_cycles': [],
                'dominant_period': 0,
                'cycle_strength': 0.0,
                'phase_analysis': {}
            }
            
            # 使用快速傅里叶变换检测周期
            for column in ['valence', 'arousal', 'dominance']:
                if column in df.columns:
                    values = df[column].dropna().values
                    if len(values) >= max_period:
                        period_result = self._detect_period_fft(values, min_period, max_period)
                        if period_result:
                            cycles['detected_cycles'].append({
                                'dimension': column,
                                'period': period_result['period'],
                                'strength': period_result['strength']
                            })
            
            # 找出主导周期
            if cycles['detected_cycles']:
                dominant_cycle = max(cycles['detected_cycles'], key=lambda x: x['strength'])
                cycles['dominant_period'] = dominant_cycle['period']
                cycles['cycle_strength'] = dominant_cycle['strength']
            
            return cycles
            
        except Exception as e:
            logger.error(f"检测情感周期失败: {e}")
            return {'error': str(e)}
    
    def _detect_period_fft(
        self, 
        values: np.ndarray,
        min_period: int,
        max_period: int
    ) -> Optional[Dict]:
        """使用FFT检测时间序列的主要周期"""
        try:
            # 去除趋势和均值
            detrended = values - np.polyval(np.polyfit(range(len(values)), values, 1), range(len(values)))
            detrended = detrended - np.mean(detrended)
            
            # FFT分析
            fft_result = np.fft.fft(detrended)
            frequencies = np.fft.fftfreq(len(values))
            
            # 找出主要频率
            power_spectrum = np.abs(fft_result)
            positive_freq_indices = np.where(frequencies > 0)[0]
            
            # 在指定周期范围内寻找峰值
            valid_periods = []
            for i in positive_freq_indices:
                period = 1 / frequencies[i] if frequencies[i] != 0 else float('inf')
                if min_period <= period <= max_period:
                    valid_periods.append((period, power_spectrum[i]))
            
            if valid_periods:
                # 找出功率最大的周期
                dominant_period, power = max(valid_periods, key=lambda x: x[1])
                strength = power / np.sum(power_spectrum)  # 相对强度
                
                return {
                    'period': int(dominant_period),
                    'strength': float(strength)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"FFT周期检测失败: {e}")
            return None
    
    def calculate_emotional_volatility(
        self, 
        states: List[EmotionState],
        window_size: int = 24  # 24小时窗口
    ) -> Dict[str, float]:
        """
        计算情感波动性指标
        
        Args:
            states: 情感状态列表
            window_size: 计算窗口大小（小时）
            
        Returns:
            波动性指标
        """
        if not states:
            return {}
        
        try:
            df = self.create_emotion_timeseries(states, time_resolution='1H')
            if df.empty:
                return {}
            
            volatility = {}
            
            for column in ['valence', 'arousal', 'dominance', 'intensity']:
                if column in df.columns:
                    values = df[column].dropna()
                    if len(values) >= 2:
                        # 计算滚动标准差
                        rolling_std = values.rolling(window=min(window_size, len(values))).std()
                        volatility[f'{column}_volatility'] = float(rolling_std.mean())
                        
                        # 计算变化率的绝对值均值
                        changes = values.diff().abs()
                        volatility[f'{column}_change_rate'] = float(changes.mean())
            
            # 综合波动性评分
            if volatility:
                volatility_scores = [
                    v for k, v in volatility.items() 
                    if k.endswith('_volatility') and not np.isnan(v)
                ]
                if volatility_scores:
                    volatility['overall_volatility'] = float(np.mean(volatility_scores))
            
            return volatility
            
        except Exception as e:
            logger.error(f"计算情感波动性失败: {e}")
            return {}
    
    def generate_temporal_report(
        self, 
        states: List[EmotionState]
    ) -> Dict[str, Any]:
        """
        生成完整的时间序列分析报告
        
        Args:
            states: 情感状态列表
            
        Returns:
            综合分析报告
        """
        if not states:
            return {'error': '没有数据'}
        
        report = {
            'summary': {
                'total_records': len(states),
                'time_span': self._calculate_time_span(states),
                'data_coverage': self._calculate_data_coverage(states)
            },
            'trends': self.detect_emotion_trends(states),
            'patterns': self.analyze_temporal_patterns(states),
            'cycles': self.detect_emotion_cycles(states),
            'volatility': self.calculate_emotional_volatility(states)
        }
        
        return report
    
    def _calculate_time_span(self, states: List[EmotionState]) -> Dict[str, str]:
        """计算数据时间跨度"""
        if not states:
            return {}
        
        timestamps = [state.timestamp for state in states]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        return {
            'start_time': min_time.isoformat(),
            'end_time': max_time.isoformat(),
            'duration_days': (max_time - min_time).days
        }
    
    def _calculate_data_coverage(self, states: List[EmotionState]) -> Dict[str, float]:
        """计算数据覆盖度"""
        if not states:
            return {}
        
        df = self.create_emotion_timeseries(states, time_resolution='1H')
        if df.empty:
            return {}
        
        # 计算时间覆盖率
        total_hours = len(df)
        non_null_hours = df.dropna().shape[0]
        coverage_rate = non_null_hours / total_hours if total_hours > 0 else 0
        
        return {
            'coverage_rate': float(coverage_rate),
            'total_hours': total_hours,
            'recorded_hours': non_null_hours
        }