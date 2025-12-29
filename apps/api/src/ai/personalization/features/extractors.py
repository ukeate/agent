import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import numpy as np
from collections import Counter

from src.core.logging import get_logger
logger = get_logger(__name__)

class FeatureExtractor:
    """特征提取器基类"""
    
    def __init__(self, extractor_type: str):
        self.extractor_type = extractor_type
        self.feature_cache = {}
        
    async def extract_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取特征
        
        Args:
            user_id: 用户ID
            context: 上下文信息
            
        Returns:
            Dict[str, float]: 提取的特征字典
        """
        if self.extractor_type == "temporal":
            return await self._extract_temporal_features(user_id, context)
        elif self.extractor_type == "behavioral":
            return await self._extract_behavioral_features(user_id, context)
        elif self.extractor_type == "contextual":
            return await self._extract_contextual_features(user_id, context)
        else:
            logger.warning(f"未知的提取器类型: {self.extractor_type}")
            return {}
    
    async def _extract_temporal_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取时间特征"""
        features = {}
        
        try:
            now = utc_now()
            
            # 时间点特征
            features["hour_of_day"] = now.hour / 24.0
            features["day_of_week"] = now.weekday() / 7.0
            features["day_of_month"] = now.day / 31.0
            features["month_of_year"] = now.month / 12.0
            
            # 时间段特征
            features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
            features["is_business_hour"] = 1.0 if 9 <= now.hour < 18 else 0.0
            features["is_morning"] = 1.0 if 6 <= now.hour < 12 else 0.0
            features["is_afternoon"] = 1.0 if 12 <= now.hour < 18 else 0.0
            features["is_evening"] = 1.0 if 18 <= now.hour < 22 else 0.0
            features["is_night"] = 1.0 if now.hour >= 22 or now.hour < 6 else 0.0
            
            # 季节特征
            season_map = {
                1: 0.0, 2: 0.0, 3: 0.25,  # 冬季、春季
                4: 0.25, 5: 0.25, 6: 0.5,  # 春季、夏季
                7: 0.5, 8: 0.5, 9: 0.75,  # 夏季、秋季
                10: 0.75, 11: 0.75, 12: 0.0  # 秋季、冬季
            }
            features["season"] = season_map.get(now.month, 0.0)
            
            # 时间周期特征（使用三角函数编码周期性）
            features["hour_sin"] = np.sin(2 * np.pi * now.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * now.hour / 24)
            features["day_sin"] = np.sin(2 * np.pi * now.weekday() / 7)
            features["day_cos"] = np.cos(2 * np.pi * now.weekday() / 7)
            features["month_sin"] = np.sin(2 * np.pi * now.month / 12)
            features["month_cos"] = np.cos(2 * np.pi * now.month / 12)
            
            # 从上下文中提取时间相关特征
            if "timestamp" in context:
                event_time = datetime.fromisoformat(context["timestamp"])
                time_diff = (now - event_time).total_seconds()
                features["time_since_event"] = min(time_diff / 3600.0, 24.0) / 24.0  # 归一化到[0,1]
            
            if "session_start" in context:
                session_start = datetime.fromisoformat(context["session_start"])
                session_duration = (now - session_start).total_seconds()
                features["session_duration"] = min(session_duration / 3600.0, 2.0) / 2.0  # 归一化到[0,1]
                
        except Exception as e:
            logger.error(f"时间特征提取失败: {e}")
            
        return features
    
    async def _extract_behavioral_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取行为特征"""
        features = {}
        
        try:
            # 从上下文提取行为历史
            behavior_history = context.get("behavior_history", [])
            
            if behavior_history:
                # 行为频率特征
                action_counts = Counter([b.get("action") for b in behavior_history])
                total_actions = sum(action_counts.values())
                
                # 常见行为类型的频率
                common_actions = ["view", "click", "purchase", "like", "share", "comment"]
                for action in common_actions:
                    features[f"action_{action}_freq"] = action_counts.get(action, 0) / max(total_actions, 1)
                
                # 行为多样性
                features["action_diversity"] = len(action_counts) / max(len(common_actions), 1)
                
                # 最近行为的时间间隔
                if len(behavior_history) > 1:
                    timestamps = [datetime.fromisoformat(b["timestamp"]) for b in behavior_history[:10]]
                    time_diffs = [(timestamps[i] - timestamps[i+1]).total_seconds() 
                                 for i in range(len(timestamps)-1)]
                    features["avg_action_interval"] = np.mean(time_diffs) / 3600.0 if time_diffs else 0.0
                    features["std_action_interval"] = np.std(time_diffs) / 3600.0 if time_diffs else 0.0
                
                # 行为序列特征
                recent_actions = [b.get("action") for b in behavior_history[:5]]
                action_sequence = "_".join(recent_actions)
                features["has_purchase_sequence"] = 1.0 if "view_click_purchase" in action_sequence else 0.0
                features["has_engagement_sequence"] = 1.0 if any(x in action_sequence for x in ["like", "share", "comment"]) else 0.0
            
            # 当前行为特征
            current_action = context.get("current_action", "")
            if current_action:
                features[f"current_action_{current_action}"] = 1.0
            
            # 用户活跃度特征
            if "activity_level" in context:
                features["activity_level"] = context["activity_level"]
            
            # 用户偏好特征
            preferences = context.get("preferences", {})
            for pref_key, pref_value in preferences.items():
                if isinstance(pref_value, (int, float)):
                    features[f"pref_{pref_key}"] = float(pref_value)
                    
        except Exception as e:
            logger.error(f"行为特征提取失败: {e}")
            
        return features
    
    async def _extract_contextual_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取上下文特征"""
        features = {}
        
        try:
            # 设备特征
            device_info = context.get("device", {})
            if device_info:
                features["is_mobile"] = 1.0 if device_info.get("type") == "mobile" else 0.0
                features["is_desktop"] = 1.0 if device_info.get("type") == "desktop" else 0.0
                features["is_tablet"] = 1.0 if device_info.get("type") == "tablet" else 0.0
                
                # 操作系统特征
                os_map = {"ios": 0.2, "android": 0.4, "windows": 0.6, "macos": 0.8, "linux": 1.0}
                features["os_type"] = os_map.get(device_info.get("os", "").lower(), 0.0)
                
                # 浏览器特征
                browser_map = {"chrome": 0.2, "safari": 0.4, "firefox": 0.6, "edge": 0.8}
                features["browser_type"] = browser_map.get(device_info.get("browser", "").lower(), 0.0)
            
            # 位置特征
            location = context.get("location", {})
            if location:
                # 地理特征
                features["latitude_norm"] = (location.get("latitude", 0) + 90) / 180.0  # 归一化纬度
                features["longitude_norm"] = (location.get("longitude", 0) + 180) / 360.0  # 归一化经度
                
                # 时区特征
                timezone_offset = location.get("timezone_offset", 0)
                features["timezone_offset"] = (timezone_offset + 12) / 24.0  # 归一化时区偏移
                
                # 国家/地区特征（可以使用one-hot编码或embedding）
                country = location.get("country", "")
                if country:
                    # 简化：使用哈希值作为特征
                    features["country_hash"] = abs(hash(country)) % 100 / 100.0
            
            # 网络特征
            network_info = context.get("network", {})
            if network_info:
                features["is_wifi"] = 1.0 if network_info.get("type") == "wifi" else 0.0
                features["is_cellular"] = 1.0 if network_info.get("type") == "cellular" else 0.0
                
                # 连接速度特征
                speed_map = {"slow": 0.2, "3g": 0.4, "4g": 0.6, "5g": 0.8, "broadband": 1.0}
                features["connection_speed"] = speed_map.get(network_info.get("speed", ""), 0.5)
            
            # 会话特征
            session_info = context.get("session", {})
            if session_info:
                features["page_views"] = min(session_info.get("page_views", 0) / 100.0, 1.0)
                features["session_depth"] = min(session_info.get("depth", 0) / 20.0, 1.0)
                features["bounce_rate"] = session_info.get("bounce_rate", 0.0)
                features["is_returning"] = 1.0 if session_info.get("is_returning", False) else 0.0
            
            # 推荐场景特征
            scenario = context.get("scenario", "")
            if scenario:
                scenario_map = {"homepage": 0.2, "search": 0.4, "category": 0.6, "detail": 0.8, "checkout": 1.0}
                features["scenario_type"] = scenario_map.get(scenario, 0.5)
            
            # 外部因素特征
            external = context.get("external", {})
            if external:
                # 天气特征
                weather = external.get("weather", {})
                if weather:
                    features["temperature"] = (weather.get("temperature", 20) + 20) / 60.0  # 归一化温度
                    features["humidity"] = weather.get("humidity", 0.5)
                    features["is_rainy"] = 1.0 if weather.get("condition") == "rain" else 0.0
                    features["is_sunny"] = 1.0 if weather.get("condition") == "sunny" else 0.0
                
                # 事件特征
                events = external.get("events", [])
                if events:
                    features["has_holiday"] = 1.0 if "holiday" in events else 0.0
                    features["has_sale"] = 1.0 if "sale" in events else 0.0
                    features["has_special_event"] = 1.0 if any(e in events for e in ["festival", "sports", "concert"]) else 0.0
                    
        except Exception as e:
            logger.error(f"上下文特征提取失败: {e}")
            
        return features
    
    def register_custom_extractor(self, name: str, extractor_func):
        """注册自定义特征提取器
        
        Args:
            name: 提取器名称
            extractor_func: 提取器函数
        """
        setattr(self, f"_extract_{name}_features", extractor_func)
        logger.info(f"注册自定义特征提取器: {name}")

class SequentialFeatureExtractor(FeatureExtractor):
    """序列特征提取器"""
    
    def __init__(self):
        super().__init__("sequential")
        self.sequence_length = 10
        
    async def extract_sequence_features(
        self,
        user_id: str,
        sequence: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """提取序列特征
        
        Args:
            user_id: 用户ID
            sequence: 事件序列
            
        Returns:
            Dict[str, float]: 序列特征
        """
        features = {}
        
        try:
            if not sequence:
                return features
            
            # 限制序列长度
            sequence = sequence[:self.sequence_length]
            
            # 序列长度特征
            features["sequence_length"] = len(sequence) / self.sequence_length
            
            # 行为转换特征
            transitions = {}
            for i in range(len(sequence) - 1):
                current_action = sequence[i].get("action", "")
                next_action = sequence[i + 1].get("action", "")
                transition = f"{current_action}_to_{next_action}"
                transitions[transition] = transitions.get(transition, 0) + 1
            
            # 添加转换频率特征
            for transition, count in transitions.items():
                features[f"transition_{transition}"] = count / max(len(sequence) - 1, 1)
            
            # 时间间隔特征
            if len(sequence) > 1:
                time_intervals = []
                for i in range(len(sequence) - 1):
                    if "timestamp" in sequence[i] and "timestamp" in sequence[i + 1]:
                        t1 = datetime.fromisoformat(sequence[i]["timestamp"])
                        t2 = datetime.fromisoformat(sequence[i + 1]["timestamp"])
                        interval = (t2 - t1).total_seconds()
                        time_intervals.append(interval)
                
                if time_intervals:
                    features["avg_time_interval"] = np.mean(time_intervals) / 3600.0
                    features["std_time_interval"] = np.std(time_intervals) / 3600.0
                    features["min_time_interval"] = np.min(time_intervals) / 3600.0
                    features["max_time_interval"] = np.max(time_intervals) / 3600.0
            
            # 行为模式特征
            action_pattern = [s.get("action", "") for s in sequence]
            features["has_repetitive_pattern"] = 1.0 if len(set(action_pattern)) < len(action_pattern) * 0.5 else 0.0
            features["action_variety"] = len(set(action_pattern)) / max(len(action_pattern), 1)
            
        except Exception as e:
            logger.error(f"序列特征提取失败: {e}")
            
        return features
