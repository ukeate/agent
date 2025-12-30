"""
情感识别结果统一格式化器
支持多种输出格式和标准化处理
"""

import json
import yaml
import csv
import io
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from .core_interfaces import (
    EmotionType, ModalityType, EmotionState, MultiModalEmotion,
    UnifiedEmotionalData, EmotionalIntelligenceResponse,
    PersonalityProfile, EmpathyResponse, EmotionalMemory,
    DecisionContext, RiskAssessment, SocialContext, GroupEmotionalState
)

from src.core.logging import get_logger
logger = get_logger(__name__)

class OutputFormat(str, Enum):
    """输出格式"""
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    XML = "xml"
    PROTOBUF = "protobuf"
    PANDAS = "pandas"
    NUMPY = "numpy"
    CUSTOM = "custom"

class CompressionType(str, Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    SNAPPY = "snappy"

class ValidationLevel(str, Enum):
    """验证级别"""
    STRICT = "strict"        # 严格验证
    NORMAL = "normal"        # 标准验证
    LENIENT = "lenient"      # 宽松验证
    DISABLED = "disabled"    # 禁用验证

@dataclass
class FormattingConfig:
    """格式化配置"""
    output_format: OutputFormat = OutputFormat.JSON
    include_metadata: bool = True
    include_raw_data: bool = False
    precision: int = 4           # 数值精度
    timestamp_format: str = "iso" # iso, unix, custom
    compression: CompressionType = CompressionType.NONE
    validation_level: ValidationLevel = ValidationLevel.NORMAL
    custom_fields: Optional[Dict[str, Any]] = None
    field_mapping: Optional[Dict[str, str]] = None  # 字段重映射

class EmotionDataNormalizer:
    """情感数据标准化器"""
    
    def __init__(self):
        self.emotion_mappings = self._init_emotion_mappings()
        self.intensity_scales = self._init_intensity_scales()
    
    def _init_emotion_mappings(self) -> Dict[str, EmotionType]:
        """初始化情感映射"""
        return {
            # 英文映射
            "happy": EmotionType.HAPPINESS,
            "joy": EmotionType.HAPPINESS,
            "pleased": EmotionType.HAPPINESS,
            "sad": EmotionType.SADNESS,
            "depressed": EmotionType.SADNESS,
            "melancholy": EmotionType.SADNESS,
            "angry": EmotionType.ANGER,
            "mad": EmotionType.ANGER,
            "furious": EmotionType.ANGER,
            "afraid": EmotionType.FEAR,
            "scared": EmotionType.FEAR,
            "terrified": EmotionType.FEAR,
            "surprised": EmotionType.SURPRISE,
            "shocked": EmotionType.SURPRISE,
            "amazed": EmotionType.SURPRISE,
            "disgusted": EmotionType.DISGUST,
            "revolted": EmotionType.DISGUST,
            "neutral": EmotionType.NEUTRAL,
            "calm": EmotionType.NEUTRAL,
            
            # 中文映射
            "高兴": EmotionType.HAPPINESS,
            "快乐": EmotionType.HAPPINESS,
            "悲伤": EmotionType.SADNESS,
            "难过": EmotionType.SADNESS,
            "愤怒": EmotionType.ANGER,
            "生气": EmotionType.ANGER,
            "恐惧": EmotionType.FEAR,
            "害怕": EmotionType.FEAR,
            "惊讶": EmotionType.SURPRISE,
            "吃惊": EmotionType.SURPRISE,
            "厌恶": EmotionType.DISGUST,
            "恶心": EmotionType.DISGUST,
            "中性": EmotionType.NEUTRAL,
            "平静": EmotionType.NEUTRAL
        }
    
    def _init_intensity_scales(self) -> Dict[str, Dict[str, float]]:
        """初始化强度量表"""
        return {
            "low_medium_high": {
                "low": 0.33,
                "medium": 0.66,
                "high": 1.0
            },
            "scale_1_5": {
                "1": 0.2, "2": 0.4, "3": 0.6, "4": 0.8, "5": 1.0
            },
            "scale_1_10": {
                str(i): i / 10.0 for i in range(1, 11)
            }
        }
    
    def normalize_emotion_type(self, emotion: Union[str, EmotionType]) -> EmotionType:
        """标准化情感类型"""
        if isinstance(emotion, EmotionType):
            return emotion
        
        emotion_lower = emotion.lower()
        if emotion_lower in self.emotion_mappings:
            return self.emotion_mappings[emotion_lower]
        
        logger.warning(f"Unknown emotion type: {emotion}, defaulting to NEUTRAL")
        return EmotionType.NEUTRAL
    
    def normalize_intensity(self, intensity: Union[float, str, int], scale_type: str = "raw") -> float:
        """标准化强度值"""
        if isinstance(intensity, (int, float)):
            if 0 <= intensity <= 1:
                return float(intensity)
            elif 0 <= intensity <= 10:
                return intensity / 10.0
            elif 0 <= intensity <= 100:
                return intensity / 100.0
            else:
                logger.warning(f"Intensity value {intensity} out of expected range")
                return max(0.0, min(1.0, intensity))
        
        if isinstance(intensity, str):
            intensity_lower = intensity.lower()
            if scale_type in self.intensity_scales:
                scale = self.intensity_scales[scale_type]
                if intensity_lower in scale:
                    return scale[intensity_lower]
            
            # 尝试转换为数字
            try:
                num_intensity = float(intensity)
                return self.normalize_intensity(num_intensity)
            except ValueError:
                logger.warning(f"Cannot parse intensity: {intensity}, defaulting to 0.5")
                return 0.5
        
        logger.warning(f"Unknown intensity type: {type(intensity)}, defaulting to 0.5")
        return 0.5
    
    def normalize_vad_values(self, valence: float, arousal: float, dominance: float) -> tuple:
        """标准化VAD值"""
        def clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
            return max(min_val, min(max_val, value))
        
        # Valence: -1 to 1
        norm_valence = clamp(valence, -1.0, 1.0)
        # Arousal: 0 to 1  
        norm_arousal = clamp(arousal, 0.0, 1.0)
        # Dominance: 0 to 1
        norm_dominance = clamp(dominance, 0.0, 1.0)
        
        return norm_valence, norm_arousal, norm_dominance

class BaseFormatter(ABC):
    """格式化器基类"""
    
    def __init__(self, config: FormattingConfig):
        self.config = config
        self.normalizer = EmotionDataNormalizer()
    
    @abstractmethod
    def format_data(self, data: UnifiedEmotionalData) -> str:
        """格式化数据"""
        ...
    
    @abstractmethod
    def format_response(self, response: EmotionalIntelligenceResponse) -> str:
        """格式化响应"""
        ...
    
    def _prepare_data(self, data: UnifiedEmotionalData) -> Dict[str, Any]:
        """准备数据用于格式化"""
        result = {}
        
        # 基础字段
        result.update({
            "user_id": data.user_id,
            "timestamp": self._format_timestamp(data.timestamp),
            "confidence": round(data.confidence, self.config.precision),
            "processing_time": round(data.processing_time, self.config.precision),
            "data_quality": round(data.data_quality, self.config.precision)
        })
        
        # 情感识别结果
        if data.recognition_result:
            result["recognition"] = self._format_multimodal_emotion(data.recognition_result)
        
        # 情感状态
        if data.emotional_state:
            result["emotional_state"] = self._format_emotion_state(data.emotional_state)
        
        # 个性画像
        if data.personality_profile:
            result["personality"] = self._format_personality_profile(data.personality_profile)
        
        # 共情响应
        if data.empathy_response:
            result["empathy"] = self._format_empathy_response(data.empathy_response)
        
        # 情感记忆
        if data.emotional_memory:
            result["memory"] = self._format_emotional_memory(data.emotional_memory)
            if data.memory_relevance is not None:
                result["memory_relevance"] = round(data.memory_relevance, self.config.precision)
        
        # 决策上下文
        if data.decision_context:
            result["decision"] = self._format_decision_context(data.decision_context)
        
        # 风险评估
        if data.risk_assessment:
            result["risk"] = self._format_risk_assessment(data.risk_assessment)
        
        # 社交上下文
        if data.social_context:
            result["social"] = self._format_social_context(data.social_context)
        
        # 群体情感
        if data.group_emotion:
            result["group"] = self._format_group_emotion(data.group_emotion)
        
        # 自定义字段
        if self.config.custom_fields:
            result.update(self.config.custom_fields)
        
        # 字段重映射
        if self.config.field_mapping:
            result = self._apply_field_mapping(result, self.config.field_mapping)
        
        return result
    
    def _format_timestamp(self, timestamp: datetime) -> Union[str, int, float]:
        """格式化时间戳"""
        if self.config.timestamp_format == "iso":
            return timestamp.isoformat()
        elif self.config.timestamp_format == "unix":
            return timestamp.timestamp()
        else:
            # 自定义格式
            return timestamp.strftime(self.config.timestamp_format)
    
    def _format_emotion_state(self, emotion: EmotionState) -> Dict[str, Any]:
        """格式化情感状态"""
        return {
            "emotion": emotion.emotion.value,
            "intensity": round(emotion.intensity, self.config.precision),
            "valence": round(emotion.valence, self.config.precision),
            "arousal": round(emotion.arousal, self.config.precision),
            "dominance": round(emotion.dominance, self.config.precision),
            "confidence": round(emotion.confidence, self.config.precision),
            "timestamp": self._format_timestamp(emotion.timestamp)
        }
    
    def _format_multimodal_emotion(self, emotion: MultiModalEmotion) -> Dict[str, Any]:
        """格式化多模态情感"""
        result = {
            "fused_emotion": self._format_emotion_state(emotion.fused_emotion),
            "confidence": round(emotion.confidence, self.config.precision),
            "processing_time": round(emotion.processing_time, self.config.precision)
        }
        
        if emotion.emotions:
            result["modalities"] = {
                modality.value: self._format_emotion_state(emotion_state)
                for modality, emotion_state in emotion.emotions.items()
            }
        
        return result
    
    def _format_personality_profile(self, profile: PersonalityProfile) -> Dict[str, Any]:
        """格式化个性画像"""
        return {
            "openness": round(profile.openness, self.config.precision),
            "conscientiousness": round(profile.conscientiousness, self.config.precision),
            "extraversion": round(profile.extraversion, self.config.precision),
            "agreeableness": round(profile.agreeableness, self.config.precision),
            "neuroticism": round(profile.neuroticism, self.config.precision),
            "updated_at": self._format_timestamp(profile.updated_at)
        }
    
    def _format_empathy_response(self, response: EmpathyResponse) -> Dict[str, Any]:
        """格式化共情响应"""
        return {
            "message": response.message,
            "response_type": response.response_type,
            "confidence": round(response.confidence, self.config.precision),
            "generation_strategy": response.generation_strategy
        }
    
    def _format_emotional_memory(self, memory: EmotionalMemory) -> Dict[str, Any]:
        """格式化情感记忆"""
        return {
            "memory_id": memory.memory_id,
            "content": memory.content,
            "emotional_context": self._format_emotion_state(memory.emotional_context),
            "importance": round(memory.importance, self.config.precision),
            "created_at": self._format_timestamp(memory.created_at),
            "last_accessed": self._format_timestamp(memory.last_accessed)
        }
    
    def _format_decision_context(self, context: DecisionContext) -> Dict[str, Any]:
        """格式化决策上下文"""
        return {
            "decision_type": context.decision_type,
            "factors": context.factors,
            "emotional_weight": round(context.emotional_weight, self.config.precision),
            "rational_weight": round(context.rational_weight, self.config.precision)
        }
    
    def _format_risk_assessment(self, risk: RiskAssessment) -> Dict[str, Any]:
        """格式化风险评估"""
        return {
            "level": risk.level.value,
            "factors": risk.factors,
            "confidence": round(risk.confidence, self.config.precision),
            "intervention_needed": risk.intervention_needed,
            "recommendations": risk.recommendations
        }
    
    def _format_social_context(self, context: SocialContext) -> Dict[str, Any]:
        """格式化社交上下文"""
        return {
            "participants": context.participants,
            "relationship_dynamics": {
                k: round(v, self.config.precision) 
                for k, v in context.relationship_dynamics.items()
            },
            "cultural_factors": context.cultural_factors,
            "communication_style": context.communication_style
        }
    
    def _format_group_emotion(self, group: GroupEmotionalState) -> Dict[str, Any]:
        """格式化群体情感"""
        return {
            "group_emotion": self._format_emotion_state(group.group_emotion),
            "individual_emotions": {
                k: self._format_emotion_state(v)
                for k, v in group.individual_emotions.items()
            },
            "consensus_level": round(group.consensus_level, self.config.precision),
            "conflict_indicators": group.conflict_indicators
        }
    
    def _apply_field_mapping(self, data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """应用字段重映射"""
        result = {}
        for key, value in data.items():
            mapped_key = mapping.get(key, key)
            result[mapped_key] = value
        return result

class JSONFormatter(BaseFormatter):
    """JSON格式化器"""
    
    def format_data(self, data: UnifiedEmotionalData) -> str:
        prepared_data = self._prepare_data(data)
        return json.dumps(prepared_data, ensure_ascii=False, indent=2)
    
    def format_response(self, response: EmotionalIntelligenceResponse) -> str:
        result = {
            "success": response.success,
            "metadata": response.metadata
        }
        
        if response.data:
            result["data"] = self._prepare_data(response.data)
        
        if response.error:
            result["error"] = response.error
        
        return json.dumps(result, ensure_ascii=False, indent=2)

class YAMLFormatter(BaseFormatter):
    """YAML格式化器"""
    
    def format_data(self, data: UnifiedEmotionalData) -> str:
        prepared_data = self._prepare_data(data)
        return yaml.dump(prepared_data, default_flow_style=False, allow_unicode=True)
    
    def format_response(self, response: EmotionalIntelligenceResponse) -> str:
        result = {
            "success": response.success,
            "metadata": response.metadata
        }
        
        if response.data:
            result["data"] = self._prepare_data(response.data)
        
        if response.error:
            result["error"] = response.error
        
        return yaml.dump(result, default_flow_style=False, allow_unicode=True)

class CSVFormatter(BaseFormatter):
    """CSV格式化器"""
    
    def format_data(self, data: UnifiedEmotionalData) -> str:
        prepared_data = self._prepare_data(data)
        flattened = self._flatten_dict(prepared_data)
        
        output = io.StringIO()
        if flattened:
            writer = csv.DictWriter(output, fieldnames=flattened.keys())
            writer.writeheader()
            writer.writerow(flattened)
        
        return output.getvalue()
    
    def format_response(self, response: EmotionalIntelligenceResponse) -> str:
        result = {
            "success": response.success,
            "metadata": json.dumps(response.metadata)
        }
        
        if response.data:
            data_dict = self._prepare_data(response.data)
            result.update(self._flatten_dict(data_dict, prefix="data_"))
        
        if response.error:
            result["error"] = json.dumps(response.error)
        
        output = io.StringIO()
        if result:
            writer = csv.DictWriter(output, fieldnames=result.keys())
            writer.writeheader()
            writer.writerow(result)
        
        return output.getvalue()
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """扁平化字典"""
        result = {}
        for key, value in d.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(self._flatten_dict(value, f"{new_key}_"))
            elif isinstance(value, list):
                result[new_key] = json.dumps(value)
            else:
                result[new_key] = value
        
        return result

class XMLFormatter(BaseFormatter):
    """XML格式化器"""
    
    def format_data(self, data: UnifiedEmotionalData) -> str:
        prepared_data = self._prepare_data(data)
        return self._dict_to_xml(prepared_data, "EmotionalData")
    
    def format_response(self, response: EmotionalIntelligenceResponse) -> str:
        result = {
            "success": response.success,
            "metadata": response.metadata
        }
        
        if response.data:
            result["data"] = self._prepare_data(response.data)
        
        if response.error:
            result["error"] = response.error
        
        return self._dict_to_xml(result, "EmotionalResponse")
    
    def _dict_to_xml(self, data: Dict[str, Any], root_name: str) -> str:
        """字典转XML"""
        def build_xml(obj, name):
            if isinstance(obj, dict):
                xml = f"<{name}>"
                for key, value in obj.items():
                    xml += build_xml(value, key)
                xml += f"</{name}>"
                return xml
            elif isinstance(obj, list):
                xml = f"<{name}>"
                for i, item in enumerate(obj):
                    xml += build_xml(item, f"item{i}")
                xml += f"</{name}>"
                return xml
            else:
                return f"<{name}>{self._escape_xml(str(obj))}</{name}>"
        
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{build_xml(data, root_name)}'
    
    def _escape_xml(self, text: str) -> str:
        """转义XML特殊字符"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#39;"))

class PandasFormatter(BaseFormatter):
    """Pandas DataFrame格式化器"""
    
    def format_data(self, data: UnifiedEmotionalData) -> pd.DataFrame:
        prepared_data = self._prepare_data(data)
        flattened = self._flatten_dict(prepared_data)
        return pd.DataFrame([flattened])
    
    def format_response(self, response: EmotionalIntelligenceResponse) -> pd.DataFrame:
        result = {
            "success": response.success,
            "metadata": json.dumps(response.metadata)
        }
        
        if response.data:
            data_dict = self._prepare_data(response.data)
            result.update(self._flatten_dict(data_dict, prefix="data_"))
        
        if response.error:
            result["error"] = json.dumps(response.error)
        
        return pd.DataFrame([result])
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """扁平化字典"""
        result = {}
        for key, value in d.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(self._flatten_dict(value, f"{new_key}_"))
            elif isinstance(value, list):
                result[new_key] = json.dumps(value)
            else:
                result[new_key] = value
        
        return result

class ResultFormatterManager:
    """结果格式化管理器"""
    
    def __init__(self):
        self.formatters: Dict[OutputFormat, BaseFormatter] = {}
        self.default_config = FormattingConfig()
    
    def register_formatter(self, format_type: OutputFormat, formatter: BaseFormatter):
        """注册格式化器"""
        self.formatters[format_type] = formatter
    
    def get_formatter(self, format_type: OutputFormat, config: Optional[FormattingConfig] = None) -> BaseFormatter:
        """获取格式化器"""
        if format_type in self.formatters:
            return self.formatters[format_type]
        
        # 创建内置格式化器
        config = config or self.default_config
        
        if format_type == OutputFormat.JSON:
            return JSONFormatter(config)
        elif format_type == OutputFormat.YAML:
            return YAMLFormatter(config)
        elif format_type == OutputFormat.CSV:
            return CSVFormatter(config)
        elif format_type == OutputFormat.XML:
            return XMLFormatter(config)
        elif format_type == OutputFormat.PANDAS:
            return PandasFormatter(config)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_data(
        self,
        data: UnifiedEmotionalData,
        format_type: OutputFormat = OutputFormat.JSON,
        config: Optional[FormattingConfig] = None
    ) -> str:
        """格式化数据"""
        formatter = self.get_formatter(format_type, config)
        return formatter.format_data(data)
    
    def format_response(
        self,
        response: EmotionalIntelligenceResponse,
        format_type: OutputFormat = OutputFormat.JSON,
        config: Optional[FormattingConfig] = None
    ) -> str:
        """格式化响应"""
        formatter = self.get_formatter(format_type, config)
        return formatter.format_response(response)
    
    def format_batch(
        self,
        data_list: List[UnifiedEmotionalData],
        format_type: OutputFormat = OutputFormat.JSON,
        config: Optional[FormattingConfig] = None
    ) -> str:
        """批量格式化数据"""
        formatter = self.get_formatter(format_type, config)
        
        if format_type == OutputFormat.PANDAS:
            dfs = []
            for data in data_list:
                df = formatter.format_data(data)
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            return combined_df.to_json(orient='records', indent=2)
        
        elif format_type == OutputFormat.CSV:
            output = io.StringIO()
            writer = None
            
            for i, data in enumerate(data_list):
                prepared = formatter._prepare_data(data)
                flattened = formatter._flatten_dict(prepared)
                
                if i == 0:
                    writer = csv.DictWriter(output, fieldnames=flattened.keys())
                    writer.writeheader()
                
                writer.writerow(flattened)
            
            return output.getvalue()
        
        else:
            # JSON, YAML, XML等格式
            results = []
            for data in data_list:
                prepared = formatter._prepare_data(data)
                results.append(prepared)
            
            if format_type == OutputFormat.JSON:
                return json.dumps(results, ensure_ascii=False, indent=2)
            elif format_type == OutputFormat.YAML:
                return yaml.dump(results, default_flow_style=False, allow_unicode=True)
            else:
                return formatter._dict_to_xml({"items": results}, "EmotionalDataBatch")
    
    def validate_format_config(self, config: FormattingConfig) -> List[str]:
        """验证格式化配置"""
        errors = []
        
        if config.precision < 0 or config.precision > 10:
            errors.append("Precision must be between 0 and 10")
        
        if config.timestamp_format not in ["iso", "unix"] and not isinstance(config.timestamp_format, str):
            errors.append("Invalid timestamp format")
        
        if config.field_mapping:
            for key, value in config.field_mapping.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    errors.append(f"Invalid field mapping: {key} -> {value}")
        
        return errors

# 全局格式化管理器实例
result_formatter_manager = ResultFormatterManager()
