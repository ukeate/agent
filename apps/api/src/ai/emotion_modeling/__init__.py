"""
情感状态建模系统

该模块提供了完整的情感状态建模功能，包括：
- 多维情感空间建模(VAD空间)
- 个性化情感画像构建
- 情感状态转换建模
- 动态跟踪和预测
- 统计分析和模式识别
- 群体情感动态分析
- 社交情感理解系统
"""

from .group_emotion_models import (
    EmotionState,
    GroupEmotionalState,
    GroupEmotionHistory,
    EmotionContagionEvent,
    ContagionPattern,
    EmotionalLeader,
    GroupEmotionInsight,
    GroupEmotionAnalysisConfig,
    EmotionContagionType,
    GroupCohesionLevel,
    generate_group_id,
    generate_event_id
)

__all__ = [
    # Group Emotion Models
    'EmotionState',
    'GroupEmotionalState', 
    'GroupEmotionHistory',
    'EmotionContagionEvent',
    'ContagionPattern',
    'EmotionalLeader',
    'GroupEmotionInsight',
    'GroupEmotionAnalysisConfig',
    
    # Enums
    'EmotionContagionType',
    'GroupCohesionLevel',
    
    # Utilities
    'generate_group_id',
    'generate_event_id'
]