"""
情感状态建模系统数据访问层
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import logging

from ..db.models import User
from ..ai.emotion_modeling.models import (
    EmotionState, PersonalityProfile, EmotionTransition, 
    EmotionPrediction, EmotionStatistics
)

logger = logging.getLogger(__name__)


class EmotionModelingRepository:
    """情感建模数据访问层"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ===== EmotionState 相关操作 =====
    
    async def save_emotion_state(self, state: EmotionState) -> bool:
        """保存情感状态"""
        try:
            query = """
            INSERT INTO emotion_states 
            (user_id, emotion, intensity, valence, arousal, dominance, confidence,
             timestamp, duration, triggers, context, source, session_id)
            VALUES (%(user_id)s, %(emotion)s, %(intensity)s, %(valence)s, %(arousal)s, 
                   %(dominance)s, %(confidence)s, %(timestamp)s, %(duration)s, 
                   %(triggers)s, %(context)s, %(source)s, %(session_id)s)
            RETURNING id
            """
            
            params = {
                'user_id': state.user_id,
                'emotion': state.emotion,
                'intensity': state.intensity,
                'valence': state.valence,
                'arousal': state.arousal,
                'dominance': state.dominance,
                'confidence': state.confidence,
                'timestamp': state.timestamp,
                'duration': state.duration,
                'triggers': state.triggers,
                'context': state.context,
                'source': state.source,
                'session_id': state.session_id
            }
            
            result = self.db.execute(query, params)
            state_id = result.fetchone()
            if state_id:
                state.id = str(state_id[0])
                self.db.commit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"保存情感状态失败: {e}")
            self.db.rollback()
            return False
    
    async def get_user_emotion_history(
        self, 
        user_id: str, 
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        emotions: Optional[List[str]] = None
    ) -> List[EmotionState]:
        """获取用户情感历史"""
        try:
            query = """
            SELECT user_id, emotion, intensity, valence, arousal, dominance, confidence,
                   timestamp, duration, triggers, context, source, session_id, id
            FROM emotion_states 
            WHERE user_id = %(user_id)s
            """
            params = {'user_id': user_id}
            
            if start_time:
                query += " AND timestamp >= %(start_time)s"
                params['start_time'] = start_time
                
            if end_time:
                query += " AND timestamp <= %(end_time)s" 
                params['end_time'] = end_time
                
            if emotions:
                placeholders = ','.join(f"%(emotion_{i})s" for i in range(len(emotions)))
                query += f" AND emotion IN ({placeholders})"
                for i, emotion in enumerate(emotions):
                    params[f'emotion_{i}'] = emotion
            
            query += " ORDER BY timestamp DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            result = self.db.execute(query, params)
            rows = result.fetchall()
            
            states = []
            for row in rows:
                state = EmotionState(
                    id=str(row.id),
                    user_id=row.user_id,
                    emotion=row.emotion,
                    intensity=row.intensity,
                    valence=row.valence,
                    arousal=row.arousal,
                    dominance=row.dominance,
                    confidence=row.confidence,
                    timestamp=row.timestamp,
                    duration=row.duration,
                    triggers=row.triggers or [],
                    context=row.context or {},
                    source=row.source,
                    session_id=row.session_id
                )
                states.append(state)
            
            return states
            
        except Exception as e:
            logger.error(f"获取用户情感历史失败: {e}")
            return []
    
    async def get_latest_emotion_state(self, user_id: str) -> Optional[EmotionState]:
        """获取用户最新情感状态"""
        history = await self.get_user_emotion_history(user_id, limit=1)
        return history[0] if history else None
    
    async def get_emotion_states_by_session(self, session_id: str) -> List[EmotionState]:
        """根据会话ID获取情感状态"""
        return await self.get_user_emotion_history(
            user_id="",  # 先查询所有用户
            limit=1000
        )  # 这里需要修改查询逻辑
    
    # ===== PersonalityProfile 相关操作 =====
    
    async def save_personality_profile(self, profile: PersonalityProfile) -> bool:
        """保存个性画像"""
        try:
            # 使用 UPSERT 逻辑
            query = """
            INSERT INTO personality_profiles 
            (user_id, emotional_traits, baseline_emotions, emotion_volatility, recovery_rate,
             dominant_emotions, trigger_patterns, created_at, updated_at, sample_count, confidence_score)
            VALUES (%(user_id)s, %(emotional_traits)s, %(baseline_emotions)s, %(emotion_volatility)s, 
                   %(recovery_rate)s, %(dominant_emotions)s, %(trigger_patterns)s, %(created_at)s,
                   %(updated_at)s, %(sample_count)s, %(confidence_score)s)
            ON CONFLICT (user_id) 
            DO UPDATE SET
                emotional_traits = %(emotional_traits)s,
                baseline_emotions = %(baseline_emotions)s,
                emotion_volatility = %(emotion_volatility)s,
                recovery_rate = %(recovery_rate)s,
                dominant_emotions = %(dominant_emotions)s,
                trigger_patterns = %(trigger_patterns)s,
                updated_at = %(updated_at)s,
                sample_count = %(sample_count)s,
                confidence_score = %(confidence_score)s
            RETURNING id
            """
            
            params = {
                'user_id': profile.user_id,
                'emotional_traits': profile.emotional_traits,
                'baseline_emotions': profile.baseline_emotions,
                'emotion_volatility': profile.emotion_volatility,
                'recovery_rate': profile.recovery_rate,
                'dominant_emotions': profile.dominant_emotions,
                'trigger_patterns': profile.trigger_patterns,
                'created_at': profile.created_at,
                'updated_at': profile.updated_at,
                'sample_count': profile.sample_count,
                'confidence_score': profile.confidence_score
            }
            
            result = self.db.execute(query, params)
            profile_id = result.fetchone()
            if profile_id:
                profile.id = str(profile_id[0])
                self.db.commit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"保存个性画像失败: {e}")
            self.db.rollback()
            return False
    
    async def get_personality_profile(self, user_id: str) -> Optional[PersonalityProfile]:
        """获取用户个性画像"""
        try:
            query = """
            SELECT id, user_id, emotional_traits, baseline_emotions, emotion_volatility, 
                   recovery_rate, dominant_emotions, trigger_patterns, created_at, updated_at,
                   sample_count, confidence_score
            FROM personality_profiles 
            WHERE user_id = %(user_id)s
            """
            
            result = self.db.execute(query, {'user_id': user_id})
            row = result.fetchone()
            
            if row:
                return PersonalityProfile(
                    id=str(row.id),
                    user_id=row.user_id,
                    emotional_traits=row.emotional_traits,
                    baseline_emotions=row.baseline_emotions,
                    emotion_volatility=row.emotion_volatility,
                    recovery_rate=row.recovery_rate,
                    dominant_emotions=row.dominant_emotions,
                    trigger_patterns=row.trigger_patterns,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    sample_count=row.sample_count,
                    confidence_score=row.confidence_score
                )
            return None
            
        except Exception as e:
            logger.error(f"获取个性画像失败: {e}")
            return None
    
    # ===== EmotionTransition 相关操作 =====
    
    async def save_emotion_transition(self, transition: EmotionTransition) -> bool:
        """保存情感转换记录"""
        try:
            # 使用 UPSERT 逻辑，基于 (user_id, from_emotion, to_emotion) 更新
            query = """
            INSERT INTO emotion_transitions 
            (user_id, from_emotion, to_emotion, transition_probability, occurrence_count,
             avg_duration, updated_at, context_factors)
            VALUES (%(user_id)s, %(from_emotion)s, %(to_emotion)s, %(transition_probability)s, 
                   %(occurrence_count)s, %(avg_duration)s, %(updated_at)s, %(context_factors)s)
            ON CONFLICT ON CONSTRAINT emotion_transitions_user_from_to_unique
            DO UPDATE SET
                transition_probability = %(transition_probability)s,
                occurrence_count = %(occurrence_count)s,
                avg_duration = %(avg_duration)s,
                updated_at = %(updated_at)s,
                context_factors = %(context_factors)s
            RETURNING id
            """
            
            params = {
                'user_id': transition.user_id,
                'from_emotion': transition.from_emotion,
                'to_emotion': transition.to_emotion,
                'transition_probability': transition.transition_probability,
                'occurrence_count': transition.occurrence_count,
                'avg_duration': transition.avg_duration,
                'updated_at': transition.updated_at,
                'context_factors': transition.context_factors
            }
            
            result = self.db.execute(query, params)
            transition_id = result.fetchone()
            if transition_id:
                transition.id = str(transition_id[0])
                self.db.commit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"保存情感转换记录失败: {e}")
            self.db.rollback()
            return False
    
    async def get_user_transitions(self, user_id: str) -> List[EmotionTransition]:
        """获取用户的情感转换记录"""
        try:
            query = """
            SELECT id, user_id, from_emotion, to_emotion, transition_probability, 
                   occurrence_count, avg_duration, updated_at, context_factors
            FROM emotion_transitions 
            WHERE user_id = %(user_id)s
            ORDER BY occurrence_count DESC
            """
            
            result = self.db.execute(query, {'user_id': user_id})
            rows = result.fetchall()
            
            transitions = []
            for row in rows:
                transition = EmotionTransition(
                    id=str(row.id),
                    user_id=row.user_id,
                    from_emotion=row.from_emotion,
                    to_emotion=row.to_emotion,
                    transition_probability=row.transition_probability,
                    occurrence_count=row.occurrence_count,
                    avg_duration=row.avg_duration,
                    updated_at=row.updated_at,
                    context_factors=row.context_factors or []
                )
                transitions.append(transition)
            
            return transitions
            
        except Exception as e:
            logger.error(f"获取用户情感转换记录失败: {e}")
            return []
    
    # ===== 预测相关操作 =====
    
    async def save_emotion_prediction(
        self, 
        user_id: str,
        prediction: EmotionPrediction
    ) -> bool:
        """保存情感预测"""
        try:
            query = """
            INSERT INTO emotion_predictions 
            (user_id, current_emotion, predicted_emotions, confidence, time_horizon_seconds,
             prediction_time, factors)
            VALUES (%(user_id)s, %(current_emotion)s, %(predicted_emotions)s, %(confidence)s, 
                   %(time_horizon_seconds)s, %(prediction_time)s, %(factors)s)
            RETURNING id
            """
            
            params = {
                'user_id': user_id,
                'current_emotion': prediction.current_emotion,
                'predicted_emotions': prediction.predicted_emotions,
                'confidence': prediction.confidence,
                'time_horizon_seconds': int(prediction.time_horizon.total_seconds()),
                'prediction_time': prediction.prediction_time,
                'factors': prediction.factors
            }
            
            result = self.db.execute(query, params)
            prediction_id = result.fetchone()
            if prediction_id:
                self.db.commit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"保存情感预测失败: {e}")
            self.db.rollback()
            return False
    
    # ===== 统计查询 =====
    
    async def get_emotion_statistics(
        self, 
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> EmotionStatistics:
        """获取情感统计信息"""
        try:
            # 基础统计查询
            query = """
            SELECT 
                emotion,
                COUNT(*) as count,
                AVG(intensity) as avg_intensity,
                AVG(valence) as avg_valence,
                AVG(arousal) as avg_arousal,
                AVG(dominance) as avg_dominance
            FROM emotion_states
            WHERE user_id = %(user_id)s 
              AND timestamp >= %(start_time)s 
              AND timestamp <= %(end_time)s
            GROUP BY emotion
            ORDER BY count DESC
            """
            
            params = {
                'user_id': user_id,
                'start_time': start_time,
                'end_time': end_time
            }
            
            result = self.db.execute(query, params)
            rows = result.fetchall()
            
            # 构建统计对象
            stats = EmotionStatistics(
                user_id=user_id,
                time_period=(start_time, end_time)
            )
            
            total_count = sum(row.count for row in rows)
            stats.total_samples = total_count
            
            if total_count > 0:
                # 情感分布
                stats.emotion_distribution = {
                    row.emotion: row.count / total_count for row in rows
                }
                
                # 各维度统计
                stats.intensity_stats = {
                    'mean': sum(row.avg_intensity * (row.count / total_count) for row in rows),
                    'emotions': {row.emotion: row.avg_intensity for row in rows}
                }
                
                stats.valence_stats = {
                    'mean': sum(row.avg_valence * (row.count / total_count) for row in rows),
                    'emotions': {row.emotion: row.avg_valence for row in rows}
                }
                
                stats.arousal_stats = {
                    'mean': sum(row.avg_arousal * (row.count / total_count) for row in rows),
                    'emotions': {row.emotion: row.avg_arousal for row in rows}
                }
                
                stats.dominance_stats = {
                    'mean': sum(row.avg_dominance * (row.count / total_count) for row in rows),
                    'emotions': {row.emotion: row.avg_dominance for row in rows}
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取情感统计失败: {e}")
            return EmotionStatistics(user_id=user_id, time_period=(start_time, end_time))
    
    async def count_user_emotion_records(self, user_id: str) -> int:
        """统计用户情感记录数量"""
        try:
            query = "SELECT COUNT(*) FROM emotion_states WHERE user_id = %(user_id)s"
            result = self.db.execute(query, {'user_id': user_id})
            count = result.fetchone()
            return count[0] if count else 0
        except Exception as e:
            logger.error(f"统计用户情感记录失败: {e}")
            return 0