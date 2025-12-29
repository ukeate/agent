"""
社交情感理解系统API端点 - Story 11.6
提供完整的社交情感理解功能API接口
"""

from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import Field
import json
import asyncio
from datetime import timedelta
from collections import defaultdict
import uuid
from src.ai.emotion_modeling.group_emotion_analyzer import GroupEmotionAnalyzer
from src.ai.emotion_modeling.relationship_analyzer import RelationshipDynamicsAnalyzer
from src.ai.emotion_modeling.social_context_adapter import (
    SocialIntelligenceEngine, DecisionContext, DecisionType
)
from src.api.base_model import ApiBaseModel
from src.ai.emotion_modeling.cultural_context_analyzer import CulturalContextAnalyzer
from src.ai.emotion_modeling.models import EmotionVector, SocialContext
from src.ai.emotion_modeling.group_emotion_models import EmotionState as GroupEmotionState
from src.core.monitoring import get_monitoring_service

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/social-emotional-understanding", tags=["Social Emotional Understanding"])

# 全局实例
group_analyzer = GroupEmotionAnalyzer()
relationship_analyzer = RelationshipDynamicsAnalyzer()
context_adapter = SocialContextAdapter()
cultural_analyzer = CulturalContextAnalyzer()
intelligence_engine = SocialIntelligenceEngine()

# WebSocket连接管理
active_connections: Dict[str, WebSocket] = {}
analysis_records: List[Dict[str, Any]] = []
group_emotion_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
relationship_history_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

def record_analysis(
    session_id: str,
    scenario: str,
    analysis_type: str,
    confidence_score: Optional[float] = None,
    dominant_emotion: Optional[str] = None,
    relationship_health: Optional[float] = None,
    cultural_profiles: Optional[List[str]] = None,
    duration_ms: Optional[float] = None
) -> None:
    analysis_records.append({
        "session_id": session_id,
        "scenario": scenario,
        "analysis_type": analysis_type,
        "confidence_score": confidence_score,
        "dominant_emotion": dominant_emotion,
        "relationship_health": relationship_health,
        "cultural_profiles": cultural_profiles or [],
        "duration_ms": duration_ms,
        "timestamp": utc_now()
    })

# 请求/响应模型
class EmotionData(ApiBaseModel):
    """情感数据"""
    emotions: Dict[str, float]
    intensity: float
    confidence: float
    context: Optional[str] = None

class ParticipantData(ApiBaseModel):
    """参与者数据"""
    participant_id: str
    name: str
    emotion_data: EmotionData
    cultural_indicators: Optional[Dict[str, Any]] = None
    relationship_history: Optional[List[Dict[str, Any]]] = None

class SocialEnvironmentData(ApiBaseModel):
    """社交环境数据"""
    scenario: str
    participants_count: int
    formality_level: float = Field(ge=0.0, le=1.0)
    emotional_intensity: float = Field(ge=0.0, le=1.0)
    time_pressure: float = Field(ge=0.0, le=1.0)
    cultural_context: Optional[str] = None

class AnalysisRequest(ApiBaseModel):
    """分析请求"""
    session_id: str
    participants: List[ParticipantData]
    social_environment: SocialEnvironmentData
    analysis_types: Optional[List[str]] = None
    real_time: bool = False

class GroupEmotionResponse(ApiBaseModel):
    """群体情感分析响应"""
    session_id: str
    group_emotion_state: Dict[str, Any]
    individual_emotions: Dict[str, Dict[str, float]]
    emotional_contagion: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float

class RelationshipAnalysisResponse(ApiBaseModel):
    """关系分析响应"""
    session_id: str
    relationships: List[Dict[str, Any]]
    relationship_dynamics: Dict[str, Any]
    interaction_patterns: Dict[str, Any]
    relationship_health_scores: Dict[str, float]
    recommendations: List[str]

class SocialDecisionResponse(ApiBaseModel):
    """社交决策响应"""
    session_id: str
    decisions: List[Dict[str, Any]]
    execution_priority: List[str]
    monitoring_plan: Dict[str, Any]
    confidence_score: float

# API端点
@router.post("/analyze/group-emotion", response_model=GroupEmotionResponse)
async def analyze_group_emotion(request: AnalysisRequest) -> GroupEmotionResponse:
    """分析群体情感状态"""
    try:
        logger.info(f"Starting group emotion analysis for session {request.session_id}")
        start_time = utc_now()
        
        # 转换参与者数据
        participants_emotions: Dict[str, GroupEmotionState] = {}
        participants_data = {}
        for participant in request.participants:
            emotions = participant.emotion_data.emotions or {}
            dominant_emotion = max(emotions.items(), key=lambda item: item[1])[0] if emotions else "neutral"
            valence = group_analyzer.emotion_polarity.get(dominant_emotion, 0.0)
            participants_emotions[participant.participant_id] = GroupEmotionState(
                participant_id=participant.participant_id,
                emotion=dominant_emotion,
                intensity=participant.emotion_data.intensity,
                valence=valence,
                arousal=max(0.0, min(1.0, participant.emotion_data.intensity)),
                dominance=0.5,
                timestamp=utc_now(),
                confidence=participant.emotion_data.confidence
            )
            participants_data[participant.participant_id] = {
                "name": participant.name,
                "emotion_vector": EmotionVector(
                    emotions=emotions,
                    intensity=participant.emotion_data.intensity,
                    confidence=participant.emotion_data.confidence,
                    context=participant.emotion_data.context
                )
            }
        
        # 执行群体情感分析
        analysis_result = await group_analyzer.analyze_group_emotion(
            participants_emotions,
            group_id=request.session_id
        )
        
        contagion_patterns = analysis_result.contagion_patterns
        contagion_strength = (
            sum(pattern.strength for pattern in contagion_patterns) / len(contagion_patterns)
            if contagion_patterns else 0.0
        )
        source_emotions = sorted({pattern.emotion for pattern in contagion_patterns})
        affected_participants = sorted({
            participant
            for pattern in contagion_patterns
            for participant in pattern.target_participants
        })
        emotional_diversity = (
            len(analysis_result.emotion_distribution) / max(len(participants_emotions), 1)
            if participants_emotions else 0.0
        )
        emotional_balance = max(0.0, min(1.0, 1 - analysis_result.polarization_index))
        
        group_state = jsonable_encoder(analysis_result)
        group_state["emotional_diversity"] = emotional_diversity
        group_state["emotional_balance"] = emotional_balance
        group_state["contagion_strength"] = contagion_strength

        # 生成建议
        recommendations = []
        dominant_emotion_intensity = analysis_result.emotion_distribution.get(
            analysis_result.dominant_emotion,
            0.0
        )
        if dominant_emotion_intensity > 0.8:
            recommendations.append("Monitor high emotional intensity")
        if emotional_diversity < 0.3:
            recommendations.append("Consider introducing emotional variety")
        if contagion_strength > 0.7:
            recommendations.append("Manage emotional contagion effects")
        
        response = GroupEmotionResponse(
            session_id=request.session_id,
            group_emotion_state=group_state,
            individual_emotions={
                pid: data["emotion_vector"].emotions
                for pid, data in participants_data.items()
            },
            emotional_contagion={
                "strength": contagion_strength,
                "source_emotions": source_emotions,
                "affected_participants": affected_participants,
                "patterns": jsonable_encoder(contagion_patterns),
                "velocity": analysis_result.contagion_velocity
            },
            recommendations=recommendations,
            confidence_score=analysis_result.analysis_confidence
        )

        history = group_emotion_history[analysis_result.group_id]
        history.append(group_state)
        if len(history) > 200:
            history[:] = history[-200:]

        record_analysis(
            session_id=request.session_id,
            scenario=request.social_environment.scenario,
            analysis_type="group_emotion",
            confidence_score=analysis_result.analysis_confidence,
            dominant_emotion=analysis_result.dominant_emotion,
            duration_ms=(utc_now() - start_time).total_seconds() * 1000
        )
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "group_emotion_analysis", response.model_dump())
        
        return response
        
    except ValueError as e:
        logger.warning(f"Group emotion analysis validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Group emotion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/relationships", response_model=RelationshipAnalysisResponse)
async def analyze_relationships(request: AnalysisRequest) -> RelationshipAnalysisResponse:
    """分析人际关系动态"""
    try:
        logger.info(f"Starting relationship analysis for session {request.session_id}")
        start_time = utc_now()
        
        # 准备关系数据
        relationships_data = []
        for i, participant1 in enumerate(request.participants):
            for participant2 in request.participants[i+1:]:
                interaction_history = []
                if participant1.relationship_history:
                    interaction_history.extend(participant1.relationship_history)
                if participant2.relationship_history:
                    interaction_history.extend(participant2.relationship_history)
                # 构建关系数据
                relationship = {
                    "participant1": participant1.participant_id,
                    "participant2": participant2.participant_id,
                    "interaction_history": interaction_history,
                    "participant1_emotion": participant1.emotion_data,
                    "participant2_emotion": participant2.emotion_data
                }
                relationships_data.append(relationship)
        
        # 执行关系分析
        analysis_results = []
        health_scores = {}
        
        for relationship in relationships_data:
            participant1_emotions = relationship["participant1_emotion"].emotions or {}
            participant2_emotions = relationship["participant2_emotion"].emotions or {}
            p1_dominant = max(participant1_emotions.items(), key=lambda item: item[1])[0] if participant1_emotions else "neutral"
            p2_dominant = max(participant2_emotions.items(), key=lambda item: item[1])[0] if participant2_emotions else "neutral"
            p1_state = GroupEmotionState(
                participant_id=relationship["participant1"],
                emotion=p1_dominant,
                intensity=relationship["participant1_emotion"].intensity,
                valence=group_analyzer.emotion_polarity.get(p1_dominant, 0.0),
                arousal=max(0.0, min(1.0, relationship["participant1_emotion"].intensity)),
                dominance=0.5,
                timestamp=utc_now(),
                confidence=relationship["participant1_emotion"].confidence
            )
            p2_state = GroupEmotionState(
                participant_id=relationship["participant2"],
                emotion=p2_dominant,
                intensity=relationship["participant2_emotion"].intensity,
                valence=group_analyzer.emotion_polarity.get(p2_dominant, 0.0),
                arousal=max(0.0, min(1.0, relationship["participant2_emotion"].intensity)),
                dominance=0.5,
                timestamp=utc_now(),
                confidence=relationship["participant2_emotion"].confidence
            )
            result = await relationship_analyzer.analyze_relationship_dynamics(
                relationship["participant1"],
                relationship["participant2"],
                [p1_state],
                [p2_state],
                relationship["interaction_history"]
            )
            analysis_results.append(result)
            
            # 计算关系健康分数
            relationship_key = f"{relationship['participant1']}-{relationship['participant2']}"
            health_scores[relationship_key] = result.relationship_health
        
        # 分析交互模式
        interaction_patterns = {
            "dominant_patterns": [],
            "frequency": {},
            "synchrony": 0.5
        }
        if analysis_results:
            dominant_patterns = []
            for item in analysis_results:
                dominant_patterns.extend([pattern.support_type.value for pattern in item.support_patterns])
                if item.conflict_indicators:
                    dominant_patterns.append("conflict")
            interaction_patterns["dominant_patterns"] = sorted(set(dominant_patterns))
            total_interactions = sum(len(r["interaction_history"]) for r in relationships_data)
            interaction_patterns["frequency"] = {"total": total_interactions}
            interaction_patterns["synchrony"] = sum(r.emotional_reciprocity for r in analysis_results) / len(analysis_results)
        
        # 生成建议
        recommendations = []
        avg_health = sum(health_scores.values()) / len(health_scores) if health_scores else 0.5
        
        if avg_health < 0.4:
            recommendations.append("Focus on relationship building activities")
        if any(score < 0.3 for score in health_scores.values()):
            recommendations.append("Address specific relationship conflicts")
        if len(set(health_scores.values())) > len(health_scores) * 0.5:
            recommendations.append("Work on balancing relationship dynamics")
        
        relationships_payload = jsonable_encoder(analysis_results)

        response = RelationshipAnalysisResponse(
            session_id=request.session_id,
            relationships=relationships_payload,
            relationship_dynamics={
                "average_relationship_strength": avg_health,
                "relationship_count": len(health_scores),
                "strongest_relationships": sorted(health_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                "weakest_relationships": sorted(health_scores.items(), key=lambda x: x[1])[:3]
            },
            interaction_patterns={
                "dominant_patterns": interaction_patterns.get("dominant_patterns", []),
                "communication_frequency": interaction_patterns.get("frequency", {}),
                "emotional_synchrony": interaction_patterns.get("synchrony", 0.5)
            },
            relationship_health_scores=health_scores,
            recommendations=recommendations
        )

        for detail in relationships_payload:
            participants = detail.get("participants") or []
            if len(participants) < 2:
                continue
            key = "_".join(sorted(participants[:2]))
            history = relationship_history_store[key]
            history.append(detail)
            if len(history) > 200:
                history[:] = history[-200:]

        record_analysis(
            session_id=request.session_id,
            scenario=request.social_environment.scenario,
            analysis_type="relationships",
            confidence_score=avg_health,
            relationship_health=avg_health,
            duration_ms=(utc_now() - start_time).total_seconds() * 1000
        )
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "relationship_analysis", response.model_dump())
        
        return response
        
    except Exception as e:
        logger.error(f"Relationship analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/relationships/history")
async def get_relationship_history(
    participant1: str,
    participant2: str,
    limit: int = 20
) -> Dict[str, Any]:
    """获取关系历史分析记录"""
    limit = max(1, min(limit, 200))
    key = "_".join(sorted([participant1, participant2]))
    history = relationship_history_store.get(key, [])
    return {
        "relationship_key": key,
        "history": history[-limit:],
        "total": len(history)
    }

@router.get("/group-emotion/history/{group_id}")
async def get_group_emotion_history(
    group_id: str,
    limit: int = 20
) -> Dict[str, Any]:
    """获取群体情感历史记录"""
    limit = max(1, min(limit, 200))
    history = group_emotion_history.get(group_id, [])
    return {
        "group_id": group_id,
        "history": history[-limit:],
        "total": len(history)
    }

@router.post("/analyze/social-context")
async def analyze_social_context(request: AnalysisRequest):
    """分析社交上下文并提供适配建议"""
    try:
        logger.info(f"Starting social context analysis for session {request.session_id}")
        start_time = utc_now()
        
        # 构建社交环境
        scenario = SocialScenario(request.social_environment.scenario)
        social_env = SocialEnvironment(
            scenario=scenario,
            participants_count=request.social_environment.participants_count,
            formality_level=request.social_environment.formality_level,
            emotional_intensity=request.social_environment.emotional_intensity,
            time_pressure=request.social_environment.time_pressure,
            cultural_context=request.social_environment.cultural_context
        )
        
        # 为每个参与者进行上下文适配
        adaptations = []
        for participant in request.participants:
            emotion_vector = EmotionVector(
                emotions=participant.emotion_data.emotions,
                intensity=participant.emotion_data.intensity,
                confidence=participant.emotion_data.confidence,
                context=participant.emotion_data.context
            )
            
            adaptation = await context_adapter.adapt_to_context(
                emotion_vector, social_env, participant.model_dump()
            )
            adaptations.append({
                "participant_id": participant.participant_id,
                "adaptation": adaptation.__dict__
            })
        
        response = {
            "session_id": request.session_id,
            "social_environment": {
                "scenario": scenario.value,
                "context_analysis": "completed",
                "formality_assessment": social_env.formality_level,
                "complexity_score": social_env.participants_count * social_env.emotional_intensity
            },
            "participant_adaptations": adaptations,
            "overall_recommendations": [
                f"Scenario-appropriate behavior for {scenario.value}",
                "Monitor emotional dynamics",
                "Maintain cultural sensitivity"
            ]
        }

        record_analysis(
            session_id=request.session_id,
            scenario=request.social_environment.scenario,
            analysis_type="social_context",
            confidence_score=None,
            duration_ms=(utc_now() - start_time).total_seconds() * 1000
        )
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "social_context_analysis", response)
        
        return response
        
    except Exception as e:
        logger.error(f"Social context analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/cultural-adaptation")
async def analyze_cultural_adaptation(request: AnalysisRequest):
    """分析文化背景并提供适配建议"""
    try:
        logger.info(f"Starting cultural adaptation analysis for session {request.session_id}")
        start_time = utc_now()
        
        # 提取文化指标
        participants_cultural = []
        for participant in request.participants:
            cultural_data = {
                "cultural_indicators": participant.cultural_indicators or {},
                "participant_id": participant.participant_id
            }
            participants_cultural.append(cultural_data)
        
        # 分析文化背景
        cultural_profiles, confidence = await cultural_analyzer.analyze_cultural_context(
            participants_cultural, {"session_type": request.social_environment.scenario}
        )
        
        # 为每个参与者进行文化适配
        adaptations = []
        for participant in request.participants:
            emotion_vector = EmotionVector(
                emotions=participant.emotion_data.emotions,
                intensity=participant.emotion_data.intensity,
                confidence=participant.emotion_data.confidence,
                context=participant.emotion_data.context
            )
            
            adaptation = await cultural_analyzer.adapt_for_cultural_context(
                emotion_vector, cultural_profiles, {"session_id": request.session_id}
            )
            adaptations.append({
                "participant_id": participant.participant_id,
                "cultural_adaptation": adaptation.__dict__
            })
        
        response = {
            "session_id": request.session_id,
            "cultural_analysis": {
                "detected_cultures": [profile.culture_id for profile in cultural_profiles],
                "cultural_diversity": len(cultural_profiles),
                "analysis_confidence": confidence,
                "primary_culture": cultural_profiles[0].culture_id if cultural_profiles else "unknown"
            },
            "participant_adaptations": adaptations,
            "cross_cultural_recommendations": [
                "Use inclusive language",
                "Be aware of cultural differences",
                "Respect different communication styles",
                "Consider cultural sensitivities"
            ]
        }

        record_analysis(
            session_id=request.session_id,
            scenario=request.social_environment.scenario,
            analysis_type="cultural_adaptation",
            confidence_score=confidence,
            cultural_profiles=[profile.culture_id for profile in cultural_profiles],
            duration_ms=(utc_now() - start_time).total_seconds() * 1000
        )
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "cultural_adaptation_analysis", response)
        
        return response
        
    except Exception as e:
        logger.error(f"Cultural adaptation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/decisions/generate", response_model=SocialDecisionResponse)
async def generate_social_decisions(request: AnalysisRequest) -> SocialDecisionResponse:
    """生成社交智能决策建议"""
    try:
        logger.info(f"Generating social decisions for session {request.session_id}")
        start_time = utc_now()
        
        # 构建决策上下文
        current_emotions = {}
        participants_list = []
        
        for participant in request.participants:
            emotion_vector = EmotionVector(
                emotions=participant.emotion_data.emotions,
                intensity=participant.emotion_data.intensity,
                confidence=participant.emotion_data.confidence,
                context=participant.emotion_data.context
            )
            current_emotions[participant.participant_id] = emotion_vector
            participants_list.append(participant.model_dump())
        
        # 构建社交环境
        social_env = SocialEnvironment(
            scenario=SocialScenario(request.social_environment.scenario),
            participants_count=request.social_environment.participants_count,
            formality_level=request.social_environment.formality_level,
            emotional_intensity=request.social_environment.emotional_intensity,
            time_pressure=request.social_environment.time_pressure
        )
        
        # 分析文化背景
        cultural_profiles, _ = await cultural_analyzer.analyze_cultural_context(
            participants_list, {"session_type": request.social_environment.scenario}
        )
        
        decision_context = DecisionContext(
            session_id=request.session_id,
            timestamp=utc_now(),
            participants=participants_list,
            current_emotions=current_emotions,
            group_dynamics={"relationship_tensions": 0.3, "trust_levels": {}},
            social_environment=social_env,
            cultural_profiles=cultural_profiles
        )
        
        # 生成决策
        decision_types = None
        if request.analysis_types:
            valid_types = [DecisionType(dt) for dt in request.analysis_types if dt in [dt.value for dt in DecisionType]]
            decision_types = valid_types or None
        
        decisions = await intelligence_engine.analyze_and_decide(decision_context, decision_types)
        
        # 构建响应
        decisions_data = []
        execution_priority = []
        
        for decision in decisions:
            decision_data = {
                "decision_id": decision.decision_id,
                "type": decision.decision_type.value,
                "priority": decision.priority.value,
                "actions": decision.recommended_actions,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence_score,
                "expected_outcomes": decision.expected_outcomes,
                "alternatives": decision.alternative_options,
                "timeline": decision.execution_timeline
            }
            decisions_data.append(decision_data)
            
            if decision.priority.value in ["critical", "high"]:
                execution_priority.append(decision.decision_id)
        
        response = SocialDecisionResponse(
            session_id=request.session_id,
            decisions=decisions_data,
            execution_priority=execution_priority,
            monitoring_plan={
                "metrics": ["participant_satisfaction", "goal_achievement", "relationship_health"],
                "frequency": "continuous",
                "alerts": ["conflict_escalation", "participation_drop", "cultural_misunderstanding"]
            },
            confidence_score=sum(d.confidence_score for d in decisions) / len(decisions) if decisions else 0.5
        )

        record_analysis(
            session_id=request.session_id,
            scenario=request.social_environment.scenario,
            analysis_type="social_decisions",
            confidence_score=response.confidence_score,
            duration_ms=(utc_now() - start_time).total_seconds() * 1000
        )
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "social_decisions", response.model_dump())
        
        return response
        
    except Exception as e:
        logger.error(f"Social decision generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decision generation failed: {str(e)}")

@router.post("/comprehensive-analysis")
async def comprehensive_analysis(request: AnalysisRequest):
    """执行综合社交情感分析"""
    try:
        logger.info(f"Starting comprehensive analysis for session {request.session_id}")
        
        # 并行执行所有分析
        tasks = [
            analyze_group_emotion(request),
            analyze_relationships(request),
            analyze_social_context(request),
            analyze_cultural_adaptation(request),
            generate_social_decisions(request)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        group_result = results[0] if not isinstance(results[0], Exception) else None
        relationship_result = results[1] if not isinstance(results[1], Exception) else None
        cultural_result = results[3] if not isinstance(results[3], Exception) else None
        decisions_result = results[4] if not isinstance(results[4], Exception) else None

        def _cohesion_value(value: Any) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                mapping = {
                    "high": 0.9,
                    "medium": 0.7,
                    "low": 0.4,
                    "fragmented": 0.2
                }
                return mapping.get(value.lower(), 0.0)
            return 0.0

        social_cohesion = 0.0
        if group_result:
            balance = group_result.group_emotion_state.get("emotional_balance")
            social_cohesion = _cohesion_value(balance) if balance is not None else _cohesion_value(
                group_result.group_emotion_state.get("group_cohesion")
            )

        relationship_health = 0.0
        communication_effectiveness = 0.0
        if relationship_result:
            relationship_health = _cohesion_value(
                relationship_result.relationship_dynamics.get("average_relationship_strength")
            )
            communication_effectiveness = _cohesion_value(
                relationship_result.interaction_patterns.get("emotional_synchrony")
            )

        cultural_harmony = 0.0
        if cultural_result:
            adaptations = cultural_result.get("participant_adaptations", [])
            sensitivity_scores = [
                item.get("cultural_adaptation", {}).get("cultural_sensitivity_score")
                for item in adaptations
                if isinstance(item.get("cultural_adaptation", {}).get("cultural_sensitivity_score"), (int, float))
            ]
            if sensitivity_scores:
                cultural_harmony = sum(sensitivity_scores) / len(sensitivity_scores)

        priority_recommendations = []
        if decisions_result:
            for decision in decisions_result.decisions:
                actions = decision.get("actions") or []
                recommendation = actions[0] if actions else decision.get("reasoning", "")
                expected_outcomes = decision.get("expected_outcomes") or {}
                impacts = [v for v in expected_outcomes.values() if isinstance(v, (int, float))]
                expected_impact = sum(impacts) / len(impacts) if impacts else 0.0
                priority = decision.get("priority", "low")
                if priority == "critical":
                    priority = "high"
                if priority not in {"high", "medium", "low"}:
                    priority = "low"
                priority_recommendations.append({
                    "priority": priority,
                    "category": decision.get("type", ""),
                    "recommendation": recommendation,
                    "expected_impact": expected_impact
                })

        confidence_candidates = []
        if group_result and isinstance(group_result.confidence_score, (int, float)):
            confidence_candidates.append(group_result.confidence_score)
        if decisions_result and isinstance(decisions_result.confidence_score, (int, float)):
            confidence_candidates.append(decisions_result.confidence_score)
        if cultural_result:
            analysis_confidence = cultural_result.get("cultural_analysis", {}).get("analysis_confidence")
            if isinstance(analysis_confidence, (int, float)):
                confidence_candidates.append(analysis_confidence)
        confidence_score = (
            sum(confidence_candidates) / len(confidence_candidates)
            if confidence_candidates else 0.0
        )
        
        # 整合结果
        comprehensive_result = {
            "session_id": request.session_id,
            "analysis_timestamp": utc_now().isoformat(),
            "group_emotion_analysis": results[0].__dict__ if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "relationship_analysis": results[1].__dict__ if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "social_context_analysis": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "cultural_adaptation_analysis": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
            "social_decisions": results[4].__dict__ if not isinstance(results[4], Exception) else {"error": str(results[4])},
            "detailed_insights": {
                "group_emotion": results[0].__dict__ if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "relationships": results[1].__dict__ if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "social_context": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "cultural_adaptation": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
                "decision_recommendations": results[4].__dict__ if not isinstance(results[4], Exception) else {"error": str(results[4])}
            },
            "analysis_summary": {
                "overall_emotional_state": group_result.group_emotion_state if group_result else {},
                "total_participants": len(request.participants),
                "social_scenario": request.social_environment.scenario,
                "analysis_completeness": sum(1 for r in results if not isinstance(r, Exception)) / len(results),
                "relationship_health": relationship_health,
                "social_cohesion": social_cohesion,
                "cultural_harmony": cultural_harmony,
                "communication_effectiveness": communication_effectiveness,
                "key_insights": [
                    "Group emotional state analyzed",
                    "Relationship dynamics assessed",
                    "Cultural considerations identified",
                    "Social decisions recommended"
                ]
            },
            "priority_recommendations": priority_recommendations,
            "confidence_score": confidence_score
        }
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "comprehensive_analysis", comprehensive_result)
        
        return comprehensive_result
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

# WebSocket端点
@router.websocket("/realtime/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """实时社交情感分析WebSocket端点"""
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        logger.info(f"WebSocket connection established for session {session_id}")
        
        # 发送连接确认
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": utc_now().isoformat(),
            "available_analyses": [
                "group_emotion_analysis",
                "relationship_analysis", 
                "social_context_analysis",
                "cultural_adaptation_analysis",
                "social_decisions",
                "comprehensive_analysis"
            ]
        }))
        
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": utc_now().isoformat()
                }))
            
            elif message_type == "analysis_request":
                # 处理分析请求
                try:
                    request_data = message.get("data")
                    analysis_request = AnalysisRequest(**request_data)
                    analysis_request.real_time = True  # 确保实时推送
                    
                    # 根据请求类型执行分析
                    analysis_type = message.get("analysis_type", "comprehensive")
                    
                    if analysis_type == "group_emotion":
                        result = await analyze_group_emotion(analysis_request)
                    elif analysis_type == "relationships":
                        result = await analyze_relationships(analysis_request)
                    elif analysis_type == "social_context":
                        result = await analyze_social_context(analysis_request)
                    elif analysis_type == "cultural_adaptation":
                        result = await analyze_cultural_adaptation(analysis_request)
                    elif analysis_type == "decisions":
                        result = await generate_social_decisions(analysis_request)
                    else:
                        result = await comprehensive_analysis(analysis_request)
                    
                    # 发送结果（除了实时推送外的直接响应）
                    await websocket.send_text(json.dumps({
                        "type": "analysis_result",
                        "analysis_type": analysis_type,
                        "result": result.__dict__ if hasattr(result, '__dict__') else result,
                        "timestamp": utc_now().isoformat()
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Analysis failed: {str(e)}",
                        "timestamp": utc_now().isoformat()
                    }))
            
            elif message_type == "get_session_status":
                # 返回会话状态
                await websocket.send_text(json.dumps({
                    "type": "session_status",
                    "session_id": session_id,
                    "connected": True,
                    "last_activity": utc_now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]

async def send_realtime_update(session_id: str, update_type: str, data: Dict[str, Any]):
    """发送实时更新"""
    if session_id in active_connections:
        try:
            message = {
                "type": "realtime_update",
                "update_type": update_type,
                "data": data,
                "timestamp": utc_now().isoformat()
            }
            await active_connections[session_id].send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send realtime update to {session_id}: {e}")

# 健康检查和系统状态端点
@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "social_emotional_understanding",
        "active_connections": len(active_connections),
        "timestamp": utc_now().isoformat(),
        "components": {
            "group_analyzer": "operational",
            "relationship_analyzer": "operational", 
            "context_adapter": "operational",
            "cultural_analyzer": "operational",
            "intelligence_engine": "operational"
        }
    }

@router.get("/analytics")
async def get_system_analytics():
    """获取系统分析数据"""
    try:
        records = list(analysis_records)
        session_ids = {record["session_id"] for record in records if record.get("session_id")}
        confidence_values = [
            record["confidence_score"] for record in records
            if record.get("confidence_score") is not None
        ]
        scenario_counts: Dict[str, int] = defaultdict(int)
        for record in records:
            if record.get("scenario"):
                scenario_counts[record["scenario"]] += 1
        most_common_scenarios = sorted(
            [{"scenario": scenario, "count": count} for scenario, count in scenario_counts.items()],
            key=lambda item: item["count"],
            reverse=True
        )[:5]

        culture_counts: Dict[str, int] = defaultdict(int)
        for record in records:
            for culture in record.get("cultural_profiles", []) or []:
                culture_counts[culture] += 1

        emotion_counts: Dict[str, int] = defaultdict(int)
        for record in records:
            if record.get("dominant_emotion"):
                emotion_counts[record["dominant_emotion"]] += 1

        relationship_values = [
            record["relationship_health"] for record in records
            if record.get("relationship_health") is not None
        ]
        avg_relationship_health = sum(relationship_values) / len(relationship_values) if relationship_values else 0.0

        recent_cutoff = utc_now() - timedelta(days=7)
        recent_relationship_values = [
            record["relationship_health"] for record in records
            if record.get("relationship_health") is not None and record.get("timestamp") >= recent_cutoff
        ]
        recent_avg = sum(recent_relationship_values) / len(recent_relationship_values) if recent_relationship_values else 0.0
        if recent_avg > avg_relationship_health + 0.05:
            relationship_trend = "improving"
        elif recent_avg < avg_relationship_health - 0.05:
            relationship_trend = "deteriorating"
        else:
            relationship_trend = "stable"

        monitoring_stats = await get_monitoring_service().performance_monitor.get_stats()
        error_rate = float(monitoring_stats.get("error_rate", 0))
        avg_response_time = float(monitoring_stats.get("average_response_time_ms", 0))

        return {
            "total_sessions": len(session_ids),
            "total_analyses_performed": len(records),
            "avg_confidence_score": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
            "most_common_scenarios": most_common_scenarios,
            "cultural_diversity_stats": {
                "unique_cultures": len(culture_counts),
                "distribution": dict(culture_counts)
            },
            "emotional_state_trends": dict(emotion_counts),
            "relationship_health_trends": {
                "average": avg_relationship_health,
                "recent_average": recent_avg,
                "trend": relationship_trend
            },
            "system_performance": {
                "avg_response_time": avg_response_time,
                "success_rate": max(0.0, 1.0 - error_rate),
                "error_rate": error_rate
            },
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        return {
            "error": "Analytics unavailable",
            "timestamp": utc_now().isoformat()
        }
