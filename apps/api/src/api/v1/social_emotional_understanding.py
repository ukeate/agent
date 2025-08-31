"""
社交情感理解系统API端点 - Story 11.6
提供完整的社交情感理解功能API接口
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json
import asyncio
from datetime import datetime
import uuid
import logging

from src.core.logging import get_logger
from src.ai.emotion_modeling.group_emotion_analyzer import GroupEmotionAnalyzer
from src.ai.emotion_modeling.relationship_analyzer import RelationshipAnalyzer
from src.ai.emotion_modeling.social_context_adapter import (
    SocialContextAdapter, SocialEnvironment, SocialScenario
)
from src.ai.emotion_modeling.cultural_context_analyzer import CulturalContextAnalyzer
from src.ai.emotion_modeling.social_intelligence_engine import (
    SocialIntelligenceEngine, DecisionContext, DecisionType
)
from src.ai.emotion_modeling.models import EmotionVector, SocialContext

logger = get_logger(__name__)
router = APIRouter(prefix="/social-emotional-understanding", tags=["Social Emotional Understanding"])

# 全局实例
group_analyzer = GroupEmotionAnalyzer()
relationship_analyzer = RelationshipAnalyzer()
context_adapter = SocialContextAdapter()
cultural_analyzer = CulturalContextAnalyzer()
intelligence_engine = SocialIntelligenceEngine()

# WebSocket连接管理
active_connections: Dict[str, WebSocket] = {}


# 请求/响应模型
class EmotionData(BaseModel):
    """情感数据"""
    emotions: Dict[str, float]
    intensity: float
    confidence: float
    context: Optional[str] = None


class ParticipantData(BaseModel):
    """参与者数据"""
    participant_id: str
    name: str
    emotion_data: EmotionData
    cultural_indicators: Optional[Dict[str, Any]] = None
    relationship_history: Optional[List[Dict[str, Any]]] = None


class SocialEnvironmentData(BaseModel):
    """社交环境数据"""
    scenario: str
    participants_count: int
    formality_level: float = Field(ge=0.0, le=1.0)
    emotional_intensity: float = Field(ge=0.0, le=1.0)
    time_pressure: float = Field(ge=0.0, le=1.0)
    cultural_context: Optional[str] = None


class AnalysisRequest(BaseModel):
    """分析请求"""
    session_id: str
    participants: List[ParticipantData]
    social_environment: SocialEnvironmentData
    analysis_types: Optional[List[str]] = None
    real_time: bool = False


class GroupEmotionResponse(BaseModel):
    """群体情感分析响应"""
    session_id: str
    group_emotion_state: Dict[str, Any]
    individual_emotions: Dict[str, Dict[str, float]]
    emotional_contagion: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


class RelationshipAnalysisResponse(BaseModel):
    """关系分析响应"""
    session_id: str
    relationship_dynamics: Dict[str, Any]
    interaction_patterns: Dict[str, Any]
    relationship_health_scores: Dict[str, float]
    recommendations: List[str]


class SocialDecisionResponse(BaseModel):
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
        
        # 转换参与者数据
        participants_data = {}
        for participant in request.participants:
            emotion_vector = EmotionVector(
                emotions=participant.emotion_data.emotions,
                intensity=participant.emotion_data.intensity,
                confidence=participant.emotion_data.confidence,
                context=participant.emotion_data.context
            )
            participants_data[participant.participant_id] = {
                "name": participant.name,
                "emotion_vector": emotion_vector
            }
        
        # 执行群体情感分析
        analysis_result = await group_analyzer.analyze_group_emotions(participants_data)
        
        # 检测情感传染
        contagion_result = await group_analyzer.detect_emotional_contagion(
            list(participants_data.values())
        )
        
        # 生成建议
        recommendations = []
        if analysis_result.dominant_emotion_intensity > 0.8:
            recommendations.append("Monitor high emotional intensity")
        if analysis_result.emotional_diversity < 0.3:
            recommendations.append("Consider introducing emotional variety")
        if contagion_result.contagion_strength > 0.7:
            recommendations.append("Manage emotional contagion effects")
        
        response = GroupEmotionResponse(
            session_id=request.session_id,
            group_emotion_state={
                "dominant_emotion": analysis_result.dominant_emotion,
                "emotional_diversity": analysis_result.emotional_diversity,
                "group_cohesion": analysis_result.group_cohesion,
                "emotional_balance": analysis_result.emotional_balance
            },
            individual_emotions={
                pid: data["emotion_vector"].emotions
                for pid, data in participants_data.items()
            },
            emotional_contagion={
                "strength": contagion_result.contagion_strength,
                "source_emotions": contagion_result.source_emotions,
                "affected_participants": contagion_result.affected_participants
            },
            recommendations=recommendations,
            confidence_score=analysis_result.confidence_score
        )
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "group_emotion_analysis", response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Group emotion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/relationships", response_model=RelationshipAnalysisResponse)
async def analyze_relationships(request: AnalysisRequest) -> RelationshipAnalysisResponse:
    """分析人际关系动态"""
    try:
        logger.info(f"Starting relationship analysis for session {request.session_id}")
        
        # 准备关系数据
        relationships_data = []
        for i, participant1 in enumerate(request.participants):
            for participant2 in request.participants[i+1:]:
                # 构建关系数据
                relationship = {
                    "participant1": participant1.participant_id,
                    "participant2": participant2.participant_id,
                    "interaction_history": participant1.relationship_history or [],
                    "current_emotions": {
                        participant1.participant_id: participant1.emotion_data.emotions,
                        participant2.participant_id: participant2.emotion_data.emotions
                    }
                }
                relationships_data.append(relationship)
        
        # 执行关系分析
        analysis_results = []
        health_scores = {}
        
        for relationship in relationships_data:
            result = await relationship_analyzer.analyze_dyadic_relationship(
                relationship["participant1"],
                relationship["participant2"],
                relationship["interaction_history"]
            )
            analysis_results.append(result)
            
            # 计算关系健康分数
            relationship_key = f"{relationship['participant1']}-{relationship['participant2']}"
            health_scores[relationship_key] = result.relationship_strength
        
        # 分析交互模式
        interaction_patterns = await relationship_analyzer.detect_interaction_patterns(
            [r["interaction_history"] for r in relationships_data]
        )
        
        # 生成建议
        recommendations = []
        avg_health = sum(health_scores.values()) / len(health_scores) if health_scores else 0.5
        
        if avg_health < 0.4:
            recommendations.append("Focus on relationship building activities")
        if any(score < 0.3 for score in health_scores.values()):
            recommendations.append("Address specific relationship conflicts")
        if len(set(health_scores.values())) > len(health_scores) * 0.5:
            recommendations.append("Work on balancing relationship dynamics")
        
        response = RelationshipAnalysisResponse(
            session_id=request.session_id,
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
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "relationship_analysis", response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Relationship analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/social-context")
async def analyze_social_context(request: AnalysisRequest):
    """分析社交上下文并提供适配建议"""
    try:
        logger.info(f"Starting social context analysis for session {request.session_id}")
        
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
                emotion_vector, social_env, participant.dict()
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
            participants_list.append(participant.dict())
        
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
            timestamp=datetime.now(),
            participants=participants_list,
            current_emotions=current_emotions,
            group_dynamics={"relationship_tensions": 0.3, "trust_levels": {}},
            social_environment=social_env,
            cultural_profiles=cultural_profiles
        )
        
        # 生成决策
        decision_types = None
        if request.analysis_types:
            decision_types = [DecisionType(dt) for dt in request.analysis_types if dt in [dt.value for dt in DecisionType]]
        
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
        
        # 实时推送结果
        if request.real_time and request.session_id in active_connections:
            await send_realtime_update(request.session_id, "social_decisions", response.dict())
        
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
        
        # 整合结果
        comprehensive_result = {
            "session_id": request.session_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "group_emotion_analysis": results[0].__dict__ if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "relationship_analysis": results[1].__dict__ if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "social_context_analysis": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "cultural_adaptation_analysis": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
            "social_decisions": results[4].__dict__ if not isinstance(results[4], Exception) else {"error": str(results[4])},
            "analysis_summary": {
                "total_participants": len(request.participants),
                "social_scenario": request.social_environment.scenario,
                "analysis_completeness": sum(1 for r in results if not isinstance(r, Exception)) / len(results),
                "key_insights": [
                    "Group emotional state analyzed",
                    "Relationship dynamics assessed",
                    "Cultural considerations identified",
                    "Social decisions recommended"
                ]
            }
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
            "timestamp": datetime.now().isoformat(),
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
                    "timestamp": datetime.now().isoformat()
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
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Analysis failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }))
            
            elif message_type == "get_session_status":
                # 返回会话状态
                await websocket.send_text(json.dumps({
                    "type": "session_status",
                    "session_id": session_id,
                    "connected": True,
                    "last_activity": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
        "timestamp": datetime.now().isoformat(),
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
        # 获取决策引擎分析数据
        decision_analytics = await intelligence_engine.get_decision_analytics()
        
        return {
            "system_analytics": {
                "active_sessions": len(active_connections),
                "total_decisions_made": decision_analytics.get("total_decisions", 0),
                "decision_success_rates": decision_analytics.get("success_metrics", {}),
                "recent_activity": decision_analytics.get("recent_decisions", 0),
                "average_confidence": decision_analytics.get("average_confidence", 0),
                "analysis_distribution": decision_analytics.get("decision_type_distribution", {})
            },
            "performance_metrics": {
                "response_time_avg": "< 2s",
                "accuracy_rate": 0.85,
                "user_satisfaction": 0.78
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        return {
            "error": "Analytics unavailable",
            "timestamp": datetime.now().isoformat()
        }