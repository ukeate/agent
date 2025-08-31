"""
社交情感理解系统集成 - Story 11.6 Task 8
统一集成所有社交情感理解组件，提供完整的系统API
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from fastapi import WebSocket
import websockets

from .models import EmotionVector, SocialContext
from .core_interfaces import EmotionModelingInterface
from .social_context_adapter import SocialContextAdapter, SocialScenario, SocialEnvironment
from .cultural_context_analyzer import CulturalContextAnalyzer, CulturalDimension
from .social_intelligence_engine import SocialIntelligenceEngine, DecisionType
from .social_analytics_tools import SocialAnalyticsTools, AnalysisType
from .privacy_ethics_guard import PrivacyEthicsGuard, PrivacyLevel, ConsentType

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """系统运行模式"""
    REAL_TIME = "real_time"
    BATCH_PROCESSING = "batch_processing"
    ANALYSIS_ONLY = "analysis_only"
    FULL_INTERACTIVE = "full_interactive"


@dataclass
class SystemConfiguration:
    """系统配置"""
    mode: SystemMode
    privacy_level: PrivacyLevel
    cultural_context: Optional[str]
    enable_real_time_monitoring: bool
    enable_predictive_analytics: bool
    enable_emotional_coaching: bool
    max_concurrent_sessions: int
    data_retention_days: int
    websocket_enabled: bool


@dataclass
class SocialEmotionRequest:
    """社交情感分析请求"""
    request_id: str
    user_id: str
    session_id: Optional[str]
    emotion_data: Dict[str, Any]
    social_context: Dict[str, Any]
    analysis_type: List[str]  # ["emotion_flow", "network_map", "decision_support"]
    privacy_consent: bool
    cultural_context: Optional[str]
    timestamp: datetime


@dataclass
class SocialEmotionResponse:
    """社交情感分析响应"""
    request_id: str
    user_id: str
    session_id: Optional[str]
    results: Dict[str, Any]
    recommendations: List[str]
    privacy_compliant: bool
    cultural_adaptations: List[str]
    confidence_score: float
    processing_time: float
    timestamp: datetime


@dataclass
class SystemStatus:
    """系统状态"""
    active_sessions: int
    total_users: int
    processing_queue_size: int
    average_response_time: float
    compliance_score: float
    cultural_contexts: List[str]
    last_updated: datetime


class SocialEmotionSystem:
    """社交情感理解系统主类"""
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.active_sessions: Dict[str, Any] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # 初始化所有子系统
        self.context_adapter = SocialContextAdapter()
        self.cultural_analyzer = CulturalContextAnalyzer()
        self.intelligence_engine = SocialIntelligenceEngine()
        self.analytics_tools = SocialAnalyticsTools()
        self.privacy_guard = PrivacyEthicsGuard()
        
        # 系统统计
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # 实时监控
        if config.enable_real_time_monitoring:
            self._start_monitoring_task()
    
    async def process_social_emotion_request(
        self,
        request: SocialEmotionRequest
    ) -> SocialEmotionResponse:
        """处理社交情感分析请求"""
        start_time = datetime.now()
        
        try:
            # 1. 验证隐私合规性
            privacy_compliant, privacy_violations = await self.privacy_guard.validate_privacy_compliance(
                request.user_id,
                request.emotion_data,
                "social_emotion_analysis",
                request.cultural_context
            )
            
            if not privacy_compliant:
                return SocialEmotionResponse(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    results={"error": "Privacy compliance check failed", "violations": privacy_violations},
                    recommendations=["Review privacy settings and obtain proper consent"],
                    privacy_compliant=False,
                    cultural_adaptations=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )
            
            # 2. 构建情感向量和社交上下文
            emotion_vector = self._build_emotion_vector(request.emotion_data)
            social_context = self._build_social_context(request.social_context)
            
            # 3. 执行伦理边界检查
            ethical_approval, ethical_message = await self.privacy_guard.enforce_ethical_boundaries(
                emotion_vector, social_context, "analysis"
            )
            
            if not ethical_approval:
                return SocialEmotionResponse(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    results={"error": "Ethical boundary violation", "message": ethical_message},
                    recommendations=["Adjust analysis parameters to comply with ethical guidelines"],
                    privacy_compliant=True,
                    cultural_adaptations=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )
            
            # 4. 执行所有分析组件
            results = {}
            recommendations = []
            cultural_adaptations = []
            confidence_scores = []
            
            # 社交场景适配
            if "context_adaptation" in request.analysis_type:
                social_environment = self._create_social_environment(request.social_context)
                adaptation_result = await self.context_adapter.adapt_to_context(
                    emotion_vector, social_environment
                )
                results["context_adaptation"] = {
                    "original_emotion": asdict(adaptation_result.original_emotion),
                    "adapted_emotion": asdict(adaptation_result.adapted_emotion),
                    "scenario": adaptation_result.scenario.value,
                    "confidence": adaptation_result.confidence_score,
                    "suggested_actions": adaptation_result.suggested_actions
                }
                recommendations.extend(adaptation_result.suggested_actions[:3])
                confidence_scores.append(adaptation_result.confidence_score)
            
            # 文化背景分析
            if "cultural_analysis" in request.analysis_type and request.cultural_context:
                cultural_result = await self.cultural_analyzer.analyze_cultural_context(
                    emotion_vector, request.cultural_context, social_context
                )
                results["cultural_analysis"] = {
                    "cultural_profile": asdict(cultural_result.cultural_profile),
                    "adaptation_recommendations": cultural_result.adaptation_recommendations,
                    "sensitivity_score": cultural_result.sensitivity_score,
                    "communication_style": cultural_result.communication_style.value
                }
                cultural_adaptations.extend(cultural_result.adaptation_recommendations[:3])
                confidence_scores.append(cultural_result.sensitivity_score)
            
            # 社交智能决策
            if "decision_support" in request.analysis_type:
                decision_context = self._create_decision_context(request)
                decision_result = await self.intelligence_engine.make_social_decision(
                    decision_context, emotion_vector
                )
                results["decision_support"] = {
                    "decision": decision_result.decision,
                    "reasoning": decision_result.reasoning,
                    "confidence": decision_result.confidence,
                    "alternatives": decision_result.alternatives,
                    "risk_assessment": decision_result.risk_assessment
                }
                recommendations.append(f"Recommended action: {decision_result.decision}")
                confidence_scores.append(decision_result.confidence)
            
            # 情感流分析
            if "emotion_flow" in request.analysis_type and request.session_id:
                conversation_data = await self._get_conversation_data(request.session_id)
                flow_result = await self.analytics_tools.analyze_emotion_flow(
                    request.session_id, conversation_data
                )
                results["emotion_flow"] = {
                    "session_id": flow_result.session_id,
                    "participants": flow_result.participants,
                    "overall_trend": flow_result.overall_trend,
                    "dominant_emotions": flow_result.dominant_emotions,
                    "peaks_count": len(flow_result.emotional_peaks),
                    "valleys_count": len(flow_result.emotional_valleys),
                    "turning_points": len(flow_result.turning_points)
                }
                if flow_result.overall_trend == "declining":
                    recommendations.append("Consider intervention to improve emotional flow")
                confidence_scores.append(0.8)  # 基于数据完整性
            
            # 社交网络分析
            if "network_analysis" in request.analysis_type:
                session_data = await self._get_session_data(request.session_id or request.user_id)
                interaction_history = await self._get_interaction_history(request.user_id)
                network_result = await self.analytics_tools.build_emotion_network(
                    session_data, interaction_history
                )
                results["network_analysis"] = {
                    "network_id": network_result.network_id,
                    "nodes_count": len(network_result.nodes),
                    "user_role": network_result.nodes.get(request.user_id, {}).role if request.user_id in network_result.nodes else "unknown",
                    "network_cohesion": network_result.network_cohesion,
                    "polarization_level": network_result.polarization_level,
                    "central_nodes": network_result.central_nodes
                }
                if network_result.polarization_level > 0.7:
                    recommendations.append("High polarization detected, consider mediation strategies")
                confidence_scores.append(0.7)
            
            # 个人统计分析
            if "personal_stats" in request.analysis_type:
                analysis_period = (datetime.now() - timedelta(days=30), datetime.now())
                interaction_data = await self._get_user_interaction_data(
                    request.user_id, analysis_period
                )
                stats_result = await self.analytics_tools.generate_social_emotion_stats(
                    request.user_id, analysis_period, interaction_data
                )
                results["personal_stats"] = {
                    "emotional_diversity": stats_result.emotional_diversity,
                    "average_valence": stats_result.average_valence,
                    "emotional_stability": stats_result.emotional_stability,
                    "social_adaptability": stats_result.social_adaptability,
                    "influence_score": stats_result.influence_score,
                    "support_ratio": stats_result.support_given / max(stats_result.support_received, 1),
                    "scenario_performance": stats_result.scenario_performance
                }
                if stats_result.emotional_stability < 0.4:
                    recommendations.append("Consider emotional stability coaching")
                confidence_scores.append(0.8)
            
            # 计算整体置信度
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            # 数据匿名化（如果需要）
            if self.config.privacy_level in [PrivacyLevel.CONFIDENTIAL, PrivacyLevel.HIGHLY_CONFIDENTIAL]:
                results = await self.privacy_guard.anonymize_emotional_data(
                    results, 
                    "high" if self.config.privacy_level == PrivacyLevel.HIGHLY_CONFIDENTIAL else "standard"
                )
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 更新统计
            self.request_count += 1
            self.total_processing_time += processing_time
            
            # 实时推送结果（如果启用）
            if self.config.websocket_enabled and request.user_id in self.websocket_connections:
                await self._send_realtime_update(request.user_id, results)
            
            return SocialEmotionResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                results=results,
                recommendations=list(set(recommendations)),  # 去重
                privacy_compliant=True,
                cultural_adaptations=list(set(cultural_adaptations)),  # 去重
                confidence_score=overall_confidence,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Social emotion processing failed: {e}")
            
            return SocialEmotionResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                results={"error": str(e)},
                recommendations=["Please try again or contact support"],
                privacy_compliant=False,
                cultural_adaptations=[],
                confidence_score=0.0,
                processing_time=0.0,
                timestamp=datetime.now()
            )
    
    def _build_emotion_vector(self, emotion_data: Dict[str, Any]) -> EmotionVector:
        """构建情感向量"""
        return EmotionVector(
            emotions=emotion_data.get("emotions", {"neutral": 1.0}),
            intensity=emotion_data.get("intensity", 0.5),
            confidence=emotion_data.get("confidence", 0.5),
            context=emotion_data.get("context", {})
        )
    
    def _build_social_context(self, social_data: Dict[str, Any]) -> SocialContext:
        """构建社交上下文"""
        return SocialContext(
            participants=social_data.get("participants", []),
            scenario=social_data.get("scenario", "general"),
            cultural_context=social_data.get("cultural_context"),
            power_dynamics=social_data.get("power_dynamics", {}),
            group_emotions=social_data.get("group_emotions", {}),
            relationship_history=social_data.get("relationship_history", [])
        )
    
    def _create_social_environment(self, social_data: Dict[str, Any]) -> SocialEnvironment:
        """创建社交环境"""
        scenario_mapping = {
            "meeting": SocialScenario.FORMAL_MEETING,
            "casual": SocialScenario.CASUAL_CONVERSATION,
            "brainstorm": SocialScenario.TEAM_BRAINSTORMING,
            "conflict": SocialScenario.CONFLICT_RESOLUTION,
            "presentation": SocialScenario.PRESENTATION
        }
        
        scenario_str = social_data.get("scenario", "casual")
        scenario = scenario_mapping.get(scenario_str, SocialScenario.CASUAL_CONVERSATION)
        
        return SocialEnvironment(
            scenario=scenario,
            participants_count=len(social_data.get("participants", [])),
            formality_level=social_data.get("formality_level", 0.5),
            emotional_intensity=social_data.get("emotional_intensity", 0.5),
            time_pressure=social_data.get("time_pressure", 0.3),
            cultural_context=social_data.get("cultural_context"),
            dominant_emotions=social_data.get("dominant_emotions", []),
            power_dynamics=social_data.get("power_dynamics", {})
        )
    
    def _create_decision_context(self, request: SocialEmotionRequest) -> Dict[str, Any]:
        """创建决策上下文"""
        return {
            "user_id": request.user_id,
            "scenario": request.social_context.get("scenario", "general"),
            "participants": request.social_context.get("participants", []),
            "urgency": request.social_context.get("urgency", 0.5),
            "stakes": request.social_context.get("stakes", 0.5),
            "cultural_context": request.cultural_context
        }
    
    async def _get_conversation_data(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话数据（模拟数据）"""
        # 实际实现中应从数据库获取
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "user_id": "user_1",
                "emotion_data": {
                    "dominant_emotion": "excited",
                    "intensity": 0.7,
                    "valence": 0.6,
                    "arousal": 0.8
                },
                "context": {"topic": "project_planning"}
            }
        ]
    
    async def _get_session_data(self, session_or_user_id: str) -> List[Dict[str, Any]]:
        """获取会话数据"""
        # 模拟数据
        return [
            {
                "user_id": session_or_user_id,
                "timestamp": datetime.now().isoformat(),
                "emotion_data": {
                    "emotions": {"positive": 0.7, "neutral": 0.3},
                    "intensity": 0.6,
                    "confidence": 0.8
                }
            }
        ]
    
    async def _get_interaction_history(self, user_id: str) -> List[Dict[str, Any]]:
        """获取交互历史"""
        # 模拟数据
        return [
            {
                "user1": user_id,
                "user2": "other_user",
                "timestamp": datetime.now().isoformat(),
                "interaction_type": "message",
                "emotion_data": {"valence": 0.5}
            }
        ]
    
    async def _get_user_interaction_data(
        self,
        user_id: str,
        period: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """获取用户交互数据"""
        # 模拟数据
        return [
            {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "emotion_data": {
                    "valence": 0.6,
                    "arousal": 0.5,
                    "intensity": 0.7,
                    "confidence": 0.8
                },
                "context": {"scenario": "meeting"},
                "behavior_type": "normal_interaction"
            }
        ]
    
    async def _send_realtime_update(
        self,
        user_id: str,
        results: Dict[str, Any]
    ) -> None:
        """发送实时更新"""
        if user_id in self.websocket_connections:
            try:
                websocket = self.websocket_connections[user_id]
                await websocket.send_text(json.dumps({
                    "type": "emotion_analysis_update",
                    "data": results,
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.warning(f"Failed to send realtime update to user {user_id}: {e}")
    
    async def register_websocket(self, user_id: str, websocket: WebSocket) -> None:
        """注册WebSocket连接"""
        await websocket.accept()
        self.websocket_connections[user_id] = websocket
        logger.info(f"WebSocket registered for user {user_id}")
    
    async def unregister_websocket(self, user_id: str) -> None:
        """取消注册WebSocket连接"""
        if user_id in self.websocket_connections:
            del self.websocket_connections[user_id]
            logger.info(f"WebSocket unregistered for user {user_id}")
    
    async def create_session(
        self,
        user_id: str,
        session_config: Dict[str, Any]
    ) -> str:
        """创建新会话"""
        session_id = f"session_{datetime.now().isoformat()}_{user_id}"
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "config": session_config,
            "interaction_count": 0,
            "last_activity": datetime.now()
        }
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    async def close_session(self, session_id: str) -> bool:
        """关闭会话"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Closed session {session_id}")
            return True
        return False
    
    async def get_system_status(self) -> SystemStatus:
        """获取系统状态"""
        return SystemStatus(
            active_sessions=len(self.active_sessions),
            total_users=len(self.privacy_guard.privacy_policies),
            processing_queue_size=0,  # 简化实现
            average_response_time=self.total_processing_time / max(self.request_count, 1),
            compliance_score=await self._calculate_compliance_score(),
            cultural_contexts=list(self.cultural_analyzer.cultural_profiles.keys()),
            last_updated=datetime.now()
        )
    
    async def _calculate_compliance_score(self) -> float:
        """计算合规分数"""
        # 基于错误率和违规记录计算
        error_rate = self.error_count / max(self.request_count, 1)
        violation_count = len(self.privacy_guard.ethical_violations)
        
        base_score = 1.0 - error_rate
        violation_penalty = min(violation_count * 0.1, 0.5)
        
        return max(0.0, base_score - violation_penalty)
    
    def _start_monitoring_task(self) -> None:
        """启动监控任务"""
        async def monitor():
            while True:
                try:
                    # 清理过期会话
                    await self._cleanup_expired_sessions()
                    
                    # 清理过期数据
                    await self.privacy_guard.cleanup_expired_data()
                    
                    # 等待下一次监控
                    await asyncio.sleep(300)  # 5分钟
                    
                except Exception as e:
                    logger.error(f"Monitoring task error: {e}")
                    await asyncio.sleep(60)  # 错误时等待1分钟
        
        # 在后台启动监控任务
        asyncio.create_task(monitor())
    
    async def _cleanup_expired_sessions(self) -> None:
        """清理过期会话"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            # 如果会话超过24小时没有活动，视为过期
            if (current_time - session_data["last_activity"]).total_seconds() > 86400:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.close_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def batch_process(
        self,
        requests: List[SocialEmotionRequest]
    ) -> List[SocialEmotionResponse]:
        """批量处理请求"""
        if self.config.mode != SystemMode.BATCH_PROCESSING:
            logger.warning("Batch processing not enabled in current mode")
        
        responses = []
        
        # 并发处理请求
        tasks = [
            self.process_social_emotion_request(request)
            for request in requests
        ]
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Batch processing error for request {i}: {response}")
                    responses[i] = SocialEmotionResponse(
                        request_id=requests[i].request_id,
                        user_id=requests[i].user_id,
                        session_id=requests[i].session_id,
                        results={"error": str(response)},
                        recommendations=[],
                        privacy_compliant=False,
                        cultural_adaptations=[],
                        confidence_score=0.0,
                        processing_time=0.0,
                        timestamp=datetime.now()
                    )
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # 返回错误响应
            responses = [
                SocialEmotionResponse(
                    request_id=req.request_id,
                    user_id=req.user_id,
                    session_id=req.session_id,
                    results={"error": "Batch processing failed"},
                    recommendations=[],
                    privacy_compliant=False,
                    cultural_adaptations=[],
                    confidence_score=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )
                for req in requests
            ]
        
        return responses
    
    async def get_analytics_dashboard(
        self,
        user_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """获取分析仪表板数据"""
        if not time_range:
            time_range = (datetime.now() - timedelta(days=7), datetime.now())
        
        dashboard_data = {
            "system_overview": await self.get_system_status(),
            "compliance_report": await self.privacy_guard.get_compliance_report(*time_range),
            "cultural_distribution": {
                culture: len([
                    policy for policy in self.privacy_guard.privacy_policies.values()
                    # 这里简化处理，实际应该从用户数据中获取文化分布
                ])
                for culture in self.cultural_analyzer.cultural_profiles.keys()
            }
        }
        
        if user_id:
            # 个人用户分析
            interaction_data = await self._get_user_interaction_data(user_id, time_range)
            user_stats = await self.analytics_tools.generate_social_emotion_stats(
                user_id, time_range, interaction_data
            )
            dashboard_data["user_analysis"] = asdict(user_stats)
        
        return dashboard_data
    
    async def export_data(
        self,
        user_id: str,
        data_types: List[str],
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """导出用户数据"""
        # 验证数据导出权限
        privacy_policy = self.privacy_guard.privacy_policies.get(user_id)
        if not privacy_policy:
            return {"error": "No privacy policy found for user"}
        
        export_data = {}
        
        if "emotion_data" in data_types:
            # 导出情感数据
            interaction_data = await self._get_user_interaction_data(
                user_id, 
                (datetime.now() - timedelta(days=privacy_policy.data_retention_days), datetime.now())
            )
            export_data["emotion_data"] = interaction_data
        
        if "privacy_settings" in data_types:
            # 导出隐私设置
            export_data["privacy_policy"] = asdict(privacy_policy)
            export_data["consent_records"] = [
                asdict(consent) for consent in self.privacy_guard.consent_records.get(user_id, [])
            ]
        
        if "analytics" in data_types:
            # 导出分析结果
            time_range = (datetime.now() - timedelta(days=30), datetime.now())
            interaction_data = await self._get_user_interaction_data(user_id, time_range)
            stats = await self.analytics_tools.generate_social_emotion_stats(
                user_id, time_range, interaction_data
            )
            export_data["analytics"] = asdict(stats)
        
        # 应用匿名化（如果需要）
        if privacy_policy.anonymization_required:
            export_data = await self.privacy_guard.anonymize_emotional_data(export_data)
        
        # 记录导出活动
        await self.privacy_guard._record_processing_activity(
            user_id, export_data, "data_export", []
        )
        
        return {
            "user_id": user_id,
            "export_timestamp": datetime.now().isoformat(),
            "format": format_type,
            "data": export_data
        }