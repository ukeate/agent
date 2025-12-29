"""
Relationship Dynamics Analyzer
人际关系动态分析器

This module provides comprehensive relationship analysis capabilities:
- Relationship type identification
- Intimacy level calculation
- Power dynamics analysis
- Emotional support pattern recognition
- Conflict and harmony detection
"""

from src.core.utils.timezone_utils import utc_now
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import timedelta
from collections import defaultdict, Counter
import asyncio
import statistics
from .relationship_models import (
    RelationshipDynamics,
    RelationshipType,
    IntimacyLevel,
    PowerDynamics,
    EmotionalSupportPattern,
    ConflictIndicator,
    RelationshipMilestone,
    RelationshipProfile,
    RelationshipAnalysisConfig,
    SupportType,
    ConflictStyle,
    generate_relationship_id,
    generate_support_id,
    generate_conflict_id,
    generate_milestone_id,
    classify_intimacy_level,
    classify_power_dynamics
)
from .group_emotion_models import EmotionState

class RelationshipDynamicsAnalyzer:
    """关系动态分析器"""
    
    def __init__(self, config: Optional[RelationshipAnalysisConfig] = None):
        self.config = config or RelationshipAnalysisConfig()
        
        # 关系类型识别特征
        self.relationship_features = {
            RelationshipType.ROMANTIC: {
                'intimacy_threshold': 0.8,
                'trust_threshold': 0.7,
                'exclusivity_indicators': ['private_conversations', 'personal_disclosure'],
                'emotional_intensity': 0.8
            },
            RelationshipType.FAMILY: {
                'intimacy_threshold': 0.6,
                'trust_threshold': 0.8,
                'support_frequency': 'high',
                'obligation_indicators': ['family_terms', 'inherited_connection']
            },
            RelationshipType.FRIENDSHIP: {
                'intimacy_threshold': 0.5,
                'reciprocity_threshold': 0.7,
                'shared_interests': True,
                'voluntary_interaction': True
            },
            RelationshipType.PROFESSIONAL: {
                'formality_level': 0.6,
                'task_orientation': 0.8,
                'hierarchy_indicators': ['title_usage', 'formal_language'],
                'goal_alignment': True
            },
            RelationshipType.MENTORSHIP: {
                'knowledge_transfer': True,
                'guidance_provision': 0.8,
                'power_imbalance': 0.4,
                'development_focus': True
            },
            RelationshipType.ACQUAINTANCE: {
                'intimacy_threshold': 0.3,
                'interaction_frequency': 'low',
                'surface_level_topics': True,
                'limited_personal_sharing': True
            }
        }
        
        # 支持行为识别模式
        self.support_patterns = {
            'emotional': [
                'empathy', 'validation', 'comfort', 'encouragement',
                'listening', 'understanding', 'sympathy'
            ],
            'informational': [
                'advice', 'guidance', 'information', 'explanation',
                'suggestion', 'recommendation', 'knowledge'
            ],
            'instrumental': [
                'help', 'assistance', 'resource', 'tool',
                'action', 'practical', 'tangible'
            ],
            'appraisal': [
                'feedback', 'evaluation', 'assessment', 'perspective',
                'opinion', 'judgment', 'reflection'
            ]
        }
        
        # 冲突指标关键词
        self.conflict_indicators = {
            'disagreement': ['disagree', 'oppose', 'conflict', 'argue'],
            'criticism': ['criticize', 'blame', 'fault', 'wrong'],
            'defensive': ['defensive', 'justify', 'excuse', 'deny'],
            'withdrawal': ['ignore', 'silent', 'avoid', 'withdraw'],
            'escalation': ['angry', 'frustrated', 'hostile', 'aggressive']
        }
        
        # 和谐指标关键词
        self.harmony_indicators = {
            'agreement': ['agree', 'support', 'understand', 'accept'],
            'collaboration': ['together', 'cooperate', 'team', 'joint'],
            'appreciation': ['thank', 'appreciate', 'grateful', 'value'],
            'affection': ['love', 'care', 'like', 'fond'],
            'respect': ['respect', 'admire', 'honor', 'esteem']
        }
    
    async def analyze_relationship_dynamics(
        self,
        participant1_id: str,
        participant2_id: str,
        participant1_emotions: List[EmotionState],
        participant2_emotions: List[EmotionState],
        interaction_history: List[Dict[str, Any]]
    ) -> RelationshipDynamics:
        """分析关系动态"""
        
        # 识别关系类型
        relationship_type = await self._identify_relationship_type(
            participant1_id, participant2_id, interaction_history
        )
        
        # 分析亲密程度
        intimacy_analysis = self._analyze_intimacy_level(
            participant1_id, participant2_id, interaction_history
        )
        
        # 分析权力平衡
        power_analysis = self._analyze_power_balance(
            participant1_id, participant2_id, interaction_history
        )
        
        # 分析情感互惠性
        reciprocity_analysis = self._analyze_emotional_reciprocity(
            participant1_id, participant2_id, 
            participant1_emotions, participant2_emotions, interaction_history
        )
        
        # 识别支持模式
        support_patterns = await self._identify_support_patterns(
            participant1_id, participant2_id, interaction_history
        )
        
        # 检测冲突指标
        conflict_analysis = self._detect_conflict_indicators(
            participant1_id, participant2_id, interaction_history
        )
        
        # 检测和谐指标
        harmony_indicators = self._detect_harmony_indicators(interaction_history)
        
        # 计算关系健康度
        relationship_health = self._calculate_relationship_health(
            intimacy_analysis, power_analysis, reciprocity_analysis,
            support_patterns, conflict_analysis
        )
        
        # 预测发展趋势
        development_trend = self._predict_relationship_trend(
            interaction_history, relationship_health
        )
        
        # 识别重要里程碑
        milestones = self._identify_relationship_milestones(
            participant1_id, participant2_id, interaction_history
        )
        
        return RelationshipDynamics(
            relationship_id=generate_relationship_id(participant1_id, participant2_id),
            participants=[participant1_id, participant2_id],
            relationship_type=relationship_type,
            
            # 亲密度
            intimacy_level=classify_intimacy_level(intimacy_analysis['intimacy_score']),
            intimacy_score=intimacy_analysis['intimacy_score'],
            trust_level=intimacy_analysis['trust_level'],
            vulnerability_sharing=intimacy_analysis['vulnerability_sharing'],
            
            # 权力关系
            power_balance=power_analysis['power_balance'],
            power_dynamics=classify_power_dynamics(power_analysis['power_balance']),
            influence_patterns=power_analysis['influence_patterns'],
            
            # 情感互惠性
            emotional_reciprocity=reciprocity_analysis['emotional_reciprocity'],
            support_balance=reciprocity_analysis['support_balance'],
            empathy_symmetry=reciprocity_analysis['empathy_symmetry'],
            
            # 支持模式
            support_patterns=support_patterns,
            primary_support_giver=self._identify_primary_support_giver(support_patterns),
            support_network_strength=self._calculate_support_network_strength(support_patterns),
            
            # 冲突分析
            conflict_indicators=conflict_analysis['indicators'],
            conflict_frequency=conflict_analysis['frequency'],
            conflict_resolution_rate=conflict_analysis['resolution_rate'],
            harmony_indicators=harmony_indicators,
            
            # 关系健康度
            relationship_health=relationship_health['overall_health'],
            stability_score=relationship_health['stability_score'],
            satisfaction_level=relationship_health['satisfaction_level'],
            
            # 发展趋势
            development_trend=development_trend,
            relationship_trajectory=self._calculate_relationship_trajectory(interaction_history),
            future_outlook=self._assess_future_outlook(relationship_health, development_trend),
            
            # 里程碑
            milestones=milestones,
            significant_events=self._extract_significant_events(interaction_history),
            
            # 元数据
            analysis_timestamp=utc_now(),
            data_quality_score=self._calculate_data_quality(interaction_history),
            confidence_level=self._calculate_analysis_confidence(
                len(interaction_history), relationship_health['overall_health']
            )
        )
    
    async def _identify_relationship_type(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> RelationshipType:
        """识别关系类型"""
        
        if not interaction_history:
            return RelationshipType.ACQUAINTANCE
        
        # 计算各种关系类型的特征分数
        type_scores = {}
        
        for rel_type, features in self.relationship_features.items():
            score = await self._calculate_relationship_type_score(
                rel_type, features, participant1_id, participant2_id, interaction_history
            )
            type_scores[rel_type] = score
        
        # 返回得分最高的关系类型
        best_type = max(type_scores.items(), key=lambda x: x[1])
        
        # 如果最高分数过低，则归类为熟人关系
        if best_type[1] < 0.3:
            return RelationshipType.ACQUAINTANCE
        
        return best_type[0]
    
    async def _calculate_relationship_type_score(
        self,
        rel_type: RelationshipType,
        features: Dict[str, Any],
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> float:
        """计算关系类型分数"""
        
        score = 0.0
        total_features = 0
        
        # 计算亲密度特征
        if 'intimacy_threshold' in features:
            intimacy_analysis = self._analyze_intimacy_level(
                participant1_id, participant2_id, interaction_history
            )
            intimacy_score = intimacy_analysis['intimacy_score']
            if intimacy_score >= features['intimacy_threshold']:
                score += 0.3
            total_features += 0.3
        
        # 计算信任度特征
        if 'trust_threshold' in features:
            trust_level = self._calculate_trust_level(interaction_history)
            if trust_level >= features['trust_threshold']:
                score += 0.25
            total_features += 0.25
        
        # 检查正式性等级
        if 'formality_level' in features:
            formality = self._calculate_formality_level(interaction_history)
            if formality >= features['formality_level']:
                score += 0.2
            total_features += 0.2
        
        # 检查任务导向性
        if 'task_orientation' in features:
            task_orientation = self._calculate_task_orientation(interaction_history)
            if task_orientation >= features['task_orientation']:
                score += 0.15
            total_features += 0.15
        
        # 检查其他指标
        if 'voluntary_interaction' in features:
            if self._is_voluntary_interaction(interaction_history):
                score += 0.1
            total_features += 0.1
        
        return score / max(total_features, 1.0)
    
    def _analyze_intimacy_level(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """分析亲密程度"""
        
        intimacy_indicators = {
            'personal_disclosure': 0.0,
            'emotional_support': 0.0,
            'shared_experiences': 0.0,
            'communication_frequency': 0.0
        }
        
        if not interaction_history:
            return {
                'intimacy_score': 0.0,
                'trust_level': 0.0,
                'vulnerability_sharing': 0.0
            }
        
        for interaction in interaction_history:
            # 检测个人信息披露
            content = interaction.get('content', '').lower()
            personal_keywords = ['personal', 'private', 'family', 'feel', 'emotion', 'secret']
            if any(keyword in content for keyword in personal_keywords):
                intimacy_indicators['personal_disclosure'] += 1
            
            # 检测情感支持行为
            if interaction.get('emotion_support_provided', False):
                intimacy_indicators['emotional_support'] += 1
            
            # 检测共享经历
            shared_keywords = ['remember', 'together', 'shared', 'experience', 'memory']
            if any(keyword in content for keyword in shared_keywords):
                intimacy_indicators['shared_experiences'] += 1
        
        # 计算沟通频率
        if interaction_history:
            time_span = (interaction_history[-1]['timestamp'] - 
                        interaction_history[0]['timestamp']).days
            intimacy_indicators['communication_frequency'] = len(interaction_history) / max(time_span, 1)
        
        # 归一化并加权计算
        weights = self.config.__dict__
        
        normalized_indicators = {}
        for indicator, value in intimacy_indicators.items():
            if indicator == 'communication_frequency':
                normalized_indicators[indicator] = min(value / 5, 1.0)  # 每天5次为满分
            else:
                normalized_indicators[indicator] = min(value / 10, 1.0)  # 10次为满分
        
        intimacy_score = sum(
            normalized_indicators[indicator] * weights.get(f"{indicator}_weight", 0.25)
            for indicator in normalized_indicators
        )
        
        # 计算信任水平
        trust_level = self._calculate_trust_level(interaction_history)
        
        # 计算脆弱性分享
        vulnerability_sharing = normalized_indicators['personal_disclosure']
        
        return {
            'intimacy_score': min(intimacy_score, 1.0),
            'trust_level': trust_level,
            'vulnerability_sharing': vulnerability_sharing
        }
    
    def _analyze_power_balance(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析权力平衡"""
        
        if not interaction_history:
            return {
                'power_balance': 0.0,
                'influence_patterns': {participant1_id: 0.5, participant2_id: 0.5}
            }
        
        # 统计主导行为指标
        dominance_indicators = {participant1_id: 0, participant2_id: 0}
        influence_counts = {participant1_id: 0, participant2_id: 0}
        
        for interaction in interaction_history:
            sender = interaction.get('sender_id')
            if sender not in [participant1_id, participant2_id]:
                continue
            
            # 统计发起对话次数
            if interaction.get('is_conversation_starter', False):
                dominance_indicators[sender] += 2
            
            # 统计指令性语言
            content = interaction.get('content', '').lower()
            command_keywords = ['should', 'must', 'need to', 'have to', 'let me', 'i think']
            command_count = sum(1 for keyword in command_keywords if keyword in content)
            dominance_indicators[sender] += command_count
            
            # 统计获得回应的频率（影响力指标）
            if interaction.get('response_count', 0) > 0:
                influence_counts[sender] += interaction['response_count']
        
        # 计算权力平衡分数
        total_dominance = sum(dominance_indicators.values())
        if total_dominance == 0:
            power_balance = 0.0
        else:
            p1_dominance = dominance_indicators[participant1_id] / total_dominance
            p2_dominance = dominance_indicators[participant2_id] / total_dominance
            power_balance = p1_dominance - p2_dominance
        
        # 计算影响模式
        total_influence = sum(influence_counts.values())
        if total_influence == 0:
            influence_patterns = {participant1_id: 0.5, participant2_id: 0.5}
        else:
            influence_patterns = {
                participant1_id: influence_counts[participant1_id] / total_influence,
                participant2_id: influence_counts[participant2_id] / total_influence
            }
        
        return {
            'power_balance': power_balance,
            'influence_patterns': influence_patterns
        }
    
    def _analyze_emotional_reciprocity(
        self,
        participant1_id: str,
        participant2_id: str,
        participant1_emotions: List[EmotionState],
        participant2_emotions: List[EmotionState],
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """分析情感互惠性"""
        
        if not participant1_emotions or not participant2_emotions:
            return {
                'emotional_reciprocity': 0.5,
                'support_balance': 0.0,
                'empathy_symmetry': 0.5
            }
        
        # 计算情感相似性
        emotion_similarity = self._calculate_emotion_similarity(
            participant1_emotions, participant2_emotions
        )
        
        # 计算支持平衡
        support_balance = self._calculate_support_balance(
            participant1_id, participant2_id, interaction_history
        )
        
        # 计算同理心对称性
        empathy_symmetry = self._calculate_empathy_symmetry(
            participant1_id, participant2_id, interaction_history
        )
        
        # 综合计算情感互惠性
        emotional_reciprocity = (
            emotion_similarity * 0.4 +
            abs(support_balance) * 0.3 +  # 支持平衡越接近0越好
            empathy_symmetry * 0.3
        )
        
        return {
            'emotional_reciprocity': emotional_reciprocity,
            'support_balance': support_balance,
            'empathy_symmetry': empathy_symmetry
        }
    
    async def _identify_support_patterns(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> List[EmotionalSupportPattern]:
        """识别支持模式"""
        
        support_patterns = []
        
        for interaction in interaction_history:
            sender = interaction.get('sender_id')
            if sender not in [participant1_id, participant2_id]:
                continue
            
            receiver = participant2_id if sender == participant1_id else participant1_id
            content = interaction.get('content', '').lower()
            
            # 识别支持类型
            support_types = []
            for support_type, keywords in self.support_patterns.items():
                if any(keyword in content for keyword in keywords):
                    support_types.append(support_type)
            
            if support_types:
                # 创建支持模式
                for support_type_name in support_types:
                    support_type = getattr(SupportType, support_type_name.upper())
                    
                    pattern = EmotionalSupportPattern(
                        support_id=generate_support_id(),
                        giver_id=sender,
                        receiver_id=receiver,
                        support_type=support_type,
                        frequency=1,  # 单次支持
                        intensity=interaction.get('emotion_intensity', 0.5),
                        reciprocity_score=0.0,  # 稍后计算
                        effectiveness_score=0.0,  # 稍后计算
                        timestamp=interaction.get('timestamp', utc_now()),
                        
                        # 支持行为特征
                        verbal_affirmation='affirm' in content or 'positive' in content,
                        active_listening='listen' in content or 'understand' in content,
                        empathy_expression='feel' in content or 'empathy' in content,
                        problem_solving='solve' in content or 'solution' in content,
                        resource_sharing='help' in content or 'resource' in content
                    )
                    
                    support_patterns.append(pattern)
        
        # 计算互惠性和有效性分数
        for pattern in support_patterns:
            pattern.reciprocity_score = self._calculate_support_reciprocity(
                pattern, support_patterns
            )
            pattern.effectiveness_score = self._estimate_support_effectiveness(pattern)
        
        return support_patterns
    
    def _detect_conflict_indicators(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """检测冲突指标"""
        
        conflict_indicators = []
        conflict_count = 0
        resolution_count = 0
        
        for interaction in interaction_history:
            content = interaction.get('content', '').lower()
            sender = interaction.get('sender_id')
            
            # 检测冲突指标
            conflict_scores = {}
            for conflict_type, keywords in self.conflict_indicators.items():
                score = sum(1 for keyword in keywords if keyword in content)
                if score > 0:
                    conflict_scores[conflict_type] = score
            
            if conflict_scores:
                # 计算冲突严重程度
                severity = min(sum(conflict_scores.values()) / 10.0, 1.0)
                
                if severity > 0.3:  # 显著冲突
                    indicator = ConflictIndicator(
                        indicator_id=generate_conflict_id(),
                        participants=[participant1_id, participant2_id],
                        conflict_type=max(conflict_scores.items(), key=lambda x: x[1])[0],
                        severity_level=severity,
                        escalation_risk=self._calculate_escalation_risk(
                            conflict_scores, interaction_history
                        ),
                        resolution_potential=self._calculate_resolution_potential(
                            interaction, interaction_history
                        ),
                        timestamp=interaction.get('timestamp', utc_now()),
                        
                        # 冲突表现
                        verbal_disagreement='disagree' in content or 'wrong' in content,
                        emotional_tension='tense' in content or 'upset' in content,
                        communication_breakdown='ignore' in content or 'silent' in content,
                        value_conflict='believe' in content or 'value' in content,
                        resource_competition='compete' in content or 'mine' in content,
                        
                        conflict_styles={sender: self._identify_conflict_style(content)}
                    )
                    
                    conflict_indicators.append(indicator)
                    conflict_count += 1
                    
                    # 检查是否有解决迹象
                    resolution_keywords = ['sorry', 'apologize', 'understand', 'agree', 'resolve']
                    if any(keyword in content for keyword in resolution_keywords):
                        resolution_count += 1
        
        # 计算冲突频率和解决率
        total_interactions = len(interaction_history)
        conflict_frequency = conflict_count / max(total_interactions, 1)
        conflict_resolution_rate = resolution_count / max(conflict_count, 1)
        
        return {
            'indicators': conflict_indicators,
            'frequency': conflict_frequency,
            'resolution_rate': conflict_resolution_rate
        }
    
    def _detect_harmony_indicators(
        self,
        interaction_history: List[Dict[str, Any]]
    ) -> List[str]:
        """检测和谐指标"""
        
        harmony_indicators = []
        
        for interaction in interaction_history:
            content = interaction.get('content', '').lower()
            
            for harmony_type, keywords in self.harmony_indicators.items():
                if any(keyword in content for keyword in keywords):
                    harmony_indicators.append(harmony_type)
        
        # 去重并统计频率
        harmony_counts = Counter(harmony_indicators)
        return list(harmony_counts.keys())
    
    def _calculate_relationship_health(
        self,
        intimacy_analysis: Dict[str, float],
        power_analysis: Dict[str, Any],
        reciprocity_analysis: Dict[str, float],
        support_patterns: List[EmotionalSupportPattern],
        conflict_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算关系健康度"""
        
        # 各维度健康分数
        intimacy_health = intimacy_analysis['intimacy_score'] * intimacy_analysis['trust_level']
        
        # 权力平衡健康度（越平衡越健康）
        power_health = 1.0 - abs(power_analysis['power_balance'])
        
        # 互惠性健康度
        reciprocity_health = reciprocity_analysis['emotional_reciprocity']
        
        # 支持网络健康度
        support_health = min(len(support_patterns) / 10.0, 1.0) if support_patterns else 0.0
        
        # 冲突管理健康度
        conflict_health = 1.0 - conflict_analysis['frequency'] + conflict_analysis['resolution_rate']
        conflict_health = max(0.0, min(1.0, conflict_health))
        
        # 加权计算整体健康度
        weights = self.config.__dict__
        overall_health = (
            intimacy_health * weights.get('intimacy_health_weight', 0.25) +
            power_health * weights.get('trust_health_weight', 0.25) +  # 使用trust权重代替power权重
            reciprocity_health * weights.get('reciprocity_health_weight', 0.2) +
            support_health * 0.15 +
            conflict_health * weights.get('conflict_management_weight', 0.15)
        )
        
        # 计算稳定性分数
        stability_score = (power_health + reciprocity_health + conflict_health) / 3
        
        # 计算满意度水平
        satisfaction_level = (intimacy_health + support_health + conflict_health) / 3
        
        return {
            'overall_health': min(max(overall_health, 0.0), 1.0),
            'stability_score': min(max(stability_score, 0.0), 1.0),
            'satisfaction_level': min(max(satisfaction_level, 0.0), 1.0)
        }
    
    def _predict_relationship_trend(
        self,
        interaction_history: List[Dict[str, Any]],
        relationship_health: Dict[str, float]
    ) -> str:
        """预测关系发展趋势"""
        
        if not interaction_history or len(interaction_history) < 5:
            return "stable"
        
        # 分析最近的交互模式
        recent_interactions = interaction_history[-10:]  # 最近10次交互
        
        # 计算情感强度趋势
        emotion_intensities = [
            interaction.get('emotion_intensity', 0.5)
            for interaction in recent_interactions
        ]
        
        if len(emotion_intensities) >= 3:
            # 简单线性趋势分析
            x = np.arange(len(emotion_intensities))
            slope = np.polyfit(x, emotion_intensities, 1)[0]
            
            # 结合健康分数判断趋势
            health_score = relationship_health['overall_health']
            
            if slope > 0.02 and health_score > 0.6:
                return "improving"
            elif slope < -0.02 or health_score < 0.4:
                return "declining"
            else:
                return "stable"
        
        return "stable"
    
    # 辅助方法
    def _calculate_trust_level(self, interaction_history: List[Dict[str, Any]]) -> float:
        """计算信任水平"""
        trust_indicators = 0
        total_interactions = len(interaction_history)
        
        if total_interactions == 0:
            return 0.5
        
        for interaction in interaction_history:
            content = interaction.get('content', '').lower()
            trust_keywords = ['trust', 'honest', 'reliable', 'depend', 'confide']
            if any(keyword in content for keyword in trust_keywords):
                trust_indicators += 1
        
        return min(trust_indicators / max(total_interactions * 0.1, 1), 1.0)
    
    def _calculate_formality_level(self, interaction_history: List[Dict[str, Any]]) -> float:
        """计算正式性水平"""
        formal_indicators = 0
        total_interactions = len(interaction_history)
        
        for interaction in interaction_history:
            content = interaction.get('content', '')
            # 检查正式语言特征
            if any(indicator in content.lower() for indicator in ['sir', 'madam', 'please', 'thank you']):
                formal_indicators += 1
        
        return formal_indicators / max(total_interactions, 1)
    
    def _calculate_task_orientation(self, interaction_history: List[Dict[str, Any]]) -> float:
        """计算任务导向性"""
        task_indicators = 0
        total_interactions = len(interaction_history)
        
        for interaction in interaction_history:
            content = interaction.get('content', '').lower()
            task_keywords = ['project', 'task', 'work', 'deadline', 'goal', 'objective']
            if any(keyword in content for keyword in task_keywords):
                task_indicators += 1
        
        return task_indicators / max(total_interactions, 1)
    
    def _is_voluntary_interaction(self, interaction_history: List[Dict[str, Any]]) -> bool:
        """判断是否为自愿交互"""
        # 简化实现：检查交互是否包含选择性语言
        for interaction in interaction_history:
            content = interaction.get('content', '').lower()
            if any(word in content for word in ['want', 'choose', 'like', 'enjoy']):
                return True
        return False
    
    def _calculate_emotion_similarity(
        self,
        emotions1: List[EmotionState],
        emotions2: List[EmotionState]
    ) -> float:
        """计算情感相似性"""
        
        if not emotions1 or not emotions2:
            return 0.5
        
        # 计算VAD维度的相似性
        similarities = []
        
        for e1 in emotions1[-5:]:  # 最近5个情感状态
            for e2 in emotions2[-5:]:
                valence_sim = 1 - abs(e1.valence - e2.valence) / 2
                arousal_sim = 1 - abs(e1.arousal - e2.arousal)
                dominance_sim = 1 - abs(e1.dominance - e2.dominance)
                
                overall_sim = (valence_sim + arousal_sim + dominance_sim) / 3
                similarities.append(overall_sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_support_balance(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> float:
        """计算支持平衡"""
        
        support_given = {participant1_id: 0, participant2_id: 0}
        
        for interaction in interaction_history:
            sender = interaction.get('sender_id')
            if sender in support_given:
                content = interaction.get('content', '').lower()
                support_keywords = ['help', 'support', 'comfort', 'encourage']
                if any(keyword in content for keyword in support_keywords):
                    support_given[sender] += 1
        
        total_support = sum(support_given.values())
        if total_support == 0:
            return 0.0
        
        p1_support = support_given[participant1_id] / total_support
        p2_support = support_given[participant2_id] / total_support
        
        return p1_support - p2_support  # [-1, 1] 范围，0表示平衡
    
    def _calculate_empathy_symmetry(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> float:
        """计算同理心对称性"""
        
        empathy_expressions = {participant1_id: 0, participant2_id: 0}
        
        for interaction in interaction_history:
            sender = interaction.get('sender_id')
            if sender in empathy_expressions:
                content = interaction.get('content', '').lower()
                empathy_keywords = ['understand', 'feel', 'empathy', 'relate']
                empathy_count = sum(1 for keyword in empathy_keywords if keyword in content)
                empathy_expressions[sender] += empathy_count
        
        total_empathy = sum(empathy_expressions.values())
        if total_empathy == 0:
            return 0.5
        
        # 计算对称性：两人表达的同理心越接近，对称性越高
        p1_empathy = empathy_expressions[participant1_id] / total_empathy
        p2_empathy = empathy_expressions[participant2_id] / total_empathy
        
        symmetry = 1.0 - abs(p1_empathy - p2_empathy)
        return symmetry
    
    def _calculate_support_reciprocity(
        self,
        pattern: EmotionalSupportPattern,
        all_patterns: List[EmotionalSupportPattern]
    ) -> float:
        """计算支持互惠性"""
        
        # 查找反向支持
        reverse_support_count = 0
        for other_pattern in all_patterns:
            if (other_pattern.giver_id == pattern.receiver_id and
                other_pattern.receiver_id == pattern.giver_id and
                other_pattern.support_type == pattern.support_type):
                reverse_support_count += 1
        
        # 计算互惠性分数
        return min(reverse_support_count / 5.0, 1.0)
    
    def _estimate_support_effectiveness(self, pattern: EmotionalSupportPattern) -> float:
        """估算支持有效性"""
        
        # 基于支持特征计算有效性
        effectiveness_factors = [
            pattern.verbal_affirmation,
            pattern.active_listening,
            pattern.empathy_expression,
            pattern.problem_solving,
            pattern.resource_sharing
        ]
        
        base_effectiveness = sum(effectiveness_factors) / len(effectiveness_factors)
        
        # 结合强度调整
        return base_effectiveness * pattern.intensity
    
    def _identify_primary_support_giver(
        self,
        support_patterns: List[EmotionalSupportPattern]
    ) -> Optional[str]:
        """识别主要支持提供者"""
        
        if not support_patterns:
            return None
        
        giver_counts = Counter(pattern.giver_id for pattern in support_patterns)
        return giver_counts.most_common(1)[0][0] if giver_counts else None
    
    def _calculate_support_network_strength(
        self,
        support_patterns: List[EmotionalSupportPattern]
    ) -> float:
        """计算支持网络强度"""
        
        if not support_patterns:
            return 0.0
        
        # 考虑支持频次、强度和多样性
        total_intensity = sum(pattern.intensity for pattern in support_patterns)
        support_types = set(pattern.support_type for pattern in support_patterns)
        
        strength = (
            len(support_patterns) * 0.4 +  # 频次
            total_intensity * 0.4 +  # 强度
            len(support_types) * 0.2  # 多样性
        )
        
        return min(strength / 10.0, 1.0)
    
    def _calculate_escalation_risk(
        self,
        conflict_scores: Dict[str, int],
        interaction_history: List[Dict[str, Any]]
    ) -> float:
        """计算冲突升级风险"""
        
        # 基于冲突类型和历史模式评估风险
        high_risk_types = ['escalation', 'defensive']
        risk_score = sum(
            conflict_scores.get(risk_type, 0) for risk_type in high_risk_types
        )
        
        return min(risk_score / 5.0, 1.0)
    
    def _calculate_resolution_potential(
        self,
        current_interaction: Dict[str, Any],
        interaction_history: List[Dict[str, Any]]
    ) -> float:
        """计算解决潜力"""
        
        # 检查解决导向的语言
        content = current_interaction.get('content', '').lower()
        resolution_keywords = ['solve', 'resolve', 'work together', 'compromise']
        
        resolution_score = sum(1 for keyword in resolution_keywords if keyword in content)
        return min(resolution_score / 3.0, 1.0)
    
    def _identify_conflict_style(self, content: str) -> ConflictStyle:
        """识别冲突风格"""
        
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['win', 'must', 'should', 'wrong']):
            return ConflictStyle.COMPETING
        elif any(word in content_lower for word in ['together', 'both', 'cooperate']):
            return ConflictStyle.COLLABORATING
        elif any(word in content_lower for word in ['compromise', 'middle', 'meet']):
            return ConflictStyle.COMPROMISING
        elif any(word in content_lower for word in ['avoid', 'ignore', 'later']):
            return ConflictStyle.AVOIDING
        elif any(word in content_lower for word in ['okay', 'fine', 'whatever']):
            return ConflictStyle.ACCOMMODATING
        
        return ConflictStyle.COMPROMISING  # 默认
    
    def _identify_relationship_milestones(
        self,
        participant1_id: str,
        participant2_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> List[RelationshipMilestone]:
        """识别关系里程碑"""
        
        milestones = []
        
        for interaction in interaction_history:
            content = interaction.get('content', '').lower()
            
            # 识别积极里程碑
            positive_keywords = ['first time', 'celebrate', 'achievement', 'success', 'happy']
            if any(keyword in content for keyword in positive_keywords):
                milestone = RelationshipMilestone(
                    milestone_id=generate_milestone_id(),
                    relationship_id=generate_relationship_id(participant1_id, participant2_id),
                    milestone_type="positive_event",
                    significance_level=0.7,
                    emotional_impact=0.8,
                    relationship_change=0.3,
                    timestamp=interaction.get('timestamp', utc_now()),
                    description=content[:100] + "..." if len(content) > 100 else content,
                    positive_milestone=True,
                    relationship_deepening=True
                )
                milestones.append(milestone)
        
        return milestones
    
    def _extract_significant_events(
        self,
        interaction_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """提取重要事件"""
        
        significant_events = []
        
        for interaction in interaction_history:
            # 检查事件重要性指标
            emotion_intensity = interaction.get('emotion_intensity', 0.0)
            response_count = interaction.get('response_count', 0)
            
            if emotion_intensity > 0.7 or response_count > 3:
                event = {
                    'timestamp': interaction.get('timestamp'),
                    'type': 'high_impact_interaction',
                    'emotion_intensity': emotion_intensity,
                    'response_count': response_count,
                    'content_preview': interaction.get('content', '')[:50] + "..."
                }
                significant_events.append(event)
        
        return significant_events
    
    def _calculate_relationship_trajectory(
        self,
        interaction_history: List[Dict[str, Any]]
    ) -> List[float]:
        """计算关系轨迹"""
        
        if not interaction_history:
            return []
        
        # 按时间分段计算关系质量
        trajectory = []
        window_size = max(len(interaction_history) // 10, 1)
        
        for i in range(0, len(interaction_history), window_size):
            window = interaction_history[i:i + window_size]
            
            # 计算该时间段的平均情感强度
            avg_intensity = np.mean([
                interaction.get('emotion_intensity', 0.5) 
                for interaction in window
            ])
            
            trajectory.append(avg_intensity)
        
        return trajectory
    
    def _assess_future_outlook(
        self,
        relationship_health: Dict[str, float],
        development_trend: str
    ) -> str:
        """评估未来展望"""
        
        health_score = relationship_health['overall_health']
        
        if development_trend == "improving" and health_score > 0.6:
            return "very_positive"
        elif development_trend == "improving" or health_score > 0.7:
            return "positive"
        elif development_trend == "declining" or health_score < 0.3:
            return "concerning"
        elif health_score < 0.5:
            return "cautious"
        else:
            return "stable"
    
    def _calculate_data_quality(self, interaction_history: List[Dict[str, Any]]) -> float:
        """计算数据质量"""
        
        if not interaction_history:
            return 0.0
        
        # 检查数据完整性
        complete_interactions = 0
        for interaction in interaction_history:
            required_fields = ['sender_id', 'content', 'timestamp']
            if all(field in interaction for field in required_fields):
                complete_interactions += 1
        
        completeness = complete_interactions / len(interaction_history)
        
        # 检查数据量充足性
        adequacy = min(len(interaction_history) / self.config.min_interaction_threshold, 1.0)
        
        return (completeness + adequacy) / 2
    
    def _calculate_analysis_confidence(
        self,
        interaction_count: int,
        relationship_health: float
    ) -> float:
        """计算分析置信度"""
        
        # 基于数据量和结果一致性
        data_confidence = min(interaction_count / 10, 1.0)
        
        # 基于健康度的合理性（极值降低置信度）
        health_confidence = 1.0 - abs(relationship_health - 0.5) * 0.5
        
        overall_confidence = (data_confidence + health_confidence) / 2
        
        return max(self.config.confidence_threshold, min(overall_confidence, 1.0))
