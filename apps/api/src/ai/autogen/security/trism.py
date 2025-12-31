"""
AI TRiSM (Trust, Risk and Security Management) 框架实现
实现Trust、Risk、Security三大组件的综合安全框架
"""

import asyncio
import re
import json
import hashlib
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from textblob import TextBlob

from src.core.logging import get_logger
logger = get_logger(__name__)

class ThreatLevel(str, Enum):
    """威胁级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TrustLevel(str, Enum):
    """信任级别"""
    UNTRUSTED = "untrusted"
    LOW_TRUST = "low_trust"
    MEDIUM_TRUST = "medium_trust"
    HIGH_TRUST = "high_trust"
    VERIFIED = "verified"

class RiskCategory(str, Enum):
    """风险类别"""
    BIAS = "bias"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_LEAKAGE = "privacy_leakage"
    MISINFORMATION = "misinformation"
    MANIPULATION = "manipulation"

@dataclass
class SecurityEvent:
    """安全事件数据结构"""
    event_id: str
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source_agent: str
    details: Dict[str, Any]
    mitigation_actions: List[str]
    trust_impact: float = 0.0
    risk_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "threat_level": self.threat_level.value,
            "source_agent": self.source_agent,
            "details": self.details,
            "mitigation_actions": self.mitigation_actions,
            "trust_impact": self.trust_impact,
            "risk_score": self.risk_score
        }

@dataclass
class TrustMetrics:
    """信任度量指标"""
    consistency_score: float = 0.0
    transparency_score: float = 0.0
    reliability_score: float = 0.0
    explainability_score: float = 0.0
    historical_performance: float = 0.0
    
    def calculate_overall_trust(self) -> float:
        """计算综合信任度"""
        weights = {
            'consistency': 0.25,
            'transparency': 0.20,
            'reliability': 0.25,
            'explainability': 0.15,
            'historical': 0.15
        }
        
        return (
            self.consistency_score * weights['consistency'] +
            self.transparency_score * weights['transparency'] +
            self.reliability_score * weights['reliability'] +
            self.explainability_score * weights['explainability'] +
            self.historical_performance * weights['historical']
        )

@dataclass
class RiskAssessment:
    """风险评估结果"""
    overall_risk: float = 0.0
    category_risks: Dict[RiskCategory, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    mitigation_recommendations: List[str] = field(default_factory=list)
    
    def get_risk_level(self) -> ThreatLevel:
        """获取风险级别"""
        if self.overall_risk >= 0.8:
            return ThreatLevel.CRITICAL
        elif self.overall_risk >= 0.6:
            return ThreatLevel.HIGH
        elif self.overall_risk >= 0.3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

class TrustModule:
    """信任管理模块"""
    
    def __init__(self):
        self.agent_trust_history: Dict[str, List[float]] = {}
        self.explanation_quality_cache: Dict[str, float] = {}
        self.consistency_patterns: Dict[str, List[str]] = {}
        
    async def evaluate_trust(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估智能体输出的可信度"""
        try:
            metrics = TrustMetrics()
            
            # 评估一致性
            metrics.consistency_score = await self._evaluate_consistency(agent_id, output, context)
            
            # 评估透明度
            metrics.transparency_score = await self._evaluate_transparency(output, context)
            
            # 评估可靠性
            metrics.reliability_score = await self._evaluate_reliability(agent_id, context)
            
            # 评估可解释性
            metrics.explainability_score = await self._evaluate_explainability(output, context)
            
            # 评估历史表现
            metrics.historical_performance = await self._evaluate_historical_performance(agent_id)
            
            # 计算整体信任度
            overall_trust = metrics.calculate_overall_trust()
            
            # 更新信任历史
            self._update_trust_history(agent_id, overall_trust)
            
            return {
                "trust_score": overall_trust,
                "trust_level": self._get_trust_level(overall_trust),
                "metrics": {
                    "consistency": metrics.consistency_score,
                    "transparency": metrics.transparency_score,
                    "reliability": metrics.reliability_score,
                    "explainability": metrics.explainability_score,
                    "historical": metrics.historical_performance
                },
                "recommendations": self._generate_trust_recommendations(metrics)
            }
            
        except Exception as e:
            logger.error("信任评估失败", agent_id=agent_id, error=str(e))
            return {
                "trust_score": 0.0,
                "trust_level": TrustLevel.UNTRUSTED,
                "error": str(e)
            }
    
    async def _evaluate_consistency(self, agent_id: str, output: str, context: Dict[str, Any]) -> float:
        """评估输出一致性"""
        if agent_id not in self.consistency_patterns:
            self.consistency_patterns[agent_id] = []
        
        # 提取关键模式
        patterns = self._extract_output_patterns(output)
        agent_patterns = self.consistency_patterns[agent_id]
        
        if not agent_patterns:
            # 第一次输出，记录模式
            self.consistency_patterns[agent_id] = patterns
            return 0.8  # 默认较高一致性
        
        # 计算模式相似度
        similarity = self._calculate_pattern_similarity(patterns, agent_patterns)
        
        # 更新模式历史
        self.consistency_patterns[agent_id].extend(patterns)
        if len(self.consistency_patterns[agent_id]) > 50:
            self.consistency_patterns[agent_id] = self.consistency_patterns[agent_id][-50:]
        
        return min(1.0, similarity)
    
    async def _evaluate_transparency(self, output: str, context: Dict[str, Any]) -> float:
        """评估输出透明度"""
        transparency_score = 0.0
        
        # 检查是否包含推理过程
        reasoning_indicators = [
            "because", "since", "due to", "therefore", "thus",
            "reasoning", "analysis", "conclusion", "evidence"
        ]
        
        reasoning_count = sum(1 for indicator in reasoning_indicators 
                             if indicator.lower() in output.lower())
        transparency_score += min(0.4, reasoning_count * 0.1)
        
        # 检查是否引用了信息源
        source_indicators = ["according to", "based on", "reference", "source"]
        source_count = sum(1 for indicator in source_indicators 
                          if indicator.lower() in output.lower())
        transparency_score += min(0.3, source_count * 0.15)
        
        # 检查不确定性表达
        uncertainty_indicators = ["might", "could", "possibly", "likely", "uncertain"]
        uncertainty_count = sum(1 for indicator in uncertainty_indicators 
                               if indicator.lower() in output.lower())
        transparency_score += min(0.3, uncertainty_count * 0.1)
        
        return min(1.0, transparency_score)
    
    async def _evaluate_reliability(self, agent_id: str, context: Dict[str, Any]) -> float:
        """评估可靠性"""
        reliability_score = 0.8  # 基础可靠性
        
        # 基于任务成功率
        task_success_rate = context.get("task_success_rate", 0.8)
        reliability_score *= task_success_rate
        
        # 基于响应时间稳定性
        response_time_variance = context.get("response_time_variance", 0.1)
        time_stability = max(0.5, 1.0 - response_time_variance)
        reliability_score *= time_stability
        
        # 基于错误频率
        error_rate = context.get("error_rate", 0.0)
        error_penalty = max(0.0, 1.0 - error_rate * 2)
        reliability_score *= error_penalty
        
        return min(1.0, max(0.0, reliability_score))
    
    async def _evaluate_explainability(self, output: str, context: Dict[str, Any]) -> float:
        """评估可解释性"""
        # 缓存检查
        output_hash = hashlib.md5(output.encode()).hexdigest()
        if output_hash in self.explanation_quality_cache:
            return self.explanation_quality_cache[output_hash]
        
        explainability_score = 0.0
        
        # 文本长度合理性（既不太短也不太长）
        text_length = len(output.split())
        if 10 <= text_length <= 200:
            explainability_score += 0.3
        elif text_length > 200:
            explainability_score += max(0.1, 0.3 - (text_length - 200) * 0.001)
        
        # 结构化程度
        structured_indicators = ["\n", "1.", "2.", "•", "-", ":", "；"]
        structure_count = sum(1 for indicator in structured_indicators 
                             if indicator in output)
        explainability_score += min(0.4, structure_count * 0.1)
        
        # 复杂度适中
        try:
            blob = TextBlob(output)
            avg_sentence_length = len(output.split()) / max(1, len(blob.sentences))
            if 8 <= avg_sentence_length <= 25:
                explainability_score += 0.3
            else:
                explainability_score += max(0.1, 0.3 - abs(avg_sentence_length - 16.5) * 0.01)
        except:
            explainability_score += 0.2  # 默认分数
        
        final_score = min(1.0, explainability_score)
        self.explanation_quality_cache[output_hash] = final_score
        
        # 限制缓存大小
        if len(self.explanation_quality_cache) > 1000:
            # 移除最旧的条目
            oldest_keys = list(self.explanation_quality_cache.keys())[:100]
            for key in oldest_keys:
                del self.explanation_quality_cache[key]
        
        return final_score
    
    async def _evaluate_historical_performance(self, agent_id: str) -> float:
        """评估历史表现"""
        if agent_id not in self.agent_trust_history:
            return 0.5  # 新智能体默认中等信任
        
        history = self.agent_trust_history[agent_id]
        if not history:
            return 0.5
        
        # 计算加权平均（最近的权重更高）
        weights = np.exp(-0.1 * np.arange(len(history))[::-1])
        weights = weights / weights.sum()
        
        weighted_avg = np.average(history, weights=weights)
        return float(weighted_avg)
    
    def _extract_output_patterns(self, output: str) -> List[str]:
        """提取输出模式"""
        patterns = []
        
        # 句子长度模式
        sentences = output.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        patterns.append(f"avg_sentence_length:{avg_sentence_length:.1f}")
        
        # 标点符号使用模式
        punctuation_ratio = len(re.findall(r'[,.!?;:]', output)) / max(1, len(output))
        patterns.append(f"punctuation_ratio:{punctuation_ratio:.3f}")
        
        # 词汇复杂度
        words = output.split()
        avg_word_length = np.mean([len(word) for word in words])
        patterns.append(f"avg_word_length:{avg_word_length:.1f}")
        
        return patterns
    
    def _calculate_pattern_similarity(self, patterns1: List[str], patterns2: List[str]) -> float:
        """计算模式相似度"""
        if not patterns1 or not patterns2:
            return 0.0
        
        # 提取数值
        def extract_values(patterns):
            values = {}
            for pattern in patterns:
                if ':' in pattern:
                    key, value = pattern.split(':', 1)
                    try:
                        values[key] = float(value)
                    except ValueError:
                        continue
            return values
        
        values1 = extract_values(patterns1)
        values2 = extract_values(patterns2)
        
        # 计算相似度
        similarities = []
        for key in set(values1.keys()) & set(values2.keys()):
            v1, v2 = values1[key], values2[key]
            if v1 == 0 and v2 == 0:
                similarity = 1.0
            else:
                similarity = 1.0 - abs(v1 - v2) / max(abs(v1), abs(v2), 1.0)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _update_trust_history(self, agent_id: str, trust_score: float):
        """更新信任历史"""
        if agent_id not in self.agent_trust_history:
            self.agent_trust_history[agent_id] = []
        
        self.agent_trust_history[agent_id].append(trust_score)
        
        # 保持历史记录在合理范围内
        if len(self.agent_trust_history[agent_id]) > 100:
            self.agent_trust_history[agent_id] = self.agent_trust_history[agent_id][-100:]
    
    def _get_trust_level(self, trust_score: float) -> TrustLevel:
        """获取信任级别"""
        if trust_score >= 0.9:
            return TrustLevel.VERIFIED
        elif trust_score >= 0.7:
            return TrustLevel.HIGH_TRUST
        elif trust_score >= 0.5:
            return TrustLevel.MEDIUM_TRUST
        elif trust_score >= 0.3:
            return TrustLevel.LOW_TRUST
        else:
            return TrustLevel.UNTRUSTED
    
    def _generate_trust_recommendations(self, metrics: TrustMetrics) -> List[str]:
        """生成信任提升建议"""
        recommendations = []
        
        if metrics.consistency_score < 0.6:
            recommendations.append("提高输出一致性，保持风格和模式的稳定性")
        
        if metrics.transparency_score < 0.6:
            recommendations.append("增加推理过程的透明度，说明决策依据")
        
        if metrics.reliability_score < 0.6:
            recommendations.append("提高任务执行的可靠性，减少错误率")
        
        if metrics.explainability_score < 0.6:
            recommendations.append("改善输出的可解释性，使用更清晰的结构")
        
        if metrics.historical_performance < 0.6:
            recommendations.append("需要更多时间建立可靠的历史表现记录")
        
        return recommendations

class RiskModule:
    """风险管理模块"""
    
    def __init__(self):
        self.bias_detectors = self._initialize_bias_detectors()
        self.harmful_content_filters = self._initialize_harmful_filters()
        self.privacy_patterns = self._initialize_privacy_patterns()
        self.risk_history: Dict[str, List[float]] = {}
        
    async def assess_risk(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估智能体输出的风险级别"""
        try:
            assessment = RiskAssessment()
            
            # 偏见风险评估
            bias_risk = await self._assess_bias_risk(output, context)
            assessment.category_risks[RiskCategory.BIAS] = bias_risk
            
            # 有害内容风险评估
            harmful_risk = await self._assess_harmful_content_risk(output, context)
            assessment.category_risks[RiskCategory.HARMFUL_CONTENT] = harmful_risk
            
            # 隐私泄露风险评估
            privacy_risk = await self._assess_privacy_leakage_risk(output, context)
            assessment.category_risks[RiskCategory.PRIVACY_LEAKAGE] = privacy_risk
            
            # 误信息风险评估
            misinformation_risk = await self._assess_misinformation_risk(output, context)
            assessment.category_risks[RiskCategory.MISINFORMATION] = misinformation_risk
            
            # 操控风险评估
            manipulation_risk = await self._assess_manipulation_risk(output, context)
            assessment.category_risks[RiskCategory.MANIPULATION] = manipulation_risk
            
            # 计算整体风险
            assessment.overall_risk = self._calculate_overall_risk(assessment.category_risks)
            
            # 生成风险因素和缓解建议
            assessment.risk_factors = self._identify_risk_factors(assessment.category_risks)
            assessment.mitigation_recommendations = self._generate_mitigation_recommendations(assessment)
            
            # 更新风险历史
            self._update_risk_history(agent_id, assessment.overall_risk)
            
            return {
                "overall_risk": assessment.overall_risk,
                "risk_level": assessment.get_risk_level(),
                "category_risks": {k.value: v for k, v in assessment.category_risks.items()},
                "risk_factors": assessment.risk_factors,
                "mitigation_recommendations": assessment.mitigation_recommendations,
                "trend": self._analyze_risk_trend(agent_id)
            }
            
        except Exception as e:
            logger.error("风险评估失败", agent_id=agent_id, error=str(e))
            return {
                "overall_risk": 1.0,  # 高风险作为安全默认值
                "risk_level": ThreatLevel.CRITICAL,
                "error": str(e)
            }
    
    async def _assess_bias_risk(self, output: str, context: Dict[str, Any]) -> float:
        """评估偏见风险"""
        bias_score = 0.0
        
        # 检查性别偏见
        gender_bias = self._check_gender_bias(output)
        bias_score = max(bias_score, gender_bias)
        
        # 检查种族偏见
        racial_bias = self._check_racial_bias(output)
        bias_score = max(bias_score, racial_bias)
        
        # 检查年龄偏见
        age_bias = self._check_age_bias(output)
        bias_score = max(bias_score, age_bias)
        
        # 检查职业偏见
        occupation_bias = self._check_occupation_bias(output)
        bias_score = max(bias_score, occupation_bias)
        
        return min(1.0, bias_score)
    
    async def _assess_harmful_content_risk(self, output: str, context: Dict[str, Any]) -> float:
        """评估有害内容风险"""
        harm_score = 0.0
        
        # 检查暴力内容
        violence_score = self._check_violence_content(output)
        harm_score = max(harm_score, violence_score)
        
        # 检查仇恨言论
        hate_score = self._check_hate_speech(output)
        harm_score = max(harm_score, hate_score)
        
        # 检查自伤内容
        self_harm_score = self._check_self_harm_content(output)
        harm_score = max(harm_score, self_harm_score)
        
        # 检查非法活动
        illegal_score = self._check_illegal_content(output)
        harm_score = max(harm_score, illegal_score)
        
        return min(1.0, harm_score)
    
    async def _assess_privacy_leakage_risk(self, output: str, context: Dict[str, Any]) -> float:
        """评估隐私泄露风险"""
        privacy_risk = 0.0
        
        # 检查个人身份信息
        pii_risk = self._check_pii_leakage(output)
        privacy_risk = max(privacy_risk, pii_risk)
        
        # 检查金融信息
        financial_risk = self._check_financial_info(output)
        privacy_risk = max(privacy_risk, financial_risk)
        
        # 检查医疗信息
        medical_risk = self._check_medical_info(output)
        privacy_risk = max(privacy_risk, medical_risk)
        
        return min(1.0, privacy_risk)
    
    async def _assess_misinformation_risk(self, output: str, context: Dict[str, Any]) -> float:
        """评估误信息风险"""
        # 检查事实声明的确定性
        certainty_indicators = ["definitely", "certainly", "absolutely", "proven fact"]
        factual_claims = sum(1 for indicator in certainty_indicators 
                           if indicator.lower() in output.lower())
        
        # 检查是否缺乏来源引用
        source_indicators = ["according to", "source:", "reference", "study shows"]
        has_sources = any(indicator.lower() in output.lower() 
                         for indicator in source_indicators)
        
        misinformation_risk = 0.0
        
        # 高确定性声明但无来源支持
        if factual_claims > 2 and not has_sources:
            misinformation_risk += 0.6
        
        # 检查争议性话题
        controversial_topics = ["vaccine", "climate change", "election", "politics"]
        controversial_mentions = sum(1 for topic in controversial_topics 
                                   if topic.lower() in output.lower())
        
        if controversial_mentions > 0 and not has_sources:
            misinformation_risk += 0.4
        
        return min(1.0, misinformation_risk)
    
    async def _assess_manipulation_risk(self, output: str, context: Dict[str, Any]) -> float:
        """评估操控风险"""
        manipulation_risk = 0.0
        
        # 检查情感操控语言
        emotional_words = ["fear", "panic", "urgent", "crisis", "disaster", "terrible"]
        emotional_count = sum(1 for word in emotional_words 
                             if word.lower() in output.lower())
        manipulation_risk += min(0.5, emotional_count * 0.1)
        
        # 检查说服技巧
        persuasion_patterns = [
            "you must", "you should immediately", "don't miss out",
            "limited time", "exclusive offer", "act now"
        ]
        persuasion_count = sum(1 for pattern in persuasion_patterns 
                              if pattern.lower() in output.lower())
        manipulation_risk += min(0.4, persuasion_count * 0.15)
        
        # 检查权威性声称
        authority_claims = ["experts say", "scientists prove", "studies show"]
        authority_count = sum(1 for claim in authority_claims 
                             if claim.lower() in output.lower())
        
        # 权威声称但无具体引用
        if authority_count > 0:
            specific_refs = ["published in", "journal", "research by"]
            has_specific_refs = any(ref.lower() in output.lower() 
                                  for ref in specific_refs)
            if not has_specific_refs:
                manipulation_risk += 0.3
        
        return min(1.0, manipulation_risk)
    
    def _calculate_overall_risk(self, category_risks: Dict[RiskCategory, float]) -> float:
        """计算整体风险评分"""
        if not category_risks:
            return 0.0
        
        # 风险权重
        weights = {
            RiskCategory.HARMFUL_CONTENT: 0.3,
            RiskCategory.PRIVACY_LEAKAGE: 0.25,
            RiskCategory.BIAS: 0.2,
            RiskCategory.MANIPULATION: 0.15,
            RiskCategory.MISINFORMATION: 0.1
        }
        
        weighted_sum = sum(
            risk * weights.get(category, 0.1) 
            for category, risk in category_risks.items()
        )
        
        # 考虑最高单项风险
        max_single_risk = max(category_risks.values()) if category_risks else 0.0
        
        # 综合评分：加权平均 + 最高风险惩罚
        overall_risk = (weighted_sum * 0.7) + (max_single_risk * 0.3)
        
        return min(1.0, overall_risk)
    
    def _initialize_bias_detectors(self) -> Dict[str, List[str]]:
        """初始化偏见检测器"""
        return {
            "gender_bias": [
                "all women are", "all men are", "typical woman", "typical man",
                "women always", "men always", "female drivers", "male ego"
            ],
            "racial_bias": [
                "all asians", "all blacks", "all whites", "typical of their race",
                "racial stereotype", "ethnic characteristic"
            ],
            "age_bias": [
                "old people are", "young people are", "millennials are",
                "boomers are", "kids these days", "elderly drivers"
            ],
            "occupation_bias": [
                "typical lawyer", "typical doctor", "typical teacher",
                "all politicians", "all engineers are"
            ]
        }
    
    def _initialize_harmful_filters(self) -> Dict[str, List[str]]:
        """初始化有害内容过滤器"""
        return {
            "violence": [
                "kill", "murder", "shoot", "stab", "attack", "violence",
                "fight", "beat up", "assault"
            ],
            "hate_speech": [
                "hate", "discriminate", "inferior race", "genetic superiority",
                "ethnic cleansing", "religious war"
            ],
            "self_harm": [
                "suicide", "self-harm", "cut yourself", "end your life",
                "kill yourself", "harm yourself"
            ],
            "illegal": [
                "drug dealing", "money laundering", "tax evasion",
                "identity theft", "fraud", "illegal weapons"
            ]
        }
    
    def _initialize_privacy_patterns(self) -> Dict[str, str]:
        """初始化隐私模式检测"""
        return {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "address": r"\b\d+\s+\w+\s+(street|st|avenue|ave|road|rd|lane|ln)\b"
        }
    
    def _check_gender_bias(self, text: str) -> float:
        """检查性别偏见"""
        bias_patterns = self.bias_detectors["gender_bias"]
        count = sum(1 for pattern in bias_patterns if pattern.lower() in text.lower())
        return min(1.0, count * 0.3)
    
    def _check_racial_bias(self, text: str) -> float:
        """检查种族偏见"""
        bias_patterns = self.bias_detectors["racial_bias"]
        count = sum(1 for pattern in bias_patterns if pattern.lower() in text.lower())
        return min(1.0, count * 0.4)
    
    def _check_age_bias(self, text: str) -> float:
        """检查年龄偏见"""
        bias_patterns = self.bias_detectors["age_bias"]
        count = sum(1 for pattern in bias_patterns if pattern.lower() in text.lower())
        return min(1.0, count * 0.3)
    
    def _check_occupation_bias(self, text: str) -> float:
        """检查职业偏见"""
        bias_patterns = self.bias_detectors["occupation_bias"]
        count = sum(1 for pattern in bias_patterns if pattern.lower() in text.lower())
        return min(1.0, count * 0.3)
    
    def _check_violence_content(self, text: str) -> float:
        """检查暴力内容"""
        violence_words = self.harmful_content_filters["violence"]
        count = sum(1 for word in violence_words if word.lower() in text.lower())
        return min(1.0, count * 0.2)
    
    def _check_hate_speech(self, text: str) -> float:
        """检查仇恨言论"""
        hate_words = self.harmful_content_filters["hate_speech"]
        count = sum(1 for word in hate_words if word.lower() in text.lower())
        return min(1.0, count * 0.4)
    
    def _check_self_harm_content(self, text: str) -> float:
        """检查自伤内容"""
        self_harm_words = self.harmful_content_filters["self_harm"]
        count = sum(1 for word in self_harm_words if word.lower() in text.lower())
        return min(1.0, count * 0.5)
    
    def _check_illegal_content(self, text: str) -> float:
        """检查非法活动内容"""
        illegal_words = self.harmful_content_filters["illegal"]
        count = sum(1 for word in illegal_words if word.lower() in text.lower())
        return min(1.0, count * 0.4)
    
    def _check_pii_leakage(self, text: str) -> float:
        """检查个人身份信息泄露"""
        pii_risk = 0.0
        
        for pii_type, pattern in self.privacy_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if pii_type in ["ssn", "credit_card"]:
                    pii_risk = 1.0  # 严重泄露
                    break
                else:
                    pii_risk += 0.3
        
        return min(1.0, pii_risk)
    
    def _check_financial_info(self, text: str) -> float:
        """检查金融信息"""
        financial_patterns = ["bank account", "routing number", "pin number", "password"]
        count = sum(1 for pattern in financial_patterns if pattern.lower() in text.lower())
        return min(1.0, count * 0.4)
    
    def _check_medical_info(self, text: str) -> float:
        """检查医疗信息"""
        medical_patterns = ["medical record", "diagnosis", "prescription", "patient id"]
        count = sum(1 for pattern in medical_patterns if pattern.lower() in text.lower())
        return min(1.0, count * 0.3)
    
    def _identify_risk_factors(self, category_risks: Dict[RiskCategory, float]) -> List[str]:
        """识别风险因素"""
        factors = []
        
        for category, risk in category_risks.items():
            if risk > 0.6:
                factors.append(f"高{category.value}风险: {risk:.2f}")
            elif risk > 0.3:
                factors.append(f"中等{category.value}风险: {risk:.2f}")
        
        return factors
    
    def _generate_mitigation_recommendations(self, assessment: RiskAssessment) -> List[str]:
        """生成风险缓解建议"""
        recommendations = []
        
        for category, risk in assessment.category_risks.items():
            if risk > 0.5:
                if category == RiskCategory.BIAS:
                    recommendations.append("加强偏见检测和公平性训练")
                elif category == RiskCategory.HARMFUL_CONTENT:
                    recommendations.append("强化内容安全过滤机制")
                elif category == RiskCategory.PRIVACY_LEAKAGE:
                    recommendations.append("实施严格的隐私保护措施")
                elif category == RiskCategory.MISINFORMATION:
                    recommendations.append("要求引用可靠来源和事实验证")
                elif category == RiskCategory.MANIPULATION:
                    recommendations.append("审查说服性语言和情感操控")
        
        if assessment.overall_risk > 0.7:
            recommendations.append("考虑人工审核或限制输出使用")
        
        return recommendations
    
    def _update_risk_history(self, agent_id: str, risk_score: float):
        """更新风险历史"""
        if agent_id not in self.risk_history:
            self.risk_history[agent_id] = []
        
        self.risk_history[agent_id].append(risk_score)
        
        # 保持历史记录在合理范围内
        if len(self.risk_history[agent_id]) > 100:
            self.risk_history[agent_id] = self.risk_history[agent_id][-100:]
    
    def _analyze_risk_trend(self, agent_id: str) -> str:
        """分析风险趋势"""
        if agent_id not in self.risk_history or len(self.risk_history[agent_id]) < 3:
            return "insufficient_data"
        
        recent_risks = self.risk_history[agent_id][-10:]
        if len(recent_risks) < 3:
            return "insufficient_data"
        
        # 计算趋势
        x = np.arange(len(recent_risks))
        slope = np.polyfit(x, recent_risks, 1)[0]
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

class SecurityModule:
    """安全管理模块"""
    
    def __init__(self):
        self.access_control_rules: Dict[str, List[str]] = {}
        self.data_classification: Dict[str, str] = {}
        self.encryption_policies: Dict[str, bool] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        
    async def security_scan(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行安全扫描"""
        try:
            violations = []
            
            # 访问控制检查
            access_violations = await self._check_access_control(agent_id, context)
            violations.extend(access_violations)
            
            # 数据分类和处理检查
            data_violations = await self._check_data_classification(output, context)
            violations.extend(data_violations)
            
            # 加密策略检查
            encryption_violations = await self._check_encryption_policy(output, context)
            violations.extend(encryption_violations)
            
            # 输出过滤检查
            filter_violations = await self._check_output_filtering(output, context)
            violations.extend(filter_violations)
            
            # 记录审计日志
            await self._log_security_audit(agent_id, output, context, violations)
            
            return {
                "violations": violations,
                "security_score": self._calculate_security_score(violations),
                "recommendations": self._generate_security_recommendations(violations)
            }
            
        except Exception as e:
            logger.error("安全扫描失败", agent_id=agent_id, error=str(e))
            return {
                "violations": [{"type": "scan_error", "severity": "high", "details": str(e)}],
                "security_score": 0.0,
                "error": str(e)
            }
    
    async def _check_access_control(self, agent_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查访问控制"""
        violations = []
        
        # 检查智能体权限
        required_permissions = context.get("required_permissions", [])
        agent_permissions = self.access_control_rules.get(agent_id, [])
        
        for permission in required_permissions:
            if permission not in agent_permissions:
                violations.append({
                    "type": "access_control",
                    "severity": "high",
                    "details": f"智能体 {agent_id} 缺少权限: {permission}"
                })
        
        return violations
    
    async def _check_data_classification(self, output: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查数据分类和处理"""
        violations = []
        
        # 检查敏感数据处理
        data_types = context.get("data_types", [])
        
        for data_type in data_types:
            classification = self.data_classification.get(data_type, "public")
            
            if classification in ["confidential", "restricted"]:
                # 检查是否在输出中泄露了敏感数据
                if self._contains_sensitive_data(output, data_type):
                    violations.append({
                        "type": "data_classification",
                        "severity": "critical",
                        "details": f"输出可能包含{classification}级别的{data_type}数据"
                    })
        
        return violations
    
    async def _check_encryption_policy(self, output: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查加密策略"""
        violations = []
        
        # 检查是否需要加密但未加密
        requires_encryption = context.get("requires_encryption", False)
        is_encrypted = context.get("is_encrypted", False)
        
        if requires_encryption and not is_encrypted:
            violations.append({
                "type": "encryption_policy",
                "severity": "high",
                "details": "输出需要加密但未执行加密"
            })
        
        return violations
    
    async def _check_output_filtering(self, output: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查输出过滤"""
        violations = []
        
        # 检查是否包含系统内部信息
        internal_patterns = [
            "api_key", "secret", "password", "token", "internal_id",
            "system_path", "config_file", "debug_info"
        ]
        
        for pattern in internal_patterns:
            if pattern.lower() in output.lower():
                violations.append({
                    "type": "output_filtering",
                    "severity": "high",
                    "details": f"输出可能包含系统内部信息: {pattern}"
                })
        
        return violations
    
    def _contains_sensitive_data(self, text: str, data_type: str) -> bool:
        """检查是否包含敏感数据"""
        sensitive_patterns = {
            "personal_info": [r"\b\d{3}-\d{2}-\d{4}\b", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "financial": [r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"],
            "medical": ["medical record", "patient", "diagnosis"],
            "legal": ["case number", "legal document", "confidential"]
        }
        
        patterns = sensitive_patterns.get(data_type, [])
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_security_score(self, violations: List[Dict[str, Any]]) -> float:
        """计算安全评分"""
        if not violations:
            return 1.0
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }
        
        total_penalty = sum(
            severity_weights.get(v.get("severity", "medium"), 0.4)
            for v in violations
        )
        
        # 基础分数减去违规惩罚
        security_score = max(0.0, 1.0 - min(1.0, total_penalty / 3.0))
        
        return security_score
    
    def _generate_security_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        violation_types = set(v.get("type") for v in violations)
        
        if "access_control" in violation_types:
            recommendations.append("验证和更新智能体访问权限")
        
        if "data_classification" in violation_types:
            recommendations.append("加强敏感数据识别和保护")
        
        if "encryption_policy" in violation_types:
            recommendations.append("实施数据加密策略")
        
        if "output_filtering" in violation_types:
            recommendations.append("增强输出内容过滤机制")
        
        return recommendations
    
    async def _log_security_audit(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any], 
        violations: List[Dict[str, Any]]
    ):
        """记录安全审计日志"""
        audit_entry = {
            "timestamp": utc_now().isoformat(),
            "agent_id": agent_id,
            "output_hash": hashlib.md5(output.encode()).hexdigest(),
            "violations_count": len(violations),
            "severity_levels": [v.get("severity") for v in violations],
            "violation_types": [v.get("type") for v in violations]
        }
        
        self.audit_trail.append(audit_entry)
        
        # 保持审计日志在合理大小
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-5000:]

class AITRiSMFramework:
    """AI Trust, Risk and Security Management Framework"""
    
    def __init__(self):
        self.trust_module = TrustModule()
        self.risk_module = RiskModule()
        self.security_module = SecurityModule()
        self.event_handlers: Dict[str, Callable] = {}
        self.security_policies: Dict[str, Any] = {}
        
        logger.info("AI TRiSM 框架初始化完成")
    
    async def evaluate_agent_output(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估智能体输出的安全性和可信度"""
        try:
            # 并行执行三个模块的评估
            trust_task = self.trust_module.evaluate_trust(agent_id, output, context)
            risk_task = self.risk_module.assess_risk(agent_id, output, context)
            security_task = self.security_module.security_scan(agent_id, output, context)
            
            trust_result, risk_result, security_result = await asyncio.gather(
                trust_task, risk_task, security_task, return_exceptions=True
            )
            
            # 处理异常结果
            if isinstance(trust_result, Exception):
                logger.error("信任评估异常", error=str(trust_result))
                trust_result = {"trust_score": 0.0, "trust_level": TrustLevel.UNTRUSTED}
            
            if isinstance(risk_result, Exception):
                logger.error("风险评估异常", error=str(risk_result))
                risk_result = {"overall_risk": 1.0, "risk_level": ThreatLevel.CRITICAL}
            
            if isinstance(security_result, Exception):
                logger.error("安全扫描异常", error=str(security_result))
                security_result = {"violations": [], "security_score": 0.0}
            
            # 综合评估结果
            evaluation_result = {
                "timestamp": utc_now().isoformat(),
                "agent_id": agent_id,
                "trust": trust_result,
                "risk": risk_result,
                "security": security_result,
                "overall_assessment": self._calculate_overall_assessment(
                    trust_result, risk_result, security_result
                ),
                "recommended_actions": self._generate_recommended_actions(
                    trust_result, risk_result, security_result
                )
            }
            
            # 触发事件处理器
            await self._trigger_event_handlers(agent_id, evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            logger.error("TRiSM框架评估失败", agent_id=agent_id, error=str(e))
            return {
                "timestamp": utc_now().isoformat(),
                "agent_id": agent_id,
                "error": str(e),
                "overall_assessment": {
                    "safe_to_use": False,
                    "confidence": 0.0,
                    "threat_level": ThreatLevel.CRITICAL
                }
            }
    
    def _calculate_overall_assessment(
        self, 
        trust_result: Dict[str, Any], 
        risk_result: Dict[str, Any], 
        security_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算综合评估结果"""
        trust_score = trust_result.get("trust_score", 0.0)
        risk_score = risk_result.get("overall_risk", 1.0)
        security_score = security_result.get("security_score", 0.0)
        
        # 综合信心度评分
        confidence = (trust_score * 0.4 + security_score * 0.4 + (1 - risk_score) * 0.2)
        
        # 安全使用判断
        safe_to_use = (
            trust_score >= 0.5 and 
            risk_score <= 0.6 and 
            security_score >= 0.7 and
            confidence >= 0.6
        )
        
        # 威胁级别
        threat_level = ThreatLevel.LOW
        if risk_score >= 0.8 or security_score <= 0.3:
            threat_level = ThreatLevel.CRITICAL
        elif risk_score >= 0.6 or security_score <= 0.5:
            threat_level = ThreatLevel.HIGH
        elif risk_score >= 0.4 or security_score <= 0.7:
            threat_level = ThreatLevel.MEDIUM
        
        return {
            "safe_to_use": safe_to_use,
            "confidence": confidence,
            "threat_level": threat_level,
            "component_scores": {
                "trust": trust_score,
                "risk": risk_score,
                "security": security_score
            }
        }
    
    def _generate_recommended_actions(
        self, 
        trust_result: Dict[str, Any], 
        risk_result: Dict[str, Any], 
        security_result: Dict[str, Any]
    ) -> List[str]:
        """生成推荐行动"""
        actions = []
        
        # 信任度建议
        trust_score = trust_result.get("trust_score", 0.0)
        if trust_score < 0.5:
            actions.append("increase_monitoring")
            actions.extend(trust_result.get("recommendations", []))
        
        # 风险处理建议
        risk_level = risk_result.get("risk_level", ThreatLevel.LOW)
        if risk_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            actions.append("block_output")
            actions.extend(risk_result.get("mitigation_recommendations", []))
        elif risk_level == ThreatLevel.MEDIUM:
            actions.append("require_review")
        
        # 安全处理建议
        security_violations = security_result.get("violations", [])
        if security_violations:
            actions.append("security_review")
            actions.extend(security_result.get("recommendations", []))
        
        return list(set(actions))  # 去重
    
    async def _trigger_event_handlers(self, agent_id: str, evaluation_result: Dict[str, Any]):
        """触发事件处理器"""
        for event_type, handler in self.event_handlers.items():
            try:
                await handler(agent_id, evaluation_result)
            except Exception as e:
                logger.error("事件处理器异常", event_type=event_type, error=str(e))
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.event_handlers[event_type] = handler
        logger.info("注册TRiSM事件处理器", event_type=event_type)
    
    def set_security_policy(self, policy_name: str, policy_config: Dict[str, Any]):
        """设置安全策略"""
        self.security_policies[policy_name] = policy_config
        logger.info("设置安全策略", policy=policy_name)
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """获取框架统计信息"""
        return {
            "trust_module_agents": len(self.trust_module.agent_trust_history),
            "risk_module_agents": len(self.risk_module.risk_history),
            "security_audit_entries": len(self.security_module.audit_trail),
            "registered_handlers": len(self.event_handlers),
            "security_policies": len(self.security_policies)
        }
