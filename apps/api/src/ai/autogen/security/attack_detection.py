"""
对抗攻击检测器集合
实现Prompt注入、数据泄露、模型中毒等攻击检测机制
"""
import re
import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import structlog
import numpy as np
from collections import defaultdict, deque

from .trism import ThreatLevel, SecurityEvent

logger = structlog.get_logger(__name__)


class AttackType(str, Enum):
    """攻击类型"""
    PROMPT_INJECTION = "prompt_injection"
    DATA_LEAKAGE = "data_leakage"
    MODEL_POISONING = "model_poisoning"
    ADVERSARIAL_INPUT = "adversarial_input"
    JAILBREAK = "jailbreak"
    EVASION = "evasion"


@dataclass
class DetectionResult:
    """检测结果"""
    attack_detected: bool
    attack_type: AttackType
    confidence: float
    threat_level: ThreatLevel
    details: Dict[str, Any]
    evidence: List[str]
    mitigation_suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_detected": self.attack_detected,
            "attack_type": self.attack_type.value,
            "confidence": self.confidence,
            "threat_level": self.threat_level.value,
            "details": self.details,
            "evidence": self.evidence,
            "mitigation_suggestions": self.mitigation_suggestions
        }


class PromptInjectionDetector:
    """Prompt注入攻击检测器"""
    
    def __init__(self):
        self.injection_patterns = self._initialize_injection_patterns()
        self.system_prompts_cache: Set[str] = set()
        self.detection_history: deque = deque(maxlen=1000)
        
    async def detect(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> DetectionResult:
        """检测Prompt注入攻击"""
        try:
            evidence = []
            confidence = 0.0
            details = {}
            
            # 检查恶意指令模式
            malicious_score, malicious_evidence = self._check_malicious_patterns(output)
            confidence += malicious_score
            evidence.extend(malicious_evidence)
            details["malicious_patterns"] = malicious_evidence
            
            # 检查系统提示泄露
            system_leak_score, system_evidence = self._check_system_prompt_leakage(output, context)
            confidence += system_leak_score
            evidence.extend(system_evidence)
            details["system_leakage"] = system_evidence
            
            # 检查角色混淆
            role_confusion_score, role_evidence = self._check_role_confusion(output, context)
            confidence += role_confusion_score
            evidence.extend(role_evidence)
            details["role_confusion"] = role_evidence
            
            # 检查逃逸尝试
            escape_score, escape_evidence = self._check_escape_attempts(output)
            confidence += escape_score
            evidence.extend(escape_evidence)
            details["escape_attempts"] = escape_evidence
            
            # 检查上下文注入
            context_injection_score, context_evidence = self._check_context_injection(output, context)
            confidence += context_injection_score
            evidence.extend(context_evidence)
            details["context_injection"] = context_evidence
            
            # 综合评估
            final_confidence = min(1.0, confidence / 5.0)
            attack_detected = final_confidence > 0.5
            threat_level = self._calculate_threat_level(final_confidence)
            
            # 记录检测历史
            self._record_detection(agent_id, attack_detected, final_confidence)
            
            result = DetectionResult(
                attack_detected=attack_detected,
                attack_type=AttackType.PROMPT_INJECTION,
                confidence=final_confidence,
                threat_level=threat_level,
                details=details,
                evidence=evidence,
                mitigation_suggestions=self._generate_mitigation_suggestions(evidence)
            )
            
            logger.info(
                "Prompt注入检测完成",
                agent_id=agent_id,
                detected=attack_detected,
                confidence=final_confidence
            )
            
            return result
            
        except Exception as e:
            logger.error("Prompt注入检测失败", agent_id=agent_id, error=str(e))
            return DetectionResult(
                attack_detected=True,  # 失败时保守处理
                attack_type=AttackType.PROMPT_INJECTION,
                confidence=1.0,
                threat_level=ThreatLevel.HIGH,
                details={"error": str(e)},
                evidence=["detection_error"],
                mitigation_suggestions=["manual_review_required"]
            )
    
    def _initialize_injection_patterns(self) -> Dict[str, List[str]]:
        """初始化注入模式"""
        return {
            "instruction_override": [
                "ignore previous instructions",
                "disregard the above",
                "forget what i told you",
                "new instructions:",
                "system override",
                "admin mode",
                "debug mode",
                "override safety",
                "bypass restrictions"
            ],
            "role_manipulation": [
                "act as if you are",
                "pretend to be",
                "roleplay as",
                "you are now",
                "switch to mode",
                "become a",
                "transform into"
            ],
            "system_exploration": [
                "show me your prompt",
                "what are your instructions",
                "print your system message",
                "reveal your guidelines",
                "display configuration",
                "show internal rules"
            ],
            "jailbreak_attempts": [
                "dan mode",
                "developer mode",
                "unrestricted mode",
                "no safety",
                "without limitations",
                "ignore ethics",
                "break character"
            ],
            "escape_sequences": [
                "\\n\\n---\\n\\n",
                "```\\nSystem:",
                "<!-- inject:",
                "<|system|>",
                "[SYSTEM]",
                "{{system}}",
                "${system}"
            ]
        }
    
    def _check_malicious_patterns(self, text: str) -> Tuple[float, List[str]]:
        """检查恶意模式"""
        evidence = []
        score = 0.0
        
        for category, patterns in self.injection_patterns.items():
            category_matches = []
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    category_matches.append(pattern)
                    score += 0.2
            
            if category_matches:
                evidence.append(f"{category}: {', '.join(category_matches)}")
        
        return min(1.0, score), evidence
    
    def _check_system_prompt_leakage(self, text: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查系统提示泄露"""
        evidence = []
        score = 0.0
        
        # 检查是否包含系统级关键词
        system_keywords = [
            "system prompt", "instructions", "guidelines", "rules",
            "configuration", "settings", "parameters", "constraints"
        ]
        
        for keyword in system_keywords:
            if keyword.lower() in text.lower():
                evidence.append(f"系统关键词: {keyword}")
                score += 0.15
        
        # 检查格式化的系统信息
        system_formats = [
            r"System:\s*[\w\s]+",
            r"Instructions:\s*[\w\s]+",
            r"Role:\s*[\w\s]+",
            r"Guidelines:\s*[\w\s]+"
        ]
        
        for pattern in system_formats:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                evidence.append(f"系统格式泄露: {len(matches)}个匹配")
                score += 0.3
        
        return min(1.0, score), evidence
    
    def _check_role_confusion(self, text: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查角色混淆"""
        evidence = []
        score = 0.0
        
        # 检查是否声称自己是其他角色
        role_claims = [
            "i am a human", "i am not an ai", "i am your assistant",
            "i am the developer", "i am admin", "i have access to"
        ]
        
        for claim in role_claims:
            if claim.lower() in text.lower():
                evidence.append(f"角色声称: {claim}")
                score += 0.25
        
        # 检查人格切换指示
        personality_switches = [
            "now i will", "switching to", "changing mode",
            "different personality", "new character"
        ]
        
        for switch in personality_switches:
            if switch.lower() in text.lower():
                evidence.append(f"人格切换: {switch}")
                score += 0.2
        
        return min(1.0, score), evidence
    
    def _check_escape_attempts(self, text: str) -> Tuple[float, List[str]]:
        """检查逃逸尝试"""
        evidence = []
        score = 0.0
        
        # 检查特殊字符序列
        escape_sequences = [
            "\\n", "\\r", "\\t", "\\\\", "\\'", '\\"',
            "{{", "}}", "${", "<%", "%>", "<!--", "-->"
        ]
        
        sequence_count = 0
        for seq in escape_sequences:
            count = text.count(seq)
            sequence_count += count
        
        if sequence_count > 5:
            evidence.append(f"大量转义序列: {sequence_count}")
            score += min(0.5, sequence_count * 0.05)
        
        # 检查编码尝试
        encoding_patterns = [
            r"&#\d+;",  # HTML实体编码
            r"%[0-9A-Fa-f]{2}",  # URL编码
            r"\\u[0-9A-Fa-f]{4}",  # Unicode编码
            r"\\x[0-9A-Fa-f]{2}"  # 十六进制编码
        ]
        
        for pattern in encoding_patterns:
            matches = re.findall(pattern, text)
            if matches:
                evidence.append(f"编码尝试: {pattern} ({len(matches)}个)")
                score += 0.3
        
        return min(1.0, score), evidence
    
    def _check_context_injection(self, text: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查上下文注入"""
        evidence = []
        score = 0.0
        
        # 检查是否试图修改上下文
        context_modification = [
            "previous conversation", "chat history", "context window",
            "memory", "conversation log", "session data"
        ]
        
        for mod in context_modification:
            if mod.lower() in text.lower():
                evidence.append(f"上下文修改尝试: {mod}")
                score += 0.2
        
        # 检查多轮对话注入
        multi_turn_patterns = [
            "human:", "ai:", "assistant:", "user:",
            "q:", "a:", "question:", "answer:"
        ]
        
        pattern_count = sum(1 for pattern in multi_turn_patterns 
                           if pattern.lower() in text.lower())
        
        if pattern_count > 2:
            evidence.append(f"多轮对话模式: {pattern_count}个")
            score += min(0.4, pattern_count * 0.1)
        
        return min(1.0, score), evidence
    
    def _calculate_threat_level(self, confidence: float) -> ThreatLevel:
        """计算威胁级别"""
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.7:
            return ThreatLevel.HIGH
        elif confidence >= 0.5:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _generate_mitigation_suggestions(self, evidence: List[str]) -> List[str]:
        """生成缓解建议"""
        suggestions = []
        
        if any("instruction" in e.lower() for e in evidence):
            suggestions.append("强化指令隔离机制")
        
        if any("system" in e.lower() for e in evidence):
            suggestions.append("加强系统提示保护")
        
        if any("role" in e.lower() for e in evidence):
            suggestions.append("验证角色一致性")
        
        if any("escape" in e.lower() or "编码" in e for e in evidence):
            suggestions.append("过滤特殊字符和编码")
        
        if any("context" in e.lower() or "对话" in e for e in evidence):
            suggestions.append("限制上下文操作权限")
        
        return suggestions
    
    def _record_detection(self, agent_id: str, detected: bool, confidence: float):
        """记录检测历史"""
        record = {
            "timestamp": datetime.now(timezone.utc),
            "agent_id": agent_id,
            "detected": detected,
            "confidence": confidence
        }
        self.detection_history.append(record)


class DataLeakageDetector:
    """数据泄露检测器"""
    
    def __init__(self):
        self.sensitive_patterns = self._initialize_sensitive_patterns()
        self.context_sensitivity_map: Dict[str, float] = {}
        self.leakage_history: deque = deque(maxlen=1000)
        
    async def detect(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> DetectionResult:
        """检测数据泄露"""
        try:
            evidence = []
            confidence = 0.0
            details = {}
            
            # 检查个人身份信息
            pii_score, pii_evidence = self._check_pii_leakage(output)
            confidence += pii_score
            evidence.extend(pii_evidence)
            details["pii_leakage"] = pii_evidence
            
            # 检查金融信息
            financial_score, financial_evidence = self._check_financial_leakage(output)
            confidence += financial_score
            evidence.extend(financial_evidence)
            details["financial_leakage"] = financial_evidence
            
            # 检查API密钥和令牌
            api_score, api_evidence = self._check_api_keys(output)
            confidence += api_score
            evidence.extend(api_evidence)
            details["api_keys"] = api_evidence
            
            # 检查系统信息泄露
            system_score, system_evidence = self._check_system_info_leakage(output)
            confidence += system_score
            evidence.extend(system_evidence)
            details["system_info"] = system_evidence
            
            # 检查上下文敏感信息
            context_score, context_evidence = self._check_context_sensitive_data(output, context)
            confidence += context_score
            evidence.extend(context_evidence)
            details["context_sensitive"] = context_evidence
            
            # 综合评估
            final_confidence = min(1.0, confidence / 5.0)
            attack_detected = final_confidence > 0.3  # 数据泄露阈值较低
            threat_level = self._calculate_threat_level(final_confidence)
            
            # 记录检测历史
            self._record_detection(agent_id, attack_detected, final_confidence)
            
            result = DetectionResult(
                attack_detected=attack_detected,
                attack_type=AttackType.DATA_LEAKAGE,
                confidence=final_confidence,
                threat_level=threat_level,
                details=details,
                evidence=evidence,
                mitigation_suggestions=self._generate_mitigation_suggestions(evidence)
            )
            
            logger.info(
                "数据泄露检测完成",
                agent_id=agent_id,
                detected=attack_detected,
                confidence=final_confidence
            )
            
            return result
            
        except Exception as e:
            logger.error("数据泄露检测失败", agent_id=agent_id, error=str(e))
            return DetectionResult(
                attack_detected=True,
                attack_type=AttackType.DATA_LEAKAGE,
                confidence=1.0,
                threat_level=ThreatLevel.CRITICAL,
                details={"error": str(e)},
                evidence=["detection_error"],
                mitigation_suggestions=["manual_review_required"]
            )
    
    def _initialize_sensitive_patterns(self) -> Dict[str, List[str]]:
        """初始化敏感信息模式"""
        return {
            "ssn": [r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{9}\b"],
            "credit_card": [r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"],
            "email": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "phone": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", r"\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b"],
            "ip_address": [r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"],
            "api_key": [
                r"\bsk-[A-Za-z0-9]{48}\b",  # OpenAI
                r"\bAIza[A-Za-z0-9_-]{35}\b",  # Google
                r"\b[A-Za-z0-9]{32}\b",  # Generic 32-char
                r"\b[A-Za-z0-9]{64}\b"   # Generic 64-char
            ],
            "aws_key": [
                r"\bAKIA[A-Z0-9]{16}\b",
                r"\b[A-Za-z0-9/+=]{40}\b"
            ],
            "database_uri": [
                r"mongodb://[^\\s]+",
                r"postgresql://[^\\s]+",
                r"mysql://[^\\s]+"
            ]
        }
    
    def _check_pii_leakage(self, text: str) -> Tuple[float, List[str]]:
        """检查个人身份信息泄露"""
        evidence = []
        score = 0.0
        
        # 检查SSN
        ssn_patterns = self.sensitive_patterns["ssn"]
        for pattern in ssn_patterns:
            matches = re.findall(pattern, text)
            if matches:
                evidence.append(f"SSN泄露: {len(matches)}个")
                score += 1.0  # SSN泄露是严重问题
        
        # 检查邮箱
        email_matches = re.findall(self.sensitive_patterns["email"][0], text)
        if email_matches:
            # 过滤掉常见的示例邮箱
            real_emails = [email for email in email_matches 
                          if not any(example in email.lower() 
                                   for example in ["example.com", "test.com", "demo.com"])]
            if real_emails:
                evidence.append(f"邮箱地址泄露: {len(real_emails)}个")
                score += 0.6
        
        # 检查电话号码
        phone_patterns = self.sensitive_patterns["phone"]
        phone_count = 0
        for pattern in phone_patterns:
            phone_count += len(re.findall(pattern, text))
        
        if phone_count > 0:
            evidence.append(f"电话号码泄露: {phone_count}个")
            score += 0.7
        
        return min(1.0, score), evidence
    
    def _check_financial_leakage(self, text: str) -> Tuple[float, List[str]]:
        """检查金融信息泄露"""
        evidence = []
        score = 0.0
        
        # 检查信用卡号
        cc_matches = re.findall(self.sensitive_patterns["credit_card"][0], text)
        if cc_matches:
            evidence.append(f"信用卡号泄露: {len(cc_matches)}个")
            score += 1.0  # 信用卡泄露极其严重
        
        # 检查银行相关信息
        bank_keywords = [
            "account number", "routing number", "bank account",
            "swift code", "iban", "sort code"
        ]
        
        bank_mentions = sum(1 for keyword in bank_keywords 
                           if keyword.lower() in text.lower())
        
        if bank_mentions > 0:
            evidence.append(f"银行信息关键词: {bank_mentions}个")
            score += 0.5
        
        return min(1.0, score), evidence
    
    def _check_api_keys(self, text: str) -> Tuple[float, List[str]]:
        """检查API密钥泄露"""
        evidence = []
        score = 0.0
        
        # 检查通用API密钥格式
        api_patterns = self.sensitive_patterns["api_key"]
        for pattern in api_patterns:
            matches = re.findall(pattern, text)
            if matches:
                evidence.append(f"API密钥格式: {len(matches)}个")
                score += 0.8
        
        # 检查AWS密钥
        aws_patterns = self.sensitive_patterns["aws_key"]
        for pattern in aws_patterns:
            matches = re.findall(pattern, text)
            if matches:
                evidence.append(f"AWS密钥: {len(matches)}个")
                score += 0.9
        
        # 检查通用密钥关键词
        key_keywords = [
            "api_key", "secret_key", "access_token", "private_key",
            "password", "passwd", "pwd", "secret"
        ]
        
        key_mentions = sum(1 for keyword in key_keywords 
                          if keyword.lower() in text.lower())
        
        if key_mentions > 2:
            evidence.append(f"密钥关键词: {key_mentions}个")
            score += 0.4
        
        return min(1.0, score), evidence
    
    def _check_system_info_leakage(self, text: str) -> Tuple[float, List[str]]:
        """检查系统信息泄露"""
        evidence = []
        score = 0.0
        
        # 检查IP地址
        ip_matches = re.findall(self.sensitive_patterns["ip_address"][0], text)
        if ip_matches:
            # 过滤掉公共IP地址
            private_ips = [ip for ip in ip_matches 
                          if not any(ip.startswith(public) 
                                   for public in ["8.8.8", "1.1.1", "208.67.222"])]
            if private_ips:
                evidence.append(f"内部IP地址: {len(private_ips)}个")
                score += 0.6
        
        # 检查数据库连接字符串
        db_patterns = self.sensitive_patterns["database_uri"]
        for pattern in db_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                evidence.append(f"数据库连接字符串: {len(matches)}个")
                score += 0.8
        
        # 检查系统路径
        path_patterns = [
            r"/home/\w+",
            r"C:\\Users\\\w+",
            r"/var/\w+",
            r"/etc/\w+"
        ]
        
        path_count = 0
        for pattern in path_patterns:
            path_count += len(re.findall(pattern, text))
        
        if path_count > 0:
            evidence.append(f"系统路径: {path_count}个")
            score += 0.3
        
        return min(1.0, score), evidence
    
    def _check_context_sensitive_data(self, text: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查上下文敏感数据"""
        evidence = []
        score = 0.0
        
        # 检查是否泄露了训练数据
        training_indicators = [
            "training data", "dataset", "training set",
            "model weights", "parameters", "embeddings"
        ]
        
        training_mentions = sum(1 for indicator in training_indicators 
                               if indicator.lower() in text.lower())
        
        if training_mentions > 0:
            evidence.append(f"训练数据相关: {training_mentions}个")
            score += 0.5
        
        # 检查用户会话信息
        session_info = context.get("session_info", {})
        if session_info:
            for key, value in session_info.items():
                if str(value).lower() in text.lower() and len(str(value)) > 5:
                    evidence.append(f"会话信息泄露: {key}")
                    score += 0.4
        
        return min(1.0, score), evidence
    
    def _calculate_threat_level(self, confidence: float) -> ThreatLevel:
        """计算威胁级别"""
        if confidence >= 0.8:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.5:
            return ThreatLevel.HIGH
        elif confidence >= 0.3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _generate_mitigation_suggestions(self, evidence: List[str]) -> List[str]:
        """生成缓解建议"""
        suggestions = []
        
        if any("SSN" in e or "信用卡" in e for e in evidence):
            suggestions.append("立即阻止输出并审查")
        
        if any("API" in e or "密钥" in e for e in evidence):
            suggestions.append("撤销相关API密钥")
        
        if any("IP" in e or "数据库" in e for e in evidence):
            suggestions.append("检查系统安全配置")
        
        if any("邮箱" in e or "电话" in e for e in evidence):
            suggestions.append("强化PII保护机制")
        
        suggestions.append("实施输出内容审查")
        return suggestions
    
    def _record_detection(self, agent_id: str, detected: bool, confidence: float):
        """记录检测历史"""
        record = {
            "timestamp": datetime.now(timezone.utc),
            "agent_id": agent_id,
            "detected": detected,
            "confidence": confidence
        }
        self.leakage_history.append(record)


class ModelPoisoningDetector:
    """模型中毒检测器"""
    
    def __init__(self):
        self.baseline_patterns: Dict[str, Dict[str, float]] = {}
        self.anomaly_threshold = 0.7
        self.poisoning_history: deque = deque(maxlen=1000)
        
    async def detect(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any]
    ) -> DetectionResult:
        """检测模型中毒"""
        try:
            evidence = []
            confidence = 0.0
            details = {}
            
            # 检查输出异常模式
            anomaly_score, anomaly_evidence = self._check_output_anomalies(agent_id, output)
            confidence += anomaly_score
            evidence.extend(anomaly_evidence)
            details["output_anomalies"] = anomaly_evidence
            
            # 检查质量退化
            quality_score, quality_evidence = self._check_quality_degradation(agent_id, output, context)
            confidence += quality_score
            evidence.extend(quality_evidence)
            details["quality_degradation"] = quality_evidence
            
            # 检查行为偏移
            behavior_score, behavior_evidence = self._check_behavior_drift(agent_id, output, context)
            confidence += behavior_score
            evidence.extend(behavior_evidence)
            details["behavior_drift"] = behavior_evidence
            
            # 检查后门触发器
            backdoor_score, backdoor_evidence = self._check_backdoor_triggers(output, context)
            confidence += backdoor_score
            evidence.extend(backdoor_evidence)
            details["backdoor_triggers"] = backdoor_evidence
            
            # 综合评估
            final_confidence = min(1.0, confidence / 4.0)
            attack_detected = final_confidence > 0.6
            threat_level = self._calculate_threat_level(final_confidence)
            
            # 记录检测历史
            self._record_detection(agent_id, attack_detected, final_confidence)
            
            result = DetectionResult(
                attack_detected=attack_detected,
                attack_type=AttackType.MODEL_POISONING,
                confidence=final_confidence,
                threat_level=threat_level,
                details=details,
                evidence=evidence,
                mitigation_suggestions=self._generate_mitigation_suggestions(evidence)
            )
            
            logger.info(
                "模型中毒检测完成",
                agent_id=agent_id,
                detected=attack_detected,
                confidence=final_confidence
            )
            
            return result
            
        except Exception as e:
            logger.error("模型中毒检测失败", agent_id=agent_id, error=str(e))
            return DetectionResult(
                attack_detected=True,
                attack_type=AttackType.MODEL_POISONING,
                confidence=1.0,
                threat_level=ThreatLevel.HIGH,
                details={"error": str(e)},
                evidence=["detection_error"],
                mitigation_suggestions=["model_retraining_required"]
            )
    
    def _check_output_anomalies(self, agent_id: str, output: str) -> Tuple[float, List[str]]:
        """检查输出异常模式"""
        evidence = []
        score = 0.0
        
        # 分析输出特征
        current_features = self._extract_output_features(output)
        
        # 获取基线模式
        if agent_id not in self.baseline_patterns:
            # 初始化基线
            self.baseline_patterns[agent_id] = current_features
            return 0.0, []
        
        baseline = self.baseline_patterns[agent_id]
        
        # 计算特征偏差
        deviation_score = 0.0
        for feature, value in current_features.items():
            baseline_value = baseline.get(feature, value)
            if baseline_value != 0:
                deviation = abs(value - baseline_value) / abs(baseline_value)
                if deviation > 0.5:  # 50%以上偏差
                    evidence.append(f"特征异常: {feature} 偏差{deviation:.2f}")
                    deviation_score += deviation
        
        # 更新基线（指数移动平均）
        alpha = 0.1
        for feature, value in current_features.items():
            baseline[feature] = alpha * value + (1 - alpha) * baseline.get(feature, value)
        
        score = min(1.0, deviation_score / len(current_features))
        
        return score, evidence
    
    def _check_quality_degradation(self, agent_id: str, output: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查质量退化"""
        evidence = []
        score = 0.0
        
        # 检查输出长度异常
        output_length = len(output.split())
        expected_length = context.get("expected_length", 50)
        
        if output_length < expected_length * 0.3:
            evidence.append(f"输出过短: {output_length} < {expected_length * 0.3}")
            score += 0.3
        elif output_length > expected_length * 3:
            evidence.append(f"输出过长: {output_length} > {expected_length * 3}")
            score += 0.2
        
        # 检查重复内容
        words = output.split()
        unique_words = set(words)
        if len(words) > 10 and len(unique_words) / len(words) < 0.5:
            evidence.append(f"重复词汇过多: {len(unique_words)}/{len(words)}")
            score += 0.4
        
        # 检查语法错误(简单检查)
        sentences = output.split('.')
        incomplete_sentences = sum(1 for s in sentences if len(s.strip().split()) < 3)
        if incomplete_sentences > len(sentences) * 0.3:
            evidence.append(f"语法问题: {incomplete_sentences}个不完整句子")
            score += 0.3
        
        # 检查任务完成度
        task_completion = context.get("task_completion_score", 1.0)
        if task_completion < 0.5:
            evidence.append(f"任务完成度低: {task_completion}")
            score += 0.5
        
        return min(1.0, score), evidence
    
    def _check_behavior_drift(self, agent_id: str, output: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查行为偏移"""
        evidence = []
        score = 0.0
        
        # 检查风格偏移
        current_style = self._analyze_style_features(output)
        baseline_style = context.get("baseline_style", current_style)
        
        style_drift = 0.0
        for feature, value in current_style.items():
            baseline_value = baseline_style.get(feature, value)
            if baseline_value != 0:
                drift = abs(value - baseline_value) / abs(baseline_value)
                style_drift += drift
        
        style_drift /= len(current_style)
        
        if style_drift > 0.3:
            evidence.append(f"风格偏移: {style_drift:.2f}")
            score += min(0.5, style_drift)
        
        # 检查主题偏移
        expected_topics = context.get("expected_topics", [])
        if expected_topics:
            topic_relevance = self._calculate_topic_relevance(output, expected_topics)
            if topic_relevance < 0.5:
                evidence.append(f"主题偏移: 相关度{topic_relevance:.2f}")
                score += 0.4
        
        # 检查情感偏移
        current_sentiment = self._analyze_sentiment(output)
        expected_sentiment = context.get("expected_sentiment", "neutral")
        
        if current_sentiment != expected_sentiment:
            evidence.append(f"情感偏移: {current_sentiment} != {expected_sentiment}")
            score += 0.3
        
        return min(1.0, score), evidence
    
    def _check_backdoor_triggers(self, output: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查后门触发器"""
        evidence = []
        score = 0.0
        
        # 检查可疑触发词
        suspicious_triggers = [
            "activate", "trigger", "backdoor", "payload",
            "execute", "deploy", "launch", "initiate"
        ]
        
        trigger_count = sum(1 for trigger in suspicious_triggers 
                           if trigger.lower() in output.lower())
        
        if trigger_count > 0:
            evidence.append(f"可疑触发词: {trigger_count}个")
            score += min(0.5, trigger_count * 0.2)
        
        # 检查编码后门
        encoded_patterns = [
            r"[A-Za-z0-9+/]{20,}={0,2}",  # Base64
            r"\\x[0-9a-fA-F]{2}",  # 十六进制
            r"%[0-9a-fA-F]{2}"     # URL编码
        ]
        
        for pattern in encoded_patterns:
            matches = re.findall(pattern, output)
            if matches:
                evidence.append(f"编码内容: {len(matches)}个")
                score += 0.3
        
        # 检查异常字符序列
        if len(set(output)) / len(output) < 0.1 and len(output) > 50:
            evidence.append("字符多样性异常低")
            score += 0.4
        
        return min(1.0, score), evidence
    
    def _extract_output_features(self, output: str) -> Dict[str, float]:
        """提取输出特征"""
        words = output.split()
        sentences = output.split('.')
        
        return {
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
            "punctuation_ratio": len(re.findall(r'[,.!?;:]', output)) / max(1, len(output)),
            "capital_ratio": len(re.findall(r'[A-Z]', output)) / max(1, len(output)),
            "digit_ratio": len(re.findall(r'\d', output)) / max(1, len(output)),
            "unique_word_ratio": len(set(words)) / max(1, len(words)),
            "total_length": len(output)
        }
    
    def _analyze_style_features(self, output: str) -> Dict[str, float]:
        """分析风格特征"""
        return {
            "formality": self._calculate_formality(output),
            "complexity": self._calculate_complexity(output),
            "positivity": self._calculate_positivity(output)
        }
    
    def _calculate_formality(self, text: str) -> float:
        """计算正式程度"""
        formal_indicators = ["furthermore", "moreover", "consequently", "therefore"]
        informal_indicators = ["gonna", "wanna", "yeah", "ok"]
        
        formal_count = sum(1 for indicator in formal_indicators 
                          if indicator.lower() in text.lower())
        informal_count = sum(1 for indicator in informal_indicators 
                            if indicator.lower() in text.lower())
        
        if formal_count + informal_count == 0:
            return 0.5  # 中性
        
        return formal_count / (formal_count + informal_count)
    
    def _calculate_complexity(self, text: str) -> float:
        """计算复杂度"""
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        return min(1.0, avg_word_length / 10.0)
    
    def _calculate_positivity(self, text: str) -> float:
        """计算积极性"""
        positive_words = ["good", "great", "excellent", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst"]
        
        positive_count = sum(1 for word in positive_words 
                            if word.lower() in text.lower())
        negative_count = sum(1 for word in negative_words 
                            if word.lower() in text.lower())
        
        if positive_count + negative_count == 0:
            return 0.5  # 中性
        
        return positive_count / (positive_count + negative_count)
    
    def _calculate_topic_relevance(self, output: str, expected_topics: List[str]) -> float:
        """计算主题相关度"""
        if not expected_topics:
            return 1.0
        
        output_lower = output.lower()
        relevant_count = sum(1 for topic in expected_topics 
                            if topic.lower() in output_lower)
        
        return relevant_count / len(expected_topics)
    
    def _analyze_sentiment(self, text: str) -> str:
        """分析情感倾向"""
        positive_words = ["good", "great", "excellent", "wonderful", "amazing", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate", "dislike"]
        
        positive_count = sum(1 for word in positive_words 
                            if word.lower() in text.lower())
        negative_count = sum(1 for word in negative_words 
                            if word.lower() in text.lower())
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_threat_level(self, confidence: float) -> ThreatLevel:
        """计算威胁级别"""
        if confidence >= 0.8:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.6:
            return ThreatLevel.HIGH
        elif confidence >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _generate_mitigation_suggestions(self, evidence: List[str]) -> List[str]:
        """生成缓解建议"""
        suggestions = []
        
        if any("异常" in e for e in evidence):
            suggestions.append("增强异常检测机制")
        
        if any("质量" in e or "语法" in e for e in evidence):
            suggestions.append("质量控制和重新训练")
        
        if any("偏移" in e for e in evidence):
            suggestions.append("校准模型行为基线")
        
        if any("触发" in e or "编码" in e for e in evidence):
            suggestions.append("深度安全扫描和模型检查")
        
        suggestions.append("考虑模型回滚或重新训练")
        return suggestions
    
    def _record_detection(self, agent_id: str, detected: bool, confidence: float):
        """记录检测历史"""
        record = {
            "timestamp": datetime.now(timezone.utc),
            "agent_id": agent_id,
            "detected": detected,
            "confidence": confidence
        }
        self.poisoning_history.append(record)


class AttackDetectionManager:
    """攻击检测管理器"""
    
    def __init__(self):
        self.detectors = {
            AttackType.PROMPT_INJECTION: PromptInjectionDetector(),
            AttackType.DATA_LEAKAGE: DataLeakageDetector(),
            AttackType.MODEL_POISONING: ModelPoisoningDetector()
        }
        self.detection_cache: Dict[str, DetectionResult] = {}
        self.cache_ttl = timedelta(minutes=5)
        
    async def detect_attacks(
        self, 
        agent_id: str, 
        output: str, 
        context: Dict[str, Any],
        attack_types: Optional[List[AttackType]] = None
    ) -> Dict[AttackType, DetectionResult]:
        """检测多种攻击类型"""
        if attack_types is None:
            attack_types = list(self.detectors.keys())
        
        # 检查缓存
        cache_key = self._generate_cache_key(agent_id, output)
        if cache_key in self.detection_cache:
            cached_result = self.detection_cache[cache_key]
            if datetime.now(timezone.utc) - cached_result.details.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)) < self.cache_ttl:
                return {cached_result.attack_type: cached_result}
        
        results = {}
        
        # 并行执行检测
        detection_tasks = []
        for attack_type in attack_types:
            if attack_type in self.detectors:
                detector = self.detectors[attack_type]
                task = detector.detect(agent_id, output, context)
                detection_tasks.append((attack_type, task))
        
        # 等待所有检测完成
        for attack_type, task in detection_tasks:
            try:
                result = await task
                results[attack_type] = result
                
                # 缓存结果
                result.details["timestamp"] = datetime.now(timezone.utc)
                self.detection_cache[cache_key] = result
                
            except Exception as e:
                logger.error("攻击检测失败", attack_type=attack_type, error=str(e))
                results[attack_type] = DetectionResult(
                    attack_detected=True,
                    attack_type=attack_type,
                    confidence=1.0,
                    threat_level=ThreatLevel.HIGH,
                    details={"error": str(e)},
                    evidence=["detection_error"],
                    mitigation_suggestions=["manual_review_required"]
                )
        
        # 清理过期缓存
        self._cleanup_cache()
        
        return results
    
    def get_detection_summary(self, results: Dict[AttackType, DetectionResult]) -> Dict[str, Any]:
        """获取检测摘要"""
        total_attacks = sum(1 for result in results.values() if result.attack_detected)
        max_confidence = max((result.confidence for result in results.values()), default=0.0)
        highest_threat = max((result.threat_level for result in results.values()), 
                           default=ThreatLevel.LOW, key=lambda x: ["low", "medium", "high", "critical"].index(x.value))
        
        detected_attacks = [
            attack_type.value for attack_type, result in results.items() 
            if result.attack_detected
        ]
        
        all_evidence = []
        all_suggestions = []
        
        for result in results.values():
            all_evidence.extend(result.evidence)
            all_suggestions.extend(result.mitigation_suggestions)
        
        return {
            "attacks_detected": total_attacks,
            "attack_types": detected_attacks,
            "max_confidence": max_confidence,
            "highest_threat_level": highest_threat.value,
            "total_evidence": len(all_evidence),
            "unique_suggestions": len(set(all_suggestions)),
            "recommendation": self._generate_overall_recommendation(results)
        }
    
    def _generate_cache_key(self, agent_id: str, output: str) -> str:
        """生成缓存键"""
        content = f"{agent_id}:{output}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        
        for key, result in self.detection_cache.items():
            timestamp = result.details.get("timestamp", datetime.min.replace(tzinfo=timezone.utc))
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.detection_cache[key]
    
    def _generate_overall_recommendation(self, results: Dict[AttackType, DetectionResult]) -> str:
        """生成总体建议"""
        critical_attacks = [r for r in results.values() 
                           if r.attack_detected and r.threat_level == ThreatLevel.CRITICAL]
        high_attacks = [r for r in results.values() 
                       if r.attack_detected and r.threat_level == ThreatLevel.HIGH]
        
        if critical_attacks:
            return "BLOCK_IMMEDIATELY"
        elif high_attacks:
            return "REQUIRE_REVIEW"
        elif any(r.attack_detected for r in results.values()):
            return "INCREASE_MONITORING"
        else:
            return "PROCEED_NORMALLY"