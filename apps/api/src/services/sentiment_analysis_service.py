"""
情感分析服务

提供文本评论的情感分析功能，支持基础情感分类和情感强度计算
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.logging import get_logger
logger = get_logger(__name__)

class SentimentType(str, Enum):
    """情感类型枚举"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    """情感分析结果"""
    sentiment: SentimentType
    confidence: float  # 置信度 0-1
    intensity: float   # 情感强度 0-1
    keywords: List[str] = None
    scores: Dict[str, float] = None  # 各情感的原始分数
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.scores is None:
            self.scores = {}

class SentimentAnalysisService:
    """情感分析服务"""
    
    def __init__(self):
        # 简单的情感词典 - 实际应用中应该使用更完整的词典
        self.positive_words = {
            '好', '棒', '赞', '优秀', '完美', '满意', '喜欢', '爱', '推荐', '值得',
            '精彩', '出色', '卓越', '惊艳', '不错', '很棒', '给力', '厉害', '牛',
            '太好了', '很好', '非常好', '超级好', '特别好', '相当好', '真的好',
            'good', 'great', 'excellent', 'awesome', 'amazing', 'wonderful',
            'fantastic', 'perfect', 'outstanding', 'brilliant', 'superb',
            'love', 'like', 'enjoy', 'recommend', 'impressive'
        }
        
        self.negative_words = {
            '差', '糟', '烂', '垃圾', '失望', '讨厌', '恶心', '无聊', '浪费', '后悔',
            '不好', '很差', '太差', '非常差', '超级差', '相当差', '真的差',
            '不满意', '不推荐', '不值得', '不行', '有问题', '坑爹', '坑人',
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'waste',
            'hate', 'dislike', 'boring', 'useless', 'worthless', 'regret',
            'poor', 'worst', 'disgusting', 'annoying', 'frustrated'
        }
        
        self.intensifiers = {
            '非常': 1.5, '很': 1.3, '特别': 1.4, '相当': 1.2, '超级': 1.6,
            '太': 1.4, '极其': 1.7, '十分': 1.3, '相当': 1.2, '比较': 0.8,
            'very': 1.4, 'extremely': 1.7, 'really': 1.3, 'quite': 1.2,
            'so': 1.3, 'too': 1.4, 'super': 1.5, 'absolutely': 1.6
        }
        
        self.negators = {
            '不', '没', '无', '非', '未', '别', '勿', '莫',
            'not', 'no', 'never', 'neither', 'none', 'nobody', 'nothing'
        }
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        分析文本情感
        
        Args:
            text: 待分析的文本
            
        Returns:
            SentimentResult: 情感分析结果
        """
        try:
            # 文本预处理
            cleaned_text = self._preprocess_text(text)
            
            # 情感词分析
            sentiment_score, found_keywords = self._calculate_sentiment_score(cleaned_text)
            
            # 确定情感类型和置信度
            sentiment_type, confidence = self._determine_sentiment(sentiment_score)
            
            # 计算情感强度
            intensity = self._calculate_intensity(sentiment_score)
            
            result = SentimentResult(
                sentiment=sentiment_type,
                confidence=confidence,
                intensity=intensity,
                keywords=found_keywords,
                scores={'sentiment_score': sentiment_score}
            )
            
            logger.debug(f"情感分析结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            # 返回中性结果作为降级处理
            return SentimentResult(
                sentiment=SentimentType.NEUTRAL,
                confidence=0.5,
                intensity=0.5,
                keywords=[],
                scores={'sentiment_score': 0.0}
            )
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符但保留基本标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：、""''（）【】《》]', ' ', text)
        
        return text
    
    def _calculate_sentiment_score(self, text: str) -> Tuple[float, List[str]]:
        """计算情感分数"""
        words = text.split()
        sentiment_score = 0.0
        found_keywords = []
        
        for i, word in enumerate(words):
            # 检查否定词
            negated = self._check_negation(words, i)
            
            # 检查情感词
            if word in self.positive_words:
                score = 1.0
                if negated:
                    score = -score
                
                # 检查强化词
                intensifier = self._check_intensifier(words, i)
                score *= intensifier
                
                sentiment_score += score
                found_keywords.append(word)
                
            elif word in self.negative_words:
                score = -1.0
                if negated:
                    score = -score
                
                # 检查强化词
                intensifier = self._check_intensifier(words, i)
                score *= intensifier
                
                sentiment_score += score
                found_keywords.append(word)
        
        return sentiment_score, found_keywords
    
    def _check_negation(self, words: List[str], index: int) -> bool:
        """检查否定词"""
        # 检查前面的2个词
        for i in range(max(0, index - 2), index):
            if words[i] in self.negators:
                return True
        return False
    
    def _check_intensifier(self, words: List[str], index: int) -> float:
        """检查强化词"""
        intensifier = 1.0
        
        # 检查前面的词
        for i in range(max(0, index - 2), index):
            if words[i] in self.intensifiers:
                intensifier *= self.intensifiers[words[i]]
        
        return intensifier
    
    def _determine_sentiment(self, score: float) -> Tuple[SentimentType, float]:
        """确定情感类型和置信度"""
        abs_score = abs(score)
        
        if abs_score < 0.5:
            return SentimentType.NEUTRAL, 0.6 + abs_score * 0.8
        elif score > 0:
            confidence = min(0.7 + abs_score * 0.15, 0.95)
            return SentimentType.POSITIVE, confidence
        else:
            confidence = min(0.7 + abs_score * 0.15, 0.95)
            return SentimentType.NEGATIVE, confidence
    
    def _calculate_intensity(self, score: float) -> float:
        """计算情感强度"""
        # 将分数映射到0-1区间
        abs_score = abs(score)
        intensity = min(abs_score / 3.0, 1.0)  # 假设最大分数为3
        return intensity
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """批量分析情感"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict:
        """获取情感分析摘要"""
        if not results:
            return {
                'total_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'average_confidence': 0.0,
                'average_intensity': 0.0,
                'sentiment_distribution': {}
            }
        
        positive_count = sum(1 for r in results if r.sentiment == SentimentType.POSITIVE)
        negative_count = sum(1 for r in results if r.sentiment == SentimentType.NEGATIVE)
        neutral_count = sum(1 for r in results if r.sentiment == SentimentType.NEUTRAL)
        
        total_count = len(results)
        average_confidence = sum(r.confidence for r in results) / total_count
        average_intensity = sum(r.intensity for r in results) / total_count
        
        return {
            'total_count': total_count,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_ratio': positive_count / total_count,
            'negative_ratio': negative_count / total_count,
            'neutral_ratio': neutral_count / total_count,
            'average_confidence': average_confidence,
            'average_intensity': average_intensity,
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        }
    
    def extract_keywords(self, text: str, sentiment_type: SentimentType = None) -> List[str]:
        """提取关键词"""
        cleaned_text = self._preprocess_text(text)
        words = cleaned_text.split()
        
        keywords = []
        for word in words:
            if sentiment_type == SentimentType.POSITIVE and word in self.positive_words:
                keywords.append(word)
            elif sentiment_type == SentimentType.NEGATIVE and word in self.negative_words:
                keywords.append(word)
            elif sentiment_type is None and (word in self.positive_words or word in self.negative_words):
                keywords.append(word)
        
        return list(set(keywords))  # 去重
    
    def is_spam_content(self, text: str) -> bool:
        """检测垃圾内容"""
        if not text or len(text.strip()) < 3:
            return True
        
        # 检测重复字符
        if re.search(r'(.)\1{5,}', text):
            return True
        
        # 检测仅符号
        if re.match(r'^[!@#$%^&*()]+$', text.strip()):
            return True
        
        # 检测仅数字
        if re.match(r'^[0-9\s]+$', text.strip()):
            return True
        
        # 检测过短内容
        if len(text.strip()) < 3:
            return True
        
        return False
    
    def enhance_with_context(self, text: str, context: Dict) -> SentimentResult:
        """基于上下文增强情感分析"""
        base_result = self.analyze_sentiment(text)
        
        # 根据上下文调整结果
        if context:
            # 如果有评分信息，可以调整情感分析结果
            rating = context.get('rating', 0)
            if rating:
                if rating >= 4 and base_result.sentiment == SentimentType.NEGATIVE:
                    # 评分很高但情感为负，可能分析有误
                    base_result.confidence *= 0.8
                elif rating <= 2 and base_result.sentiment == SentimentType.POSITIVE:
                    # 评分很低但情感为正，可能分析有误
                    base_result.confidence *= 0.8
        
        return base_result

# 全局服务实例
sentiment_service = SentimentAnalysisService()
