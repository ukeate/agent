"""文本情感分析器"""

from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import re
import torch
from .base_analyzer import BaseEmotionAnalyzer
from ..models.emotion_models import (
    EmotionResult, EmotionDimension, Modality,
    EmotionCategory, EMOTION_DIMENSIONS
)
from src.core.utils.timezone_utils import utc_now

logger = get_logger(__name__)

class TextEmotionAnalyzer(BaseEmotionAnalyzer):
    """基于Transformer的文本情感分析器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文本情感分析器
        
        Args:
            config: 配置参数
        """
        default_config = {
            "model_name": "j-hartmann/emotion-english-distilroberta-base",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_length": 512,
            "batch_size": 32,
            "temperature": 1.0
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(Modality.TEXT, default_config)
        
    async def _load_model(self):
        """加载预训练模型和分词器"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model_name"]
            )
            
            # 创建pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config["device"] == "cuda" else -1,
                top_k=None  # 返回所有标签的分数
            )
            
            logger.info(f"文本情感模型加载成功: {self.config['model_name']}")
            
        except Exception as e:
            logger.error(f"加载文本情感模型失败: {e}")
            # 使用备用的简单模型
            self._load_fallback_model()
            
    def _load_fallback_model(self):
        """加载备用模型"""
        from transformers import pipeline
        
        try:
            # 使用默认的sentiment-analysis模型
            self.pipeline = pipeline(
                "sentiment-analysis",
                device=-1,  # CPU
                top_k=None
            )
            logger.info("使用备用情感分析模型")
        except Exception as e:
            logger.error(f"备用模型加载失败: {e}")
            self.pipeline = None
            
    async def preprocess(self, text: str) -> Dict[str, Any]:
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的数据
        """
        # 清理文本
        text = text.strip()
        
        # 提取文本特征
        features = {
            "text": text,
            "length": len(text),
            "word_count": len(text.split()),
            "has_punctuation": bool(re.search(r'[!?]+', text)),
            "has_caps": bool(re.search(r'[A-Z]{2,}', text)),
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "emoji_count": self._count_emojis(text)
        }
        
        # 计算文本复杂度权重
        features["complexity_weight"] = self._calculate_text_complexity(features)
        
        return features
        
    def _count_emojis(self, text: str) -> int:
        """统计emoji数量"""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # 表情符号
            u"\U0001F300-\U0001F5FF"  # 符号和图标
            u"\U0001F680-\U0001F6FF"  # 交通和地图符号
            u"\U0001F1E0-\U0001F1FF"  # 国旗
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
        
    def _calculate_text_complexity(self, features: Dict[str, Any]) -> float:
        """计算文本复杂度权重"""
        weight = 1.0
        
        # 标点符号增强
        if features["has_punctuation"]:
            weight += 0.1 * min(features["exclamation_count"], 3)
            weight += 0.05 * min(features["question_count"], 2)
            
        # 大写字母增强
        if features["has_caps"]:
            weight += 0.1
            
        # Emoji增强
        if features["emoji_count"] > 0:
            weight += 0.15 * min(features["emoji_count"], 3)
            
        return min(weight, 1.5)
        
    async def analyze(self, text: str) -> EmotionResult:
        """
        分析文本情感
        
        Args:
            text: 输入文本
            
        Returns:
            EmotionResult: 情感分析结果
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not self.validate_input(text):
            raise ValueError("输入文本无效")
            
        # 预处理文本
        features = await self.preprocess(text)
        
        # 使用模型进行预测
        if self.pipeline:
            try:
                predictions = self.pipeline(text)
                
                # 后处理结果
                result = await self.postprocess({
                    "predictions": predictions,
                    "features": features
                })
                
                return result
                
            except Exception as e:
                logger.error(f"文本情感分析出错: {e}")
                # 返回中性结果
                return self._create_neutral_result(features)
        else:
            return self._create_neutral_result(features)
            
    async def postprocess(self, raw_output: Dict[str, Any]) -> EmotionResult:
        """
        后处理模型输出
        
        Args:
            raw_output: 模型原始输出
            
        Returns:
            EmotionResult: 格式化的情感结果
        """
        predictions = raw_output["predictions"]
        features = raw_output["features"]
        
        # 整理预测结果
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                # top_k模式，包含所有标签
                emotions = predictions[0]
            else:
                emotions = predictions
        else:
            emotions = []
            
        if not emotions:
            return self._create_neutral_result(features)
            
        # 排序并标准化情感标签
        emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)
        
        # 主要情感
        primary = emotions[0]
        emotion_label = self._standardize_emotion_label(primary["label"])
        confidence = primary["score"]
        
        # 次要情感
        sub_emotions = []
        for emo in emotions[1:5]:  # 最多取前5个
            if emo["score"] > 0.1:  # 置信度阈值
                sub_emotions.append((
                    self._standardize_emotion_label(emo["label"]),
                    emo["score"]
                ))
                
        # 计算强度
        intensity = self.calculate_intensity(confidence, features)
        
        # 映射到VAD维度
        dimension = self.map_to_dimension(emotion_label, intensity)
        
        # 创建结果
        return EmotionResult(
            emotion=emotion_label,
            confidence=confidence,
            intensity=intensity * features["complexity_weight"],
            timestamp=utc_now(),
            modality=str(self.modality.value),
            details={
                "text_length": features["length"],
                "word_count": features["word_count"],
                "has_punctuation": features["has_punctuation"],
                "emoji_count": features["emoji_count"],
                "raw_predictions": emotions[:3]  # 保存前3个预测
            },
            sub_emotions=sub_emotions,
            dimension=dimension
        )
        
    def _standardize_emotion_label(self, label: str) -> str:
        """标准化情感标签"""
        label = label.lower().strip()
        
        # 映射到标准情感类别
        emotion_mapping = {
            "happy": EmotionCategory.HAPPINESS,
            "joy": EmotionCategory.JOY,
            "excited": EmotionCategory.EXCITEMENT,
            "sad": EmotionCategory.SADNESS,
            "angry": EmotionCategory.ANGER,
            "anger": EmotionCategory.ANGER,
            "fear": EmotionCategory.FEAR,
            "fearful": EmotionCategory.FEAR,
            "disgust": EmotionCategory.DISGUST,
            "disgusted": EmotionCategory.DISGUST,
            "surprise": EmotionCategory.SURPRISE,
            "surprised": EmotionCategory.SURPRISE,
            "neutral": EmotionCategory.NEUTRAL,
            "positive": EmotionCategory.HAPPINESS,
            "negative": EmotionCategory.SADNESS,
            "love": EmotionCategory.JOY,
            "optimism": EmotionCategory.SATISFACTION,
            "pessimism": EmotionCategory.DISAPPOINTMENT,
            "trust": EmotionCategory.TRUST,
            "anticipation": EmotionCategory.ANTICIPATION
        }
        
        for key, value in emotion_mapping.items():
            if key in label:
                return value.value
                
        # 如果没有匹配，返回原始标签
        return label
        
    def _create_neutral_result(self, features: Dict[str, Any]) -> EmotionResult:
        """创建中性情感结果"""
        return EmotionResult(
            emotion=EmotionCategory.NEUTRAL.value,
            confidence=0.5,
            intensity=0.3,
            timestamp=utc_now(),
            modality=str(self.modality.value),
            details=features,
            dimension=EMOTION_DIMENSIONS[EmotionCategory.NEUTRAL]
        )
        
    def validate_input(self, text: str) -> bool:
        """验证输入文本"""
        if not text or not isinstance(text, str):
            return False
            
        # 清理后检查
        text = text.strip()
        if len(text) == 0:
            return False
            
        # 检查文本长度
        if len(text) > 10000:  # 最大10000字符
            logger.warning(f"文本过长: {len(text)} 字符")
            return False
            
        return True
        
    async def analyze_with_context(
        self,
        text: str,
        context: Optional[List[str]] = None
    ) -> EmotionResult:
        """
        带上下文的情感分析
        
        Args:
            text: 当前文本
            context: 上下文文本列表
            
        Returns:
            EmotionResult: 情感分析结果
        """
        if context and len(context) > 0:
            # 将上下文与当前文本合并
            full_text = " ".join(context[-3:]) + " " + text  # 最多使用最近3条上下文
            
            # 分析合并后的文本
            result = await self.analyze(full_text)
            
            # 调整结果以反映上下文影响
            result.details["has_context"] = True
            result.details["context_length"] = len(context)
            
            return result
        else:
            return await self.analyze(text)
            
    async def get_sentiment_score(self, text: str) -> float:
        """
        获取情感倾向分数
        
        Args:
            text: 输入文本
            
        Returns:
            float: 情感分数 [-1, 1]
        """
        result = await self.analyze(text)
        
        # 使用效价(valence)作为情感分数
        if result.dimension:
            return result.dimension.valence
        else:
            # 根据情感类别返回默认分数
            if result.emotion in ["happiness", "joy", "excitement"]:
                return 0.8
            elif result.emotion in ["sadness", "anger", "fear"]:
                return -0.7
            else:
                return 0.0
from src.core.logging import get_logger
