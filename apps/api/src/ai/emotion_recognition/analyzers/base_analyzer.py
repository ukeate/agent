"""基础情感分析器抽象类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime
from ..models.emotion_models import EmotionResult, EmotionDimension, Modality

logger = get_logger(__name__)

class BaseEmotionAnalyzer(ABC):
    """情感分析器基类"""
    
    def __init__(self, modality: Modality, config: Optional[Dict[str, Any]] = None):
        """
        初始化分析器
        
        Args:
            modality: 模态类型
            config: 配置参数
        """
        self.modality = modality
        self.config = config or {}
        self.model = None
        self.processor = None
        self.is_initialized = False
        
    async def initialize(self):
        """异步初始化模型和处理器"""
        if not self.is_initialized:
            logger.info(f"正在初始化 {self.modality} 情感分析器...")
            await self._load_model()
            self.is_initialized = True
            logger.info(f"{self.modality} 情感分析器初始化完成")
            
    @abstractmethod
    async def _load_model(self):
        """加载模型和处理器"""
        raise NotImplementedError
        
    @abstractmethod
    async def analyze(self, input_data: Any) -> EmotionResult:
        """
        分析输入数据的情感
        
        Args:
            input_data: 输入数据(文本/音频/图像)
            
        Returns:
            EmotionResult: 情感分析结果
        """
        raise NotImplementedError
        
    @abstractmethod
    async def preprocess(self, input_data: Any) -> Any:
        """
        预处理输入数据
        
        Args:
            input_data: 原始输入数据
            
        Returns:
            预处理后的数据
        """
        raise NotImplementedError
        
    @abstractmethod
    async def postprocess(self, raw_output: Any) -> EmotionResult:
        """
        后处理模型输出
        
        Args:
            raw_output: 模型原始输出
            
        Returns:
            EmotionResult: 格式化的情感结果
        """
        raise NotImplementedError
        
    def calculate_intensity(self, confidence: float, features: Dict[str, Any]) -> float:
        """
        计算情感强度
        
        Args:
            confidence: 置信度
            features: 特征字典
            
        Returns:
            float: 情感强度 [0, 1]
        """
        # 基础强度计算，子类可重写
        base_intensity = confidence
        
        # 根据特征调整强度
        if "arousal" in features:
            base_intensity = (base_intensity + features["arousal"]) / 2
            
        return min(max(base_intensity, 0.0), 1.0)
        
    def map_to_dimension(self, emotion: str, intensity: float) -> EmotionDimension:
        """
        将情感映射到VAD维度
        
        Args:
            emotion: 情感标签
            intensity: 情感强度
            
        Returns:
            EmotionDimension: 情感维度
        """
        from ..models.emotion_models import EMOTION_DIMENSIONS, EmotionCategory
        
        # 尝试获取预定义的维度
        try:
            emotion_enum = EmotionCategory(emotion.lower())
            base_dimension = EMOTION_DIMENSIONS.get(emotion_enum)
            
            if base_dimension:
                # 根据强度调整维度值
                return EmotionDimension(
                    valence=base_dimension.valence * intensity,
                    arousal=base_dimension.arousal * intensity,
                    dominance=base_dimension.dominance * intensity
                )
        except ValueError:
            logger.warning("捕获到ValueError，已继续执行", exc_info=True)
            
        # 默认维度(中性)
        return EmotionDimension(
            valence=0.0,
            arousal=0.3 * intensity,
            dominance=0.5
        )
        
    async def batch_analyze(self, input_list: List[Any]) -> List[EmotionResult]:
        """
        批量分析
        
        Args:
            input_list: 输入数据列表
            
        Returns:
            List[EmotionResult]: 情感结果列表
        """
        if not self.is_initialized:
            await self.initialize()
            
        # 并发处理批量数据
        tasks = [self.analyze(input_data) for input_data in input_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量处理第{i}个数据时出错: {result}")
            else:
                valid_results.append(result)
                
        return valid_results
        
    def validate_input(self, input_data: Any) -> bool:
        """
        验证输入数据有效性
        
        Args:
            input_data: 输入数据
            
        Returns:
            bool: 是否有效
        """
        # 基础验证，子类应重写
        return input_data is not None
        
    async def warmup(self):
        """预热模型"""
        if not self.is_initialized:
            await self.initialize()
            
        logger.info(f"正在预热 {self.modality} 分析器...")
        # 子类可以重写以执行具体的预热逻辑
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "modality": self.modality,
            "is_initialized": self.is_initialized,
            "config": self.config
        }
        
    async def cleanup(self):
        """清理资源"""
        logger.info(f"清理 {self.modality} 分析器资源...")
        self.model = None
        self.processor = None
        self.is_initialized = False
from src.core.logging import get_logger
