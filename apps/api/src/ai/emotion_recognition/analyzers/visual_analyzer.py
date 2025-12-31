"""视觉情感分析器"""

from typing import Any, Dict, Optional, List, Tuple, Union
from datetime import datetime
import numpy as np
import torch
import io
from PIL import Image
from .base_analyzer import BaseEmotionAnalyzer
from ..models.emotion_models import (
    EmotionResult, EmotionDimension, Modality,
    EmotionCategory, EMOTION_DIMENSIONS
)
from src.core.utils.timezone_utils import utc_now

logger = get_logger(__name__)

class VisualEmotionAnalyzer(BaseEmotionAnalyzer):
    """基于深度学习的视觉情感分析器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化视觉情感分析器
        
        Args:
            config: 配置参数
        """
        default_config = {
            "model_name": "dima806/facial_emotions_image_detection",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "image_size": (224, 224),
            "face_detection": "opencv",  # opencv or mtcnn
            "min_face_size": 40,
            "batch_size": 16,
            "confidence_threshold": 0.5
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(Modality.VISUAL, default_config)
        self.face_cascade = None
        
    async def _load_model(self):
        """加载预训练模型和处理器"""
        try:
            from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
            import cv2
            
            # 加载面部检测器
            if self.config["face_detection"] == "opencv":
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("OpenCV面部检测器加载成功")
            
            # 加载表情识别模型
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.config["model_name"])
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.config["model_name"]
                )
                
                # 创建图像分类pipeline
                self.pipeline = pipeline(
                    "image-classification",
                    model=self.model,
                    image_processor=self.processor,
                    device=0 if self.config["device"] == "cuda" else -1
                )
            except:
                # 使用备用模型
                self.pipeline = pipeline(
                    "image-classification",
                    model="trpakov/vit-face-expression",
                    device=-1
                )
                
            logger.info(f"视觉情感模型加载成功")
            
        except Exception as e:
            logger.error(f"加载视觉情感模型失败: {e}")
            self._load_fallback_model()
            
    def _load_fallback_model(self):
        """加载备用模型"""
        try:
            # 使用简单的基于规则的方法
            self.pipeline = None
            logger.info("使用备用视觉情感分析方法")
        except Exception as e:
            logger.error(f"备用模型加载失败: {e}")
            
    async def preprocess(self, image_data: Union[np.ndarray, bytes, str, Image.Image]) -> Dict[str, Any]:
        """
        预处理图像数据
        
        Args:
            image_data: 图像数据(numpy数组、字节、文件路径或PIL Image)
            
        Returns:
            预处理后的数据
        """
        # 加载图像
        image, original_image = await self._load_image(image_data)
        
        # 检测面部
        faces = await self._detect_faces(original_image)
        
        # 提取面部特征
        features = await self._extract_visual_features(original_image, faces)
        
        # 裁剪面部区域
        face_images = []
        for face in faces:
            x, y, w, h = face
            face_img = original_image[y:y+h, x:x+w]
            face_images.append(face_img)
            
        return {
            "image": image,
            "original_image": original_image,
            "faces": faces,
            "face_images": face_images,
            "features": features,
            "num_faces": len(faces)
        }
        
    async def _load_image(self, image_data: Union[np.ndarray, bytes, str, Image.Image]) -> Tuple[Image.Image, np.ndarray]:
        """加载图像数据"""
        try:
            import cv2
            
            if isinstance(image_data, np.ndarray):
                # numpy数组
                original_image = image_data
                if len(original_image.shape) == 2:  # 灰度图
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                image = Image.fromarray(original_image)
                
            elif isinstance(image_data, bytes):
                # 字节数据
                image = Image.open(io.BytesIO(image_data))
                original_image = np.array(image)
                
            elif isinstance(image_data, str):
                # 文件路径
                image = Image.open(image_data)
                original_image = cv2.imread(image_data)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_data, Image.Image):
                # PIL Image
                image = image_data
                original_image = np.array(image)
                
            else:
                raise ValueError(f"不支持的图像数据类型: {type(image_data)}")
                
            return image, original_image
            
        except Exception as e:
            logger.error(f"加载图像失败: {e}")
            raise
            
    async def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测图像中的面部"""
        try:
            import cv2
            
            if self.face_cascade is None:
                # 加载面部检测器
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 检测面部
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.config["min_face_size"], self.config["min_face_size"])
            )
            
            # 转换为列表
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append((x, y, w, h))
                
            return face_list
            
        except Exception as e:
            logger.error(f"面部检测失败: {e}")
            # 返回整个图像作为一个"面部"
            h, w = image.shape[:2]
            return [(0, 0, w, h)]
            
    async def _extract_visual_features(self, image: np.ndarray, faces: List[Tuple]) -> Dict[str, Any]:
        """提取视觉特征"""
        try:
            import cv2
            
            features = {}
            
            # 图像基础特征
            features["image_shape"] = image.shape
            features["num_faces"] = len(faces)
            
            if len(faces) > 0:
                # 面部大小统计
                face_sizes = [(w * h) for _, _, w, h in faces]
                features["avg_face_size"] = np.mean(face_sizes)
                features["max_face_size"] = np.max(face_sizes)
                
                # 面部位置分析
                face_centers = [(x + w/2, y + h/2) for x, y, w, h in faces]
                features["face_centers"] = face_centers
                
                # 计算最大面部的特征
                largest_face_idx = np.argmax(face_sizes)
                x, y, w, h = faces[largest_face_idx]
                face_roi = image[y:y+h, x:x+w]
                
                # 颜色特征
                features["face_brightness"] = np.mean(face_roi)
                features["face_contrast"] = np.std(face_roi)
                
                # 边缘特征
                edges = cv2.Canny(cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY), 50, 150)
                features["edge_density"] = np.sum(edges > 0) / edges.size
                
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return {}
            
    async def analyze(self, image_data: Union[np.ndarray, bytes, str, Image.Image]) -> EmotionResult:
        """
        分析图像情感
        
        Args:
            image_data: 图像数据
            
        Returns:
            EmotionResult: 情感分析结果
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not self.validate_input(image_data):
            raise ValueError("输入图像无效")
            
        # 预处理图像
        preprocessed = await self.preprocess(image_data)
        
        if preprocessed["num_faces"] == 0:
            logger.warning("未检测到面部")
            return self._create_neutral_result(preprocessed["features"])
            
        if self.pipeline:
            try:
                # 分析每个面部
                if preprocessed["num_faces"] == 1:
                    # 单个面部
                    predictions = self.pipeline(preprocessed["image"])
                    result = await self.postprocess({
                        "predictions": predictions,
                        "features": preprocessed["features"],
                        "num_faces": 1
                    })
                else:
                    # 多个面部，分析每个面部并聚合
                    result = await self._analyze_multiple_faces(preprocessed)
                    
                return result
                
            except Exception as e:
                logger.error(f"视觉情感分析出错: {e}")
                return self._create_neutral_result(preprocessed["features"])
        else:
            # 使用基于规则的方法
            return await self._rule_based_analysis(preprocessed)
            
    async def _analyze_multiple_faces(self, preprocessed: Dict[str, Any]) -> EmotionResult:
        """分析多个面部"""
        all_predictions = []
        
        for face_img in preprocessed["face_images"]:
            try:
                face_pil = Image.fromarray(face_img)
                predictions = self.pipeline(face_pil)
                all_predictions.append(predictions)
            except Exception as e:
                logger.error(f"分析面部时出错: {e}")
                
        # 聚合所有面部的结果
        aggregated = await self._aggregate_face_results(all_predictions)
        
        return await self.postprocess({
            "predictions": aggregated,
            "features": preprocessed["features"],
            "num_faces": preprocessed["num_faces"]
        })
        
    async def _aggregate_face_results(self, face_results: List[Any]) -> List[Dict]:
        """聚合多个面部的结果"""
        if not face_results:
            return []
            
        # 统计各情感出现频率和平均分数
        emotion_scores = {}
        
        for face in face_results:
            for pred in face:
                label = pred["label"]
                score = pred["score"]
                
                if label not in emotion_scores:
                    emotion_scores[label] = []
                emotion_scores[label].append(score)
                
        # 计算平均分数
        aggregated = []
        for label, scores in emotion_scores.items():
            aggregated.append({
                "label": label,
                "score": np.mean(scores)
            })
            
        return sorted(aggregated, key=lambda x: x["score"], reverse=True)
        
    async def _rule_based_analysis(self, preprocessed: Dict[str, Any]) -> EmotionResult:
        """基于规则的情感分析(备用方法)"""
        features = preprocessed["features"]
        
        # 基于特征的简单规则
        emotion = EmotionCategory.NEUTRAL.value
        confidence = 0.5
        intensity = 0.3
        
        if "face_brightness" in features:
            brightness = features["face_brightness"]
            if brightness > 180:  # 较亮
                emotion = EmotionCategory.HAPPINESS.value
                confidence = 0.6
                intensity = 0.5
            elif brightness < 100:  # 较暗
                emotion = EmotionCategory.SADNESS.value
                confidence = 0.6
                intensity = 0.4
                
        if "edge_density" in features:
            edge_density = features["edge_density"]
            if edge_density > 0.3:  # 较多边缘(可能是皱眉等)
                emotion = EmotionCategory.ANGER.value
                confidence = 0.55
                intensity = 0.6
                
        dimension = self.map_to_dimension(emotion, intensity)
        
        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            intensity=intensity,
            timestamp=utc_now(),
            modality=str(self.modality.value),
            details=features,
            dimension=dimension
        )
        
    async def postprocess(self, raw_output: Dict[str, Any]) -> EmotionResult:
        """
        后处理模型输出
        
        Args:
            raw_output: 模型原始输出
            
        Returns:
            EmotionResult: 格式化的情感结果
        """
        predictions = raw_output.get("predictions", [])
        features = raw_output.get("features", {})
        num_faces = raw_output.get("num_faces", 1)
        
        if not predictions:
            return self._create_neutral_result(features)
            
        # 主要情感
        primary = predictions[0] if isinstance(predictions, list) else predictions
        emotion_label = self._standardize_visual_emotion_label(primary["label"])
        confidence = primary["score"]
        
        # 次要情感
        sub_emotions = []
        if isinstance(predictions, list):
            for pred in predictions[1:5]:
                if pred["score"] > 0.1:
                    sub_emotions.append((
                        self._standardize_visual_emotion_label(pred["label"]),
                        pred["score"]
                    ))
                    
        # 基于视觉特征计算强度
        intensity = self._calculate_visual_intensity(confidence, features)
        
        # 映射到VAD维度
        dimension = self.map_to_dimension(emotion_label, intensity)
        
        return EmotionResult(
            emotion=emotion_label,
            confidence=confidence,
            intensity=intensity,
            timestamp=utc_now(),
            modality=str(self.modality.value),
            details={
                "num_faces": num_faces,
                "avg_face_size": features.get("avg_face_size", 0),
                "face_brightness": features.get("face_brightness", 0),
                "face_contrast": features.get("face_contrast", 0),
                "edge_density": features.get("edge_density", 0),
                "raw_predictions": predictions[:3] if isinstance(predictions, list) else [predictions]
            },
            sub_emotions=sub_emotions,
            dimension=dimension
        )
        
    def _standardize_visual_emotion_label(self, label: str) -> str:
        """标准化视觉情感标签"""
        label = label.lower().strip()
        
        # 映射到标准情感类别
        visual_emotion_mapping = {
            "happy": EmotionCategory.HAPPINESS,
            "happiness": EmotionCategory.HAPPINESS,
            "joy": EmotionCategory.JOY,
            "excited": EmotionCategory.EXCITEMENT,
            "sad": EmotionCategory.SADNESS,
            "sadness": EmotionCategory.SADNESS,
            "angry": EmotionCategory.ANGER,
            "anger": EmotionCategory.ANGER,
            "fear": EmotionCategory.FEAR,
            "fearful": EmotionCategory.FEAR,
            "disgust": EmotionCategory.DISGUST,
            "disgusted": EmotionCategory.DISGUST,
            "surprise": EmotionCategory.SURPRISE,
            "surprised": EmotionCategory.SURPRISE,
            "neutral": EmotionCategory.NEUTRAL,
            "contempt": EmotionCategory.DISGUST
        }
        
        for key, value in visual_emotion_mapping.items():
            if key in label:
                return value.value
                
        return label
        
    def _calculate_visual_intensity(self, confidence: float, features: Dict[str, Any]) -> float:
        """基于视觉特征计算情感强度"""
        intensity = confidence
        
        # 对比度影响强度
        if "face_contrast" in features:
            contrast_factor = min(features["face_contrast"] / 100, 1.0)
            intensity = (intensity + contrast_factor) / 2
            
        # 边缘密度影响强度
        if "edge_density" in features:
            edge_factor = min(features["edge_density"] * 2, 1.0)
            intensity = (intensity + edge_factor) / 2
            
        return min(max(intensity, 0.0), 1.0)
        
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
        
    def validate_input(self, image_data: Any) -> bool:
        """验证输入图像"""
        if image_data is None:
            return False
            
        # 检查类型
        valid_types = (np.ndarray, bytes, str, Image.Image)
        if not isinstance(image_data, valid_types):
            return False
            
        return True
        
    async def analyze_video_frame_sequence(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0
    ) -> List[EmotionResult]:
        """
        分析视频帧序列
        
        Args:
            frames: 视频帧列表
            fps: 帧率
            
        Returns:
            List[EmotionResult]: 每帧的情感结果
        """
        results = []
        
        # 采样策略(不需要分析每一帧)
        sample_interval = max(1, int(fps / 5))  # 每秒最多分析5帧
        
        for i in range(0, len(frames), sample_interval):
            frame = frames[i]
            result = await self.analyze(frame)
            results.append(result)
            
        return results
from src.core.logging import get_logger
