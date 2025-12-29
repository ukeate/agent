"""
多模态向量搜索引擎

支持图像、文本、音频等多种模态的向量搜索和跨模态检索
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import json
import base64
import io
from PIL import Image
import hashlib
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
logger = get_logger(__name__)

class ModalityType(str, Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class EncodingModel(str, Enum):
    """编码模型类型"""
    CLIP = "clip"                    # OpenAI CLIP
    BLIP = "blip"                    # Salesforce BLIP
    WHISPER = "whisper"              # OpenAI Whisper
    IMAGEBIND = "imagebind"          # Meta ImageBind
    CUSTOM = "custom"                # 自定义模型

@dataclass
class MultimodalVector:
    """多模态向量"""
    modality: ModalityType
    vector: np.ndarray
    metadata: Dict[str, Any]
    encoding_model: EncodingModel
    dimension: int
    timestamp: datetime

@dataclass
class MultimodalSearchConfig:
    """多模态搜索配置"""
    source_modality: ModalityType
    target_modality: ModalityType
    encoding_model: EncodingModel = EncodingModel.CLIP
    fusion_strategy: str = "late"    # early, late, hybrid
    top_k: int = 10
    enable_reranking: bool = True
    similarity_threshold: float = 0.7

class MultimodalSearchEngine:
    """多模态搜索引擎"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.encoders = {}
        self.search_stats = {
            "text_searches": 0,
            "image_searches": 0,
            "audio_searches": 0,
            "cross_modal_searches": 0,
            "avg_encoding_time_ms": 0.0,
            "avg_search_time_ms": 0.0
        }
        
    async def encode_image(
        self,
        image_data: Union[bytes, BinaryIO, str],
        model: EncodingModel = EncodingModel.CLIP
    ) -> np.ndarray:
        """编码图像为向量"""
        try:
            start_time = asyncio.get_running_loop().time()
            
            # 处理不同类型的输入
            if isinstance(image_data, str):
                # Base64编码的图像
                image_bytes = base64.b64decode(image_data)
            elif isinstance(image_data, BinaryIO):
                image_bytes = image_data.read()
            else:
                image_bytes = image_data
            
            # 使用PIL加载图像
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except Exception:
                # 如果不是有效的图像，创建一个默认图像
                image = Image.new('RGB', (224, 224), color='white')
            
            # 预处理图像
            processed_image = await self._preprocess_image(image, model)
            
            # 模拟图像编码（实际应用中调用真实的模型）
            if model == EncodingModel.CLIP:
                vector = await self._encode_with_clip(processed_image)
            elif model == EncodingModel.BLIP:
                vector = await self._encode_with_blip(processed_image)
            else:
                vector = await self._encode_with_generic_model(processed_image)
            
            # 更新统计
            end_time = asyncio.get_running_loop().time()
            self._update_encoding_stats((end_time - start_time) * 1000)
            
            return vector
            
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            raise
    
    async def encode_audio(
        self,
        audio_data: Union[bytes, BinaryIO],
        sample_rate: int = 16000,
        model: EncodingModel = EncodingModel.WHISPER
    ) -> np.ndarray:
        """编码音频为向量"""
        try:
            start_time = asyncio.get_running_loop().time()
            
            # 处理音频数据
            if isinstance(audio_data, BinaryIO):
                audio_bytes = audio_data.read()
            else:
                audio_bytes = audio_data
            
            # 预处理音频
            processed_audio = await self._preprocess_audio(
                audio_bytes, sample_rate, model
            )
            
            # 模拟音频编码
            if model == EncodingModel.WHISPER:
                vector = await self._encode_with_whisper(processed_audio)
            elif model == EncodingModel.IMAGEBIND:
                vector = await self._encode_with_imagebind(processed_audio)
            else:
                vector = await self._encode_audio_generic(processed_audio)
            
            # 更新统计
            end_time = asyncio.get_running_loop().time()
            self._update_encoding_stats((end_time - start_time) * 1000)
            
            return vector
            
        except Exception as e:
            logger.error(f"音频编码失败: {e}")
            raise
    
    async def encode_text(
        self,
        text: str,
        model: EncodingModel = EncodingModel.CLIP
    ) -> np.ndarray:
        """编码文本为向量"""
        raise RuntimeError("text encoder not integrated")
    
    async def cross_modal_search(
        self,
        query: Union[str, bytes, np.ndarray],
        config: MultimodalSearchConfig
    ) -> List[Dict[str, Any]]:
        """跨模态搜索"""
        try:
            start_time = asyncio.get_running_loop().time()
            
            # 编码查询
            if config.source_modality == ModalityType.TEXT:
                query_vector = await self.encode_text(query, config.encoding_model)
            elif config.source_modality == ModalityType.IMAGE:
                query_vector = await self.encode_image(query, config.encoding_model)
            elif config.source_modality == ModalityType.AUDIO:
                query_vector = await self.encode_audio(query, model=config.encoding_model)
            else:
                query_vector = query if isinstance(query, np.ndarray) else np.array(query)
            
            # 执行向量搜索
            results = await self._vector_search(
                query_vector,
                config.target_modality,
                config.top_k,
                config.similarity_threshold
            )
            
            # 重排序（如果启用）
            if config.enable_reranking:
                results = await self._rerank_results(
                    query_vector, results, config
                )
            
            # 更新统计
            end_time = asyncio.get_running_loop().time()
            self._update_search_stats(
                config.source_modality,
                config.target_modality,
                (end_time - start_time) * 1000
            )
            
            self.search_stats["cross_modal_searches"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"跨模态搜索失败: {e}")
            return []
    
    async def multimodal_fusion_search(
        self,
        queries: Dict[ModalityType, Any],
        fusion_weights: Optional[Dict[ModalityType, float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """多模态融合搜索"""
        try:
            if fusion_weights is None:
                # 默认等权重
                fusion_weights = {
                    modality: 1.0 / len(queries)
                    for modality in queries.keys()
                }
            
            # 并行编码所有模态
            encoding_tasks = []
            modalities = []
            
            for modality, query_data in queries.items():
                if modality == ModalityType.TEXT:
                    task = self.encode_text(query_data)
                elif modality == ModalityType.IMAGE:
                    task = self.encode_image(query_data)
                elif modality == ModalityType.AUDIO:
                    task = self.encode_audio(query_data)
                else:
                    continue
                
                encoding_tasks.append(task)
                modalities.append(modality)
            
            # 获取所有向量
            vectors = await asyncio.gather(*encoding_tasks)
            
            # 融合向量
            fused_vector = await self._fuse_vectors(
                dict(zip(modalities, vectors)),
                fusion_weights
            )
            
            # 执行搜索
            results = await self._vector_search(
                fused_vector,
                ModalityType.MULTIMODAL,
                top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"多模态融合搜索失败: {e}")
            return []
    
    async def _preprocess_image(
        self,
        image: Image.Image,
        model: EncodingModel
    ) -> np.ndarray:
        """预处理图像"""
        # 根据模型调整图像大小
        if model == EncodingModel.CLIP:
            target_size = (224, 224)
        elif model == EncodingModel.BLIP:
            target_size = (384, 384)
        else:
            target_size = (256, 256)
        
        # 调整大小
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        return image_array
    
    async def _preprocess_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        model: EncodingModel
    ) -> np.ndarray:
        """预处理音频"""
        # 这里应该使用实际的音频处理库（如librosa）
        # 暂时返回模拟数据
        
        # 模拟音频特征提取
        audio_features = np.frombuffer(audio_bytes[:1024], dtype=np.float32)
        
        # 重采样到目标采样率
        if model == EncodingModel.WHISPER:
            target_rate = 16000
        else:
            target_rate = 22050
        
        # 这里应该进行实际的重采样
        # resampled = librosa.resample(audio_features, sample_rate, target_rate)
        
        return audio_features
    
    async def _encode_with_clip(self, image: np.ndarray) -> np.ndarray:
        """使用CLIP编码图像"""
        raise RuntimeError("CLIP encoder not integrated")
    
    async def _encode_with_blip(self, image: np.ndarray) -> np.ndarray:
        """使用BLIP编码图像"""
        raise RuntimeError("BLIP encoder not integrated")
    
    async def _encode_with_generic_model(self, data: np.ndarray) -> np.ndarray:
        """使用通用模型编码"""
        raise RuntimeError("generic image encoder not integrated")
    
    async def _encode_with_whisper(self, audio: np.ndarray) -> np.ndarray:
        """使用Whisper编码音频"""
        raise RuntimeError("Whisper encoder not integrated")
    
    async def _encode_with_imagebind(self, data: np.ndarray) -> np.ndarray:
        """使用ImageBind编码"""
        raise RuntimeError("ImageBind encoder not integrated")
    
    async def _encode_audio_generic(self, audio: np.ndarray) -> np.ndarray:
        """通用音频编码"""
        raise RuntimeError("generic audio encoder not integrated")
    
    async def _vector_search(
        self,
        query_vector: np.ndarray,
        target_modality: ModalityType,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """执行向量搜索"""
        try:
            # 构建搜索SQL
            search_sql = """
            SELECT 
                id,
                content,
                modality,
                metadata,
                embedding <=> %s::vector AS distance,
                1 - (embedding <=> %s::vector) AS similarity
            FROM multimodal_items
            WHERE modality = %s
                AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            vector_list = query_vector.tolist()
            
            result = await self.db.execute(
                text(search_sql),
                (
                    vector_list,
                    vector_list,
                    target_modality.value,
                    vector_list,
                    similarity_threshold,
                    vector_list,
                    top_k
                )
            )
            
            results = []
            for row in result.fetchall():
                results.append({
                    "id": str(row.id),
                    "content": row.content,
                    "modality": row.modality,
                    "metadata": row.metadata or {},
                    "distance": float(row.distance),
                    "similarity": float(row.similarity)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    async def _rerank_results(
        self,
        query_vector: np.ndarray,
        results: List[Dict[str, Any]],
        config: MultimodalSearchConfig
    ) -> List[Dict[str, Any]]:
        """重排序结果"""
        # 这里可以实现更复杂的重排序逻辑
        # 例如使用交叉注意力机制或更精细的相似度计算
        
        for result in results:
            # 添加额外的评分因素
            modality_bonus = 0.1 if result["modality"] == config.target_modality.value else 0
            result["final_score"] = result["similarity"] + modality_bonus
        
        # 按最终分数排序
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return results
    
    async def _fuse_vectors(
        self,
        vectors: Dict[ModalityType, np.ndarray],
        weights: Dict[ModalityType, float]
    ) -> np.ndarray:
        """融合多个模态的向量"""
        # 加权平均融合
        fused = np.zeros_like(list(vectors.values())[0])
        
        for modality, vector in vectors.items():
            weight = weights.get(modality, 1.0)
            fused += weight * vector
        
        # 归一化
        fused = fused / np.linalg.norm(fused)
        
        return fused
    
    def _update_encoding_stats(self, encoding_time_ms: float) -> None:
        """更新编码统计"""
        n = sum([
            self.search_stats["text_searches"],
            self.search_stats["image_searches"],
            self.search_stats["audio_searches"],
            self.search_stats["cross_modal_searches"]
        ]) + 1
        
        if n == 1:
            self.search_stats["avg_encoding_time_ms"] = encoding_time_ms
        else:
            current_avg = self.search_stats["avg_encoding_time_ms"]
            self.search_stats["avg_encoding_time_ms"] = (
                (current_avg * (n - 1) + encoding_time_ms) / n
            )
    
    def _update_search_stats(
        self,
        source_modality: ModalityType,
        target_modality: ModalityType,
        search_time_ms: float
    ) -> None:
        """更新搜索统计"""
        # 更新模态计数
        if source_modality == ModalityType.TEXT:
            self.search_stats["text_searches"] += 1
        elif source_modality == ModalityType.IMAGE:
            self.search_stats["image_searches"] += 1
        elif source_modality == ModalityType.AUDIO:
            self.search_stats["audio_searches"] += 1
        
        # 更新平均搜索时间
        n = sum([
            self.search_stats["text_searches"],
            self.search_stats["image_searches"],
            self.search_stats["audio_searches"],
            self.search_stats["cross_modal_searches"]
        ])
        
        if n == 1:
            self.search_stats["avg_search_time_ms"] = search_time_ms
        else:
            current_avg = self.search_stats["avg_search_time_ms"]
            self.search_stats["avg_search_time_ms"] = (
                (current_avg * (n - 1) + search_time_ms) / n
            )
    
    async def create_multimodal_table(self) -> bool:
        """创建多模态数据表"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS multimodal_items (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT,
                modality VARCHAR(50) NOT NULL,
                encoding_model VARCHAR(50),
                metadata JSONB DEFAULT '{}',
                embedding VECTOR(1024),  -- 支持不同维度的向量
                raw_data BYTEA,          -- 原始数据（图像、音频等）
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 创建索引
            CREATE INDEX IF NOT EXISTS idx_multimodal_modality 
            ON multimodal_items(modality);
            
            CREATE INDEX IF NOT EXISTS idx_multimodal_embedding_hnsw
            ON multimodal_items USING hnsw (embedding vector_cosine_ops);
            """
            
            await self.db.execute(text(create_table_sql))
            await self.db.commit()
            
            logger.info("多模态数据表创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建多模态表失败: {e}")
            await self.db.rollback()
            return False
    
    async def store_multimodal_vector(
        self,
        vector: np.ndarray,
        modality: ModalityType,
        content: Optional[str] = None,
        raw_data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        encoding_model: Optional[EncodingModel] = None
    ) -> str:
        """存储多模态向量"""
        try:
            insert_sql = """
            INSERT INTO multimodal_items 
            (content, modality, encoding_model, metadata, embedding, raw_data)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            result = await self.db.execute(
                text(insert_sql),
                (
                    content,
                    modality.value,
                    encoding_model.value if encoding_model else None,
                    json.dumps(metadata) if metadata else "{}",
                    vector.tolist(),
                    raw_data
                )
            )
            
            row = result.fetchone()
            await self.db.commit()
            
            return str(row.id)
            
        except Exception as e:
            logger.error(f"存储多模态向量失败: {e}")
            await self.db.rollback()
            raise
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """获取搜索统计"""
        return self.search_stats.copy()
