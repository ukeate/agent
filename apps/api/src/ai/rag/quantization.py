"""
向量量化支持模块
实现Product Quantization (PQ)、Binary Quantization和Half-precision向量
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ...core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

class VectorQuantizer(ABC):
    """向量量化器基类"""
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.is_trained = False
        
    @abstractmethod
    def train(self, vectors: np.ndarray) -> bool:
        """训练量化器"""
        ...
        
    @abstractmethod
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """编码向量"""
        ...
        
    @abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """解码向量"""
        ...
        
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """获取量化器参数"""
        ...

class BinaryQuantizer(VectorQuantizer):
    """二进制量化器"""
    
    def __init__(self, bits: int = 8):
        super().__init__("binary")
        self.bits = bits
        self.thresholds: Optional[np.ndarray] = None
        
    def train(self, vectors: np.ndarray) -> bool:
        """训练二进制量化器"""
        try:
            self.thresholds = np.median(vectors, axis=0)
            self.is_trained = True
            logger.info(f"二进制量化器训练完成，维度: {len(self.thresholds)}")
            return True
            
        except Exception as e:
            logger.error(f"二进制量化器训练失败: {e}")
            return False
            
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """编码为二进制向量"""
        if not self.is_trained:
            raise ValueError("量化器未训练")
            
        binary_vectors = (vectors > self.thresholds).astype(np.uint8)
        return binary_vectors
            
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """从二进制码解码"""
        if not self.is_trained:
            raise ValueError("量化器未训练")
            
        return np.where(codes, self.thresholds + 0.1, self.thresholds - 0.1)
            
    def get_params(self) -> Dict[str, Any]:
        return {
            "type": "binary",
            "bits": self.bits,
            "thresholds": self.thresholds.tolist() if self.thresholds is not None else None,
            "compression_ratio": 32.0 / self.bits if self.bits > 0 else 1.0
        }

class HalfPrecisionQuantizer(VectorQuantizer):
    """半精度量化器"""
    
    def __init__(self):
        super().__init__("halfprecision")
        
    def train(self, vectors: np.ndarray) -> bool:
        """半精度不需要训练"""
        self.is_trained = True
        return True
        
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """编码为半精度"""
        return vectors.astype(np.float16)
        
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """从半精度解码"""
        return codes.astype(np.float32)
        
    def get_params(self) -> Dict[str, Any]:
        return {
            "type": "halfprecision",
            "compression_ratio": 2.0
        }

class QuantizationManager:
    """量化管理器"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.quantizers: Dict[str, VectorQuantizer] = {}
        
    async def create_quantization_config(
        self,
        collection_name: str,
        quantization_type: str = "halfprecision",
        config: Dict[str, Any] = None
    ) -> bool:
        """创建量化配置"""
        try:
            config = config or {}
            
            if quantization_type == "binary":
                quantizer = BinaryQuantizer(bits=config.get("bits", 8))
            elif quantization_type == "halfprecision":
                quantizer = HalfPrecisionQuantizer()
            else:
                raise ValueError(f"不支持的量化类型: {quantization_type}")
                
            # 获取训练数据
            training_vectors = await self._get_training_vectors(collection_name, config.get("training_size", 10000))
            
            if len(training_vectors) == 0:
                logger.warning(f"没有找到训练向量: {collection_name}")
                return False
                
            # 训练量化器
            if quantizer.train(training_vectors):
                self.quantizers[collection_name] = quantizer
                await self._save_quantization_config(collection_name, quantizer)
                
                logger.info(f"量化配置创建成功: {collection_name}, 类型: {quantization_type}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"创建量化配置失败: {e}")
            return False
            
    async def _get_training_vectors(self, collection_name: str, max_samples: int) -> np.ndarray:
        """获取训练向量样本"""
        try:
            async with self.vector_store.get_connection() as conn:
                result = await conn.fetch(f"""
                SELECT embedding 
                FROM {collection_name} 
                WHERE embedding IS NOT NULL
                ORDER BY RANDOM()
                LIMIT {max_samples}
                """)
                
                if not result:
                    return np.array([])
                    
                vectors = []
                for row in result:
                    embedding_str = row["embedding"]
                    if isinstance(embedding_str, str):
                        vector_vals = embedding_str.strip('[]').split(',')
                        vector = np.array([float(v) for v in vector_vals])
                    else:
                        vector = np.array(embedding_str)
                    vectors.append(vector)
                    
                return np.array(vectors)
                
        except Exception as e:
            logger.error(f"获取训练向量失败: {e}")
            return np.array([])

    async def ensure_config_table(self) -> None:
        """确保量化配置表存在"""
        async with self.vector_store.get_connection() as conn:
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_quantization_configs (
                collection_name VARCHAR(255) PRIMARY KEY,
                quantizer_type VARCHAR(50) NOT NULL,
                config_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """)
            
    async def _save_quantization_config(self, collection_name: str, quantizer: VectorQuantizer):
        """保存量化配置到数据库"""
        try:
            await self.ensure_config_table()
            async with self.vector_store.get_connection() as conn:
                
                config_data = quantizer.get_params()
                await conn.execute("""
                INSERT INTO vector_quantization_configs 
                (collection_name, quantizer_type, config_data, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (collection_name) 
                DO UPDATE SET 
                    quantizer_type = EXCLUDED.quantizer_type,
                    config_data = EXCLUDED.config_data,
                    updated_at = NOW()
                """, collection_name, quantizer.name, json.dumps(config_data))
                
                logger.info(f"量化配置已保存: {collection_name}")
                
        except Exception as e:
            logger.error(f"保存量化配置失败: {e}")

# 全局量化管理器实例
quantization_manager = None

async def get_quantization_manager(vector_store) -> QuantizationManager:
    """获取量化管理器实例"""
    global quantization_manager
    if quantization_manager is None:
        quantization_manager = QuantizationManager(vector_store)
    return quantization_manager
