"""
实体识别器 - 多模型集成的NER系统

支持spaCy、Transformers、Stanza等多个模型
实现模型结果融合和置信度计算
支持中英文实体识别
"""

import spacy
import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from .data_models import Entity, EntityType

from src.core.logging import get_logger
logger = get_logger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, some NER models will be disabled")

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    logger.warning("Stanza not available, some NER models will be disabled")

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    weight: float
    enabled: bool = True
    model_path: Optional[str] = None
    language: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

class BaseEntityRecognizer(ABC):
    """实体识别器基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    @abstractmethod
    async def load_model(self):
        """加载模型"""
        raise NotImplementedError
    
    @abstractmethod  
    async def extract_entities(self, text: str, language: str = "auto") -> List[Entity]:
        """抽取实体"""
        raise NotImplementedError
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.config.name,
            "weight": self.config.weight,
            "enabled": self.config.enabled,
            "loaded": self._loaded,
            "language": self.config.language
        }

class SpacyEntityRecognizer(BaseEntityRecognizer):
    """spaCy实体识别器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.nlp_models = {}  # 多语言模型缓存
    
    async def load_model(self):
        """加载spaCy模型"""
        try:
            # 根据语言加载不同模型
            models_to_load = {
                "en": "en_core_web_sm",
                "zh": "zh_core_web_sm", 
                "de": "de_core_news_sm",
                "es": "es_core_news_sm",
                "fr": "fr_core_news_sm"
            }
            
            if self.config.language and self.config.language in models_to_load:
                # 只加载指定语言模型
                model_name = models_to_load[self.config.language]
                try:
                    self.nlp_models[self.config.language] = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                except OSError:
                    logger.warning(f"spaCy model {model_name} not found, using en_core_web_sm")
                    self.nlp_models["en"] = spacy.load("en_core_web_sm")
            else:
                # 加载英文模型作为默认
                try:
                    self.nlp_models["en"] = spacy.load("en_core_web_sm")
                    logger.info("Loaded default spaCy model: en_core_web_sm")
                except OSError:
                    logger.error("Cannot load spaCy models")
                    raise RuntimeError("spaCy models not available")
            
            self._loaded = True
            logger.info(f"SpacyEntityRecognizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
    
    async def extract_entities(self, text: str, language: str = "auto") -> List[Entity]:
        """使用spaCy抽取实体"""
        if not self._loaded:
            await self.load_model()
        
        # 语言检测和模型选择
        if language == "auto":
            language = self._detect_language(text)
        
        nlp = self.nlp_models.get(language, self.nlp_models.get("en"))
        if not nlp:
            logger.warning(f"No spaCy model for language {language}, skipping")
            return []
        
        try:
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                # 映射spaCy实体类型到我们的EntityType
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entity = Entity(
                        text=ent.text,
                        label=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.85,  # spaCy默认置信度
                        language=language,
                        metadata={
                            "model": "spacy",
                            "original_label": ent.label_,
                            "model_version": spacy.__version__
                        }
                    )
                    entities.append(entity)
            
            logger.debug(f"SpaCy extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"SpaCy entity extraction failed: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        # 检测中文字符
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return "zh"
        return "en"
    
    def _map_spacy_label(self, label: str) -> Optional[EntityType]:
        """映射spaCy标签到EntityType"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.GPE,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENTAGE,
            "FACILITY": EntityType.FACILITY,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.WORK_OF_ART,
            "LAW": EntityType.LAW,
            "LANGUAGE": EntityType.LANGUAGE,
            "NORP": EntityType.NATIONALITY,
            "CARDINAL": EntityType.CARDINAL,
            "ORDINAL": EntityType.ORDINAL,
            "QUANTITY": EntityType.QUANTITY
        }
        return mapping.get(label)

class TransformerEntityRecognizer(BaseEntityRecognizer):
    """Transformer模型实体识别器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.pipeline = None
    
    async def load_model(self):
        """加载Transformer模型"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available")
            return
        
        try:
            model_name = self.config.model_path or "dbmdz/bert-large-cased-finetuned-conll03-english"
            
            # 创建NER pipeline
            self.pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=-1  # 使用CPU，如需GPU设置为0
            )
            
            self._loaded = True
            logger.info(f"Loaded Transformer model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            raise
    
    async def extract_entities(self, text: str, language: str = "auto") -> List[Entity]:
        """使用Transformer模型抽取实体"""
        if not self._loaded:
            await self.load_model()
        
        if not self.pipeline:
            logger.warning("Transformer pipeline not available")
            return []
        
        try:
            # 运行NER pipeline
            results = self.pipeline(text)
            entities = []
            
            for result in results:
                # 映射BERT标签到我们的EntityType
                entity_type = self._map_bert_label(result["entity_group"])
                if entity_type:
                    entity = Entity(
                        text=result["word"],
                        label=entity_type,
                        start=result["start"],
                        end=result["end"],
                        confidence=result["score"],
                        language=language if language != "auto" else "en",
                        metadata={
                            "model": "transformers",
                            "original_label": result["entity_group"],
                            "model_path": self.config.model_path
                        }
                    )
                    entities.append(entity)
            
            logger.debug(f"Transformer extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Transformer entity extraction failed: {e}")
            return []
    
    def _map_bert_label(self, label: str) -> Optional[EntityType]:
        """映射BERT标签到EntityType"""
        # 处理B-、I-前缀
        if label.startswith(("B-", "I-")):
            label = label[2:]
        
        mapping = {
            "PER": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "MISC": EntityType.MISC
        }
        return mapping.get(label)

class StanzaEntityRecognizer(BaseEntityRecognizer):
    """Stanza实体识别器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.nlp = None
    
    async def load_model(self):
        """加载Stanza模型"""
        if not STANZA_AVAILABLE:
            logger.error("Stanza library not available")
            return
        
        try:
            lang = self.config.language or "en"
            self.nlp = stanza.Pipeline(
                lang,
                processors="tokenize,ner",
                use_gpu=False,
                download_method=None  # 不自动下载
            )
            
            self._loaded = True
            logger.info(f"Loaded Stanza model for language: {lang}")
            
        except Exception as e:
            logger.error(f"Failed to load Stanza model: {e}")
            # Stanza模型加载失败不抛出异常，只是标记为未加载
            self._loaded = False
    
    async def extract_entities(self, text: str, language: str = "auto") -> List[Entity]:
        """使用Stanza抽取实体"""
        if not self._loaded or not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for sent in doc.sentences:
                for ent in sent.ents:
                    entity_type = self._map_stanza_label(ent.type)
                    if entity_type:
                        entity = Entity(
                            text=ent.text,
                            label=entity_type,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.80,  # Stanza默认置信度
                            language=language if language != "auto" else self.config.language,
                            metadata={
                                "model": "stanza",
                                "original_label": ent.type,
                                "model_version": stanza.__version__
                            }
                        )
                        entities.append(entity)
            
            logger.debug(f"Stanza extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Stanza entity extraction failed: {e}")
            return []
    
    def _map_stanza_label(self, label: str) -> Optional[EntityType]:
        """映射Stanza标签到EntityType"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.GPE,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENTAGE,
            "FACILITY": EntityType.FACILITY,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.WORK_OF_ART,
            "LAW": EntityType.LAW,
            "LANGUAGE": EntityType.LANGUAGE,
            "NORP": EntityType.NATIONALITY,
            "CARDINAL": EntityType.CARDINAL,
            "ORDINAL": EntityType.ORDINAL,
            "QUANTITY": EntityType.QUANTITY
        }
        return mapping.get(label)

class MultiModelEntityRecognizer:
    """多模型实体识别器 - 融合多个NER模型结果"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.recognizers: List[BaseEntityRecognizer] = []
        self.model_weights = {}
        self._initialize_recognizers()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "models": {
                "spacy": {"weight": 0.3, "enabled": True},
                "transformers": {
                    "weight": 0.4, 
                    "enabled": TRANSFORMERS_AVAILABLE,
                    "model_path": "dbmdz/bert-large-cased-finetuned-conll03-english"
                },
                "stanza": {"weight": 0.3, "enabled": STANZA_AVAILABLE, "language": "en"}
            },
            "confidence_threshold": 0.5,
            "overlap_threshold": 0.7,  # 实体重叠阈值
            "max_entity_length": 100    # 最大实体长度
        }
    
    def _initialize_recognizers(self):
        """初始化所有识别器"""
        models_config = self.config.get("models", {})
        
        # 初始化spaCy识别器
        spacy_config = models_config.get("spacy", {})
        if spacy_config.get("enabled", True):
            config = ModelConfig(
                name="spacy",
                weight=spacy_config.get("weight", 0.3),
                enabled=True,
                language=spacy_config.get("language")
            )
            self.recognizers.append(SpacyEntityRecognizer(config))
            self.model_weights["spacy"] = config.weight
        
        # 初始化Transformer识别器
        transformers_config = models_config.get("transformers", {})
        if transformers_config.get("enabled", TRANSFORMERS_AVAILABLE):
            config = ModelConfig(
                name="transformers",
                weight=transformers_config.get("weight", 0.4),
                enabled=True,
                model_path=transformers_config.get("model_path")
            )
            self.recognizers.append(TransformerEntityRecognizer(config))
            self.model_weights["transformers"] = config.weight
        
        # 初始化Stanza识别器
        stanza_config = models_config.get("stanza", {})
        if stanza_config.get("enabled", STANZA_AVAILABLE):
            config = ModelConfig(
                name="stanza",
                weight=stanza_config.get("weight", 0.3),
                enabled=True,
                language=stanza_config.get("language", "en")
            )
            self.recognizers.append(StanzaEntityRecognizer(config))
            self.model_weights["stanza"] = config.weight
    
    async def load_models(self):
        """加载所有模型"""
        start_time = time.time()
        
        # 并发加载所有模型
        tasks = []
        for recognizer in self.recognizers:
            if recognizer.config.enabled:
                tasks.append(recognizer.load_model())
        
        # 等待所有模型加载完成，忽略加载失败的模型
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查加载结果
            loaded_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to load model {self.recognizers[i].config.name}: {result}")
                    self.recognizers[i].config.enabled = False
                else:
                    loaded_count += 1
            
            logger.info(f"Loaded {loaded_count}/{len(self.recognizers)} NER models in {time.time() - start_time:.2f}s")
        
        # 重新计算权重（排除未加载的模型）
        self._recompute_weights()
    
    def _recompute_weights(self):
        """重新计算模型权重"""
        loaded_recognizers = [r for r in self.recognizers if r.is_loaded()]
        if not loaded_recognizers:
            logger.error("No NER models loaded successfully")
            return
        
        total_weight = sum(r.config.weight for r in loaded_recognizers)
        if total_weight > 0:
            for recognizer in loaded_recognizers:
                self.model_weights[recognizer.config.name] = (
                    recognizer.config.weight / total_weight
                )
    
    async def extract_entities(
        self, 
        text: str, 
        language: str = "auto",
        confidence_threshold: Optional[float] = None
    ) -> List[Entity]:
        """多模型实体抽取与融合"""
        if not text or not text.strip():
            return []
        
        confidence_threshold = confidence_threshold or self.config.get("confidence_threshold", 0.5)
        
        # 并发运行所有已加载的模型
        tasks = []
        model_names = []
        
        for recognizer in self.recognizers:
            if recognizer.is_loaded():
                tasks.append(recognizer.extract_entities(text, language))
                model_names.append(recognizer.config.name)
        
        if not tasks:
            logger.warning("No loaded NER models available")
            return []
        
        # 等待所有模型结果
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集有效结果
        model_results = {}
        for i, result in enumerate(results):
            model_name = model_names[i]
            if isinstance(result, Exception):
                logger.error(f"Model {model_name} extraction failed: {result}")
                model_results[model_name] = []
            else:
                model_results[model_name] = result or []
                logger.debug(f"Model {model_name} extracted {len(result or [])} entities")
        
        # 融合多模型结果
        merged_entities = self._merge_entities(model_results, confidence_threshold)
        
        processing_time = time.time() - start_time
        logger.info(f"Multi-model NER completed in {processing_time:.2f}s, found {len(merged_entities)} entities")
        
        return merged_entities
    
    def _merge_entities(
        self, 
        model_results: Dict[str, List[Entity]], 
        confidence_threshold: float
    ) -> List[Entity]:
        """融合多个模型的实体识别结果"""
        # 按位置对实体进行分组
        position_groups = defaultdict(list)
        
        for model_name, entities in model_results.items():
            model_weight = self.model_weights.get(model_name, 0.0)
            if model_weight == 0.0:
                continue
                
            for entity in entities:
                # 创建位置键：(start, end, label)
                key = (entity.start, entity.end, entity.label)
                position_groups[key].append((entity, model_weight))
        
        # 合并重叠和相似的实体
        merged_entities = []
        processed_positions = set()
        
        # 按起始位置排序
        sorted_groups = sorted(position_groups.items(), key=lambda x: x[0][0])
        
        for (start, end, label), entity_weight_pairs in sorted_groups:
            if (start, end, label) in processed_positions:
                continue
            
            # 检查与已处理实体的重叠
            overlapping_groups = []
            for other_key, other_pairs in position_groups.items():
                other_start, other_end, other_label = other_key
                
                # 计算重叠比例
                overlap_ratio = self._calculate_overlap(start, end, other_start, other_end)
                
                if (overlap_ratio > self.config.get("overlap_threshold", 0.7) and
                    label == other_label and
                    other_key not in processed_positions):
                    overlapping_groups.append((other_key, other_pairs))
            
            # 合并重叠的实体组
            all_entity_weight_pairs = entity_weight_pairs[:]
            for other_key, other_pairs in overlapping_groups:
                all_entity_weight_pairs.extend(other_pairs)
                processed_positions.add(other_key)
            
            # 计算融合后的实体
            merged_entity = self._compute_merged_entity(all_entity_weight_pairs)
            
            if merged_entity and merged_entity.confidence >= confidence_threshold:
                # 验证实体长度
                if len(merged_entity.text) <= self.config.get("max_entity_length", 100):
                    merged_entities.append(merged_entity)
            
            processed_positions.add((start, end, label))
        
        return merged_entities
    
    def _calculate_overlap(self, start1: int, end1: int, start2: int, end2: int) -> float:
        """计算两个实体位置的重叠比例"""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        min_length = min(end1 - start1, end2 - start2)
        
        return overlap_length / min_length if min_length > 0 else 0.0
    
    def _compute_merged_entity(self, entity_weight_pairs: List[Tuple[Entity, float]]) -> Optional[Entity]:
        """计算多个实体的融合结果"""
        if not entity_weight_pairs:
            return None
        
        # 计算加权平均置信度
        weighted_confidence = 0.0
        total_weight = 0.0
        
        # 选择最佳文本和位置（基于权重）
        best_entity = None
        best_weight = 0.0
        
        # 收集所有模型的信息
        model_info = []
        
        for entity, weight in entity_weight_pairs:
            weighted_confidence += entity.confidence * weight
            total_weight += weight
            
            if weight > best_weight:
                best_entity = entity
                best_weight = weight
            
            model_info.append({
                "model": entity.metadata.get("model", "unknown"),
                "confidence": entity.confidence,
                "weight": weight,
                "original_label": entity.metadata.get("original_label")
            })
        
        if not best_entity or total_weight == 0:
            return None
        
        # 创建融合后的实体
        merged_confidence = weighted_confidence / total_weight
        
        merged_entity = Entity(
            text=best_entity.text,
            label=best_entity.label,
            start=best_entity.start,
            end=best_entity.end,
            confidence=merged_confidence,
            canonical_form=best_entity.canonical_form,
            language=best_entity.language,
            metadata={
                "fusion_method": "weighted_average",
                "model_count": len(entity_weight_pairs),
                "total_weight": total_weight,
                "model_details": model_info,
                "best_model": best_entity.metadata.get("model", "unknown")
            }
        )
        
        return merged_entity
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取所有模型的信息"""
        return {
            "total_models": len(self.recognizers),
            "loaded_models": sum(1 for r in self.recognizers if r.is_loaded()),
            "model_details": [r.get_model_info() for r in self.recognizers],
            "model_weights": self.model_weights,
            "config": self.config
        }
