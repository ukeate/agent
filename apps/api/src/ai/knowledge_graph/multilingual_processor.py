"""
多语言处理器

支持中英文等多语言文本处理
实现语言自动检测功能
跨语言实体对齐和映射
多语言结果统一格式化
"""

import re
import logging
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import unicodedata

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, language detection will use simple heuristics")

from .data_models import Entity, Relation, EntityType, RelationType
from .entity_recognizer import MultiModelEntityRecognizer
from .relation_extractor import RelationExtractor


logger = logging.getLogger(__name__)


class Language(str, Enum):
    """支持的语言"""
    ENGLISH = "en"
    CHINESE = "zh"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    GERMAN = "de"
    SPANISH = "es"
    FRENCH = "fr"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    AUTO = "auto"


@dataclass
class LanguageConfidence:
    """语言检测结果"""
    language: Language
    confidence: float
    script: Optional[str] = None


@dataclass
class MultilingualExtractionResult:
    """多语言抽取结果"""
    text: str
    detected_language: Language
    language_confidence: float
    entities: List[Entity]
    relations: List[Relation]
    aligned_entities: Dict[str, List[Entity]]  # 跨语言对齐的实体
    processing_metadata: Dict[str, Any]


class LanguageDetector:
    """语言检测器"""
    
    def __init__(self):
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.japanese_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
        self.korean_pattern = re.compile(r'[\uac00-\ud7af]')
        self.arabic_pattern = re.compile(r'[\u0600-\u06ff]')
        self.cyrillic_pattern = re.compile(r'[\u0400-\u04ff]')
        
        # 常见语言标识词
        self.language_indicators = {
            Language.ENGLISH: ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"],
            Language.CHINESE: ["的", "是", "在", "了", "和", "有", "与", "为", "等", "中"],
            Language.GERMAN: ["der", "die", "das", "und", "ist", "in", "zu", "den", "von", "mit"],
            Language.SPANISH: ["el", "la", "de", "que", "y", "es", "en", "un", "se", "no"],
            Language.FRENCH: ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
            Language.RUSSIAN: ["и", "в", "не", "на", "я", "быть", "он", "с", "что", "а"],
            Language.JAPANESE: ["の", "に", "は", "を", "が", "と", "で", "て", "も", "か"],
            Language.KOREAN: ["의", "은", "는", "이", "가", "을", "를", "에", "와", "과"]
        }
    
    def detect_language(self, text: str) -> LanguageConfidence:
        """检测文本语言"""
        if not text or not text.strip():
            return LanguageConfidence(Language.ENGLISH, 0.0)
        
        # 首先尝试使用langdetect
        if LANGDETECT_AVAILABLE:
            try:
                detected_langs = detect_langs(text)
                if detected_langs:
                    lang_code = detected_langs[0].lang
                    confidence = detected_langs[0].prob
                    
                    # 映射到我们的Language枚举
                    language = self._map_langdetect_code(lang_code)
                    return LanguageConfidence(language, confidence)
            except LangDetectException:
                logger.debug("langdetect failed, using fallback method")
        
        # 使用基于规则的检测
        return self._rule_based_detection(text)
    
    def _map_langdetect_code(self, lang_code: str) -> Language:
        """映射langdetect语言代码到Language枚举"""
        mapping = {
            "en": Language.ENGLISH,
            "zh-cn": Language.CHINESE_SIMPLIFIED,
            "zh": Language.CHINESE,
            "de": Language.GERMAN,
            "es": Language.SPANISH,
            "fr": Language.FRENCH,
            "ja": Language.JAPANESE,
            "ko": Language.KOREAN,
            "ru": Language.RUSSIAN,
            "ar": Language.ARABIC
        }
        return mapping.get(lang_code, Language.ENGLISH)
    
    def _rule_based_detection(self, text: str) -> LanguageConfidence:
        """基于规则的语言检测"""
        text_lower = text.lower()
        total_chars = len(text)
        
        if total_chars == 0:
            return LanguageConfidence(Language.ENGLISH, 0.0)
        
        # 字符集检测
        chinese_count = len(self.chinese_pattern.findall(text))
        japanese_count = len(self.japanese_pattern.findall(text))
        korean_count = len(self.korean_pattern.findall(text))
        arabic_count = len(self.arabic_pattern.findall(text))
        cyrillic_count = len(self.cyrillic_pattern.findall(text))
        
        # 中文检测
        if chinese_count > total_chars * 0.3:
            confidence = min(chinese_count / total_chars * 2, 1.0)
            return LanguageConfidence(Language.CHINESE, confidence, "cjk")
        
        # 日文检测
        if japanese_count > total_chars * 0.2:
            confidence = min(japanese_count / total_chars * 3, 1.0)
            return LanguageConfidence(Language.JAPANESE, confidence, "japanese")
        
        # 韩文检测
        if korean_count > total_chars * 0.2:
            confidence = min(korean_count / total_chars * 3, 1.0)
            return LanguageConfidence(Language.KOREAN, confidence, "korean")
        
        # 阿拉伯文检测
        if arabic_count > total_chars * 0.2:
            confidence = min(arabic_count / total_chars * 3, 1.0)
            return LanguageConfidence(Language.ARABIC, confidence, "arabic")
        
        # 俄文检测
        if cyrillic_count > total_chars * 0.2:
            confidence = min(cyrillic_count / total_chars * 3, 1.0)
            return LanguageConfidence(Language.RUSSIAN, confidence, "cyrillic")
        
        # 基于关键词的欧洲语言检测
        return self._detect_european_language(text_lower)
    
    def _detect_european_language(self, text_lower: str) -> LanguageConfidence:
        """检测欧洲语言"""
        words = re.findall(r'\b\w+\b', text_lower)
        if not words:
            return LanguageConfidence(Language.ENGLISH, 0.0)
        
        language_scores = {}
        
        for language, indicators in self.language_indicators.items():
            if language in [Language.CHINESE, Language.JAPANESE, Language.KOREAN]:
                continue
            
            score = 0
            for word in words:
                if word in indicators:
                    score += 1
            
            if words:
                language_scores[language] = score / len(words)
        
        if not language_scores:
            return LanguageConfidence(Language.ENGLISH, 0.5)
        
        best_language = max(language_scores.items(), key=lambda x: x[1])
        return LanguageConfidence(best_language[0], min(best_language[1] * 2, 1.0))


class TextNormalizer:
    """文本规范化器"""
    
    def __init__(self):
        # Unicode规范化配置
        self.normalization_forms = {
            Language.CHINESE: "NFKC",
            Language.JAPANESE: "NFKC", 
            Language.KOREAN: "NFKC",
            Language.ARABIC: "NFKD",
            Language.ENGLISH: "NFC"
        }
        
        # 语言特定的清理规则
        self.cleaning_rules = {
            Language.CHINESE: {
                "remove_patterns": [r'[a-zA-Z\s]+(?=[。，！？；：])', r'\s+'],
                "replace_patterns": [(r'([。，！？；：])\s*', r'\1')]
            },
            Language.ENGLISH: {
                "remove_patterns": [r'\s+'],
                "replace_patterns": [(r'([.!?])\s*', r'\1 ')]
            }
        }
    
    def normalize_text(self, text: str, language: Language) -> str:
        """规范化文本"""
        if not text:
            return text
        
        # Unicode规范化
        normalization_form = self.normalization_forms.get(language, "NFC")
        normalized_text = unicodedata.normalize(normalization_form, text)
        
        # 应用语言特定的清理规则
        if language in self.cleaning_rules:
            rules = self.cleaning_rules[language]
            
            # 移除模式
            for pattern in rules.get("remove_patterns", []):
                normalized_text = re.sub(pattern, "", normalized_text)
            
            # 替换模式
            for old_pattern, new_pattern in rules.get("replace_patterns", []):
                normalized_text = re.sub(old_pattern, new_pattern, normalized_text)
        
        return normalized_text.strip()


class CrossLingualEntityAligner:
    """跨语言实体对齐器"""
    
    def __init__(self):
        # 跨语言实体类型映射
        self.cross_lingual_type_mapping = {
            "人物": EntityType.PERSON,
            "人名": EntityType.PERSON,
            "组织": EntityType.ORGANIZATION,
            "机构": EntityType.ORGANIZATION,
            "公司": EntityType.ORGANIZATION,
            "地点": EntityType.LOCATION,
            "地名": EntityType.LOCATION,
            "国家": EntityType.COUNTRY,
            "城市": EntityType.CITY,
            "时间": EntityType.DATE,
            "日期": EntityType.DATE,
            "金额": EntityType.MONEY,
            "货币": EntityType.MONEY
        }
        
        # 常见实体翻译映射
        self.entity_translations = {
            "United States": ["美国", "美利坚合众国"],
            "China": ["中国", "中华人民共和国"],
            "Japan": ["日本"],
            "Germany": ["德国"],
            "France": ["法国"],
            "United Kingdom": ["英国", "英国"],
            "Apple": ["苹果", "苹果公司"],
            "Microsoft": ["微软", "微软公司"],
            "Google": ["谷歌", "谷歌公司"],
            "Amazon": ["亚马逊", "亚马逊公司"]
        }
    
    def align_entities(
        self, 
        entities_by_language: Dict[Language, List[Entity]]
    ) -> Dict[str, List[Entity]]:
        """跨语言实体对齐"""
        if len(entities_by_language) <= 1:
            # 只有一种语言，无需对齐
            all_entities = []
            for entities in entities_by_language.values():
                all_entities.extend(entities)
            return {"default": all_entities}
        
        aligned_groups = {}
        processed_entities = set()
        
        # 获取所有实体
        all_entities = []
        for entities in entities_by_language.values():
            all_entities.extend(entities)
        
        group_id = 0
        
        for entity in all_entities:
            if entity.entity_id in processed_entities:
                continue
            
            # 查找对齐的实体
            aligned_entities = [entity]
            processed_entities.add(entity.entity_id)
            
            for other_entity in all_entities:
                if (other_entity.entity_id in processed_entities or 
                    entity.entity_id == other_entity.entity_id):
                    continue
                
                if self._are_entities_aligned(entity, other_entity):
                    aligned_entities.append(other_entity)
                    processed_entities.add(other_entity.entity_id)
            
            # 创建对齐组
            group_key = f"group_{group_id}"
            aligned_groups[group_key] = aligned_entities
            group_id += 1
        
        return aligned_groups
    
    def _are_entities_aligned(self, entity1: Entity, entity2: Entity) -> bool:
        """判断两个实体是否应该对齐"""
        # 实体类型必须相同或兼容
        if not self._are_types_compatible(entity1.label, entity2.label):
            return False
        
        # 文本相似性检查
        text1 = entity1.canonical_form or entity1.text
        text2 = entity2.canonical_form or entity2.text
        
        # 完全匹配
        if text1.lower() == text2.lower():
            return True
        
        # 翻译映射检查
        if self._check_translation_mapping(text1, text2):
            return True
        
        # 数字和日期的跨语言匹配
        if entity1.label in [EntityType.DATE, EntityType.MONEY, EntityType.CARDINAL]:
            return self._align_numeric_entities(text1, text2)
        
        return False
    
    def _are_types_compatible(self, type1: EntityType, type2: EntityType) -> bool:
        """检查实体类型是否兼容"""
        if type1 == type2:
            return True
        
        # 兼容类型组
        compatible_groups = [
            {EntityType.LOCATION, EntityType.GPE, EntityType.COUNTRY, EntityType.CITY},
            {EntityType.ORGANIZATION, EntityType.COMPANY},
            {EntityType.DATE, EntityType.TIME}
        ]
        
        for group in compatible_groups:
            if type1 in group and type2 in group:
                return True
        
        return False
    
    def _check_translation_mapping(self, text1: str, text2: str) -> bool:
        """检查翻译映射"""
        # 正向映射
        translations = self.entity_translations.get(text1, [])
        if text2 in translations:
            return True
        
        # 反向映射
        for english_name, translations in self.entity_translations.items():
            if text1 in translations and text2 == english_name:
                return True
            if text2 in translations and text1 == english_name:
                return True
        
        return False
    
    def _align_numeric_entities(self, text1: str, text2: str) -> bool:
        """对齐数字实体"""
        # 提取数字
        numbers1 = re.findall(r'\d+', text1)
        numbers2 = re.findall(r'\d+', text2)
        
        # 如果数字相同，认为是同一实体
        return numbers1 == numbers2 and len(numbers1) > 0


class MultilingualProcessor:
    """多语言处理器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self.language_detector = LanguageDetector()
        self.text_normalizer = TextNormalizer()
        self.entity_aligner = CrossLingualEntityAligner()
        
        # 初始化处理器（延迟加载）
        self.entity_recognizers = {}
        self.relation_extractors = {}
        
        # 支持的语言
        self.supported_languages = [
            Language.ENGLISH,
            Language.CHINESE,
            Language.CHINESE_SIMPLIFIED,
            Language.GERMAN,
            Language.SPANISH,
            Language.FRENCH
        ]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "auto_detect_language": True,
            "normalize_text": True,
            "align_cross_lingual_entities": True,
            "confidence_threshold": 0.6,
            "language_detection_threshold": 0.7,
            "fallback_language": Language.ENGLISH,
            "max_supported_languages": 3
        }
    
    async def initialize_processors(self, languages: Optional[List[Language]] = None):
        """初始化语言处理器"""
        if not languages:
            languages = self.supported_languages
        
        # 限制支持的语言数量
        max_languages = self.config.get("max_supported_languages", 3)
        languages = languages[:max_languages]
        
        # 初始化每种语言的处理器
        for language in languages:
            if language not in self.entity_recognizers:
                # 创建语言特定的实体识别器配置
                recognizer_config = self._get_language_recognizer_config(language)
                self.entity_recognizers[language] = MultiModelEntityRecognizer(recognizer_config)
                await self.entity_recognizers[language].load_models()
            
            if language not in self.relation_extractors:
                # 创建语言特定的关系抽取器配置
                extractor_config = self._get_language_extractor_config(language)
                self.relation_extractors[language] = RelationExtractor(extractor_config)
        
        logger.info(f"Initialized processors for {len(languages)} languages")
    
    def _get_language_recognizer_config(self, language: Language) -> Dict[str, Any]:
        """获取语言特定的实体识别器配置"""
        base_config = {
            "confidence_threshold": self.config.get("confidence_threshold", 0.6),
            "models": {
                "spacy": {"weight": 0.3, "enabled": True},
                "transformers": {"weight": 0.4, "enabled": True},
                "stanza": {"weight": 0.3, "enabled": True}
            }
        }
        
        # 语言特定配置
        if language == Language.CHINESE:
            base_config["models"]["spacy"]["language"] = "zh"
            base_config["models"]["stanza"]["language"] = "zh"
            base_config["models"]["transformers"]["model_path"] = "ckiplab/bert-base-chinese-ner"
        elif language == Language.GERMAN:
            base_config["models"]["spacy"]["language"] = "de"
            base_config["models"]["stanza"]["language"] = "de"
            base_config["models"]["transformers"]["model_path"] = "dbmdz/bert-large-cased-finetuned-conll03-english"
        elif language == Language.SPANISH:
            base_config["models"]["spacy"]["language"] = "es"
            base_config["models"]["stanza"]["language"] = "es"
        elif language == Language.FRENCH:
            base_config["models"]["spacy"]["language"] = "fr"
            base_config["models"]["stanza"]["language"] = "fr"
        
        return base_config
    
    def _get_language_extractor_config(self, language: Language) -> Dict[str, Any]:
        """获取语言特定的关系抽取器配置"""
        base_config = {
            "confidence_threshold": self.config.get("confidence_threshold", 0.6),
            "use_pattern_based": True,
            "use_dependency_based": True
        }
        
        # 中文需要特殊处理
        if language == Language.CHINESE:
            base_config["max_entity_distance"] = 50  # 中文字符间距更小
        
        return base_config
    
    async def process_multilingual_text(
        self, 
        text: str, 
        target_language: Optional[Language] = None
    ) -> MultilingualExtractionResult:
        """处理多语言文本"""
        if not text or not text.strip():
            return MultilingualExtractionResult(
                text=text,
                detected_language=Language.ENGLISH,
                language_confidence=0.0,
                entities=[],
                relations=[],
                aligned_entities={},
                processing_metadata={}
            )
        
        start_time = asyncio.get_event_loop().time()
        
        # 语言检测
        if target_language and target_language != Language.AUTO:
            detected_language = target_language
            language_confidence = 1.0
        else:
            lang_result = self.language_detector.detect_language(text)
            detected_language = lang_result.language
            language_confidence = lang_result.confidence
        
        # 文本规范化
        if self.config.get("normalize_text", True):
            normalized_text = self.text_normalizer.normalize_text(text, detected_language)
        else:
            normalized_text = text
        
        # 确保有对应语言的处理器
        if detected_language not in self.entity_recognizers:
            await self.initialize_processors([detected_language])
        
        # 如果检测置信度太低，使用fallback语言
        if language_confidence < self.config.get("language_detection_threshold", 0.7):
            fallback_lang = self.config.get("fallback_language", Language.ENGLISH)
            if fallback_lang not in self.entity_recognizers:
                await self.initialize_processors([fallback_lang])
            detected_language = fallback_lang
        
        # 实体识别
        entities = []
        if detected_language in self.entity_recognizers:
            try:
                entities = await self.entity_recognizers[detected_language].extract_entities(
                    normalized_text, detected_language.value
                )
            except Exception as e:
                logger.error(f"Entity recognition failed for language {detected_language}: {e}")
        
        # 关系抽取
        relations = []
        if entities and detected_language in self.relation_extractors:
            try:
                relations = await self.relation_extractors[detected_language].extract_relations(
                    normalized_text, entities
                )
            except Exception as e:
                logger.error(f"Relation extraction failed for language {detected_language}: {e}")
        
        # 跨语言实体对齐（如果需要）
        aligned_entities = {}
        if self.config.get("align_cross_lingual_entities", True):
            entities_by_language = {detected_language: entities}
            aligned_entities = self.entity_aligner.align_entities(entities_by_language)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # 构建结果
        result = MultilingualExtractionResult(
            text=normalized_text,
            detected_language=detected_language,
            language_confidence=language_confidence,
            entities=entities,
            relations=relations,
            aligned_entities=aligned_entities,
            processing_metadata={
                "original_text": text,
                "processing_time": processing_time,
                "normalization_applied": self.config.get("normalize_text", True),
                "entity_count": len(entities),
                "relation_count": len(relations),
                "aligned_groups": len(aligned_entities)
            }
        )
        
        logger.info(
            f"Multilingual processing completed for {detected_language.value} "
            f"in {processing_time:.2f}s: {len(entities)} entities, {len(relations)} relations"
        )
        
        return result
    
    async def process_multiple_languages(
        self, 
        texts: List[Tuple[str, Optional[Language]]]
    ) -> List[MultilingualExtractionResult]:
        """处理多个语言的文本"""
        if not texts:
            return []
        
        # 并发处理所有文本
        tasks = []
        for text, language in texts:
            task = self.process_multilingual_text(text, language)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process text {i}: {result}")
                # 创建空结果
                text, language = texts[i]
                empty_result = MultilingualExtractionResult(
                    text=text,
                    detected_language=language or Language.ENGLISH,
                    language_confidence=0.0,
                    entities=[],
                    relations=[],
                    aligned_entities={},
                    processing_metadata={"error": str(result)}
                )
                valid_results.append(empty_result)
            else:
                valid_results.append(result)
        
        return valid_results
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """获取支持的语言列表"""
        languages = []
        for lang in self.supported_languages:
            languages.append({
                "code": lang.value,
                "name": lang.name,
                "loaded": lang in self.entity_recognizers
            })
        return languages
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            "supported_languages": len(self.supported_languages),
            "loaded_recognizers": len(self.entity_recognizers),
            "loaded_extractors": len(self.relation_extractors),
            "config": self.config,
            "detector_available": LANGDETECT_AVAILABLE
        }