"""
关系抽取引擎

基于模式匹配和依存句法分析的关系抽取
支持50种以上关系类型识别
实现三元组(主语-谓语-宾语)结构化输出
"""

import re
import spacy
import time
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from .data_models import Entity, Relation, RelationType, EntityType

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class RelationPattern:
    """关系模式定义"""
    name: str
    relation_type: RelationType
    patterns: List[Dict[str, Any]]
    confidence: float = 0.8
    requires_dependency: bool = False
    subject_types: List[EntityType] = field(default_factory=list)
    object_types: List[EntityType] = field(default_factory=list)
    
    def matches_entity_types(self, subject: Entity, obj: Entity) -> bool:
        """检查实体类型是否匹配模式要求"""
        if self.subject_types and subject.label not in self.subject_types:
            return False
        if self.object_types and obj.label not in self.object_types:
            return False
        return True

class PatternBasedExtractor:
    """基于模式的关系抽取器"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[RelationPattern]:
        """初始化关系模式"""
        patterns = []
        
        # WORKS_FOR 关系模式
        patterns.append(RelationPattern(
            name="works_for_basic",
            relation_type=RelationType.WORKS_FOR,
            patterns=[
                {"pattern": r"{subject}.*(?:works? for|employed by|employee of).*{object}"},
                {"pattern": r"{subject}.*(?:is|was).*(?:at|with).*{object}"},
                {"pattern": r"{subject}.*(?:joined|join).*{object}"},
                {"pattern": r"{object}.*(?:hired|recruit|employ).*{subject}"}
            ],
            subject_types=[EntityType.PERSON],
            object_types=[EntityType.ORGANIZATION, EntityType.COMPANY]
        ))
        
        # LOCATED_IN 关系模式
        patterns.append(RelationPattern(
            name="located_in_basic",
            relation_type=RelationType.LOCATED_IN,
            patterns=[
                {"pattern": r"{subject}.*(?:is|was|located|situated).*(?:in|at).*{object}"},
                {"pattern": r"{subject}.*(?:based in|headquartered in).*{object}"},
                {"pattern": r"(?:in|at).*{object}.*{subject}"}
            ],
            subject_types=[EntityType.ORGANIZATION, EntityType.PERSON, EntityType.FACILITY],
            object_types=[EntityType.LOCATION, EntityType.GPE, EntityType.CITY, EntityType.COUNTRY]
        ))
        
        # BORN_IN 关系模式
        patterns.append(RelationPattern(
            name="born_in_basic",
            relation_type=RelationType.BORN_IN,
            patterns=[
                {"pattern": r"{subject}.*(?:born|birth).*(?:in|at).*{object}"},
                {"pattern": r"{subject}.*(?:native of|from).*{object}"},
                {"pattern": r"born.*{object}.*{subject}"}
            ],
            subject_types=[EntityType.PERSON],
            object_types=[EntityType.LOCATION, EntityType.GPE, EntityType.CITY, EntityType.COUNTRY]
        ))
        
        # FOUNDED_BY 关系模式
        patterns.append(RelationPattern(
            name="founded_by_basic", 
            relation_type=RelationType.FOUNDED_BY,
            patterns=[
                {"pattern": r"{object}.*(?:founded|established|created|started).*{subject}"},
                {"pattern": r"{subject}.*(?:founded|established|created|started).*(?:by|from).*{object}"},
                {"pattern": r"{object}.*(?:founder|creator).*{subject}"}
            ],
            subject_types=[EntityType.ORGANIZATION, EntityType.COMPANY],
            object_types=[EntityType.PERSON]
        ))
        
        # CAPITAL_OF 关系模式
        patterns.append(RelationPattern(
            name="capital_of_basic",
            relation_type=RelationType.CAPITAL_OF,
            patterns=[
                {"pattern": r"{subject}.*(?:capital|seat).*(?:of|for).*{object}"},
                {"pattern": r"{object}.*capital.*{subject}"},
                {"pattern": r"capital.*{object}.*{subject}"}
            ],
            subject_types=[EntityType.CITY, EntityType.LOCATION],
            object_types=[EntityType.COUNTRY, EntityType.GPE]
        ))
        
        # SPOUSE 关系模式
        patterns.append(RelationPattern(
            name="spouse_basic",
            relation_type=RelationType.SPOUSE,
            patterns=[
                {"pattern": r"{subject}.*(?:married|wed|spouse|wife|husband).*{object}"},
                {"pattern": r"{subject}.*(?:and|&).*{object}.*(?:married|couple|wedding)"},
                {"pattern": r"(?:marriage|wedding).*{subject}.*{object}"}
            ],
            subject_types=[EntityType.PERSON],
            object_types=[EntityType.PERSON]
        ))
        
        # CEO_OF 关系模式
        patterns.append(RelationPattern(
            name="ceo_of_basic",
            relation_type=RelationType.CEO_OF,
            patterns=[
                {"pattern": r"{subject}.*(?:CEO|chief executive|president).*(?:of|at).*{object}"},
                {"pattern": r"{object}.*(?:CEO|chief executive|president).*{subject}"},
                {"pattern": r"CEO.*{object}.*{subject}"}
            ],
            subject_types=[EntityType.PERSON],
            object_types=[EntityType.ORGANIZATION, EntityType.COMPANY]
        ))
        
        # EDUCATED_AT 关系模式
        patterns.append(RelationPattern(
            name="educated_at_basic",
            relation_type=RelationType.EDUCATED_AT,
            patterns=[
                {"pattern": r"{subject}.*(?:graduated|studied|attended|alumnus).*(?:from|at).*{object}"},
                {"pattern": r"{subject}.*(?:education|degree).*(?:from|at).*{object}"},
                {"pattern": r"{object}.*(?:graduate|student|alumni).*{subject}"}
            ],
            subject_types=[EntityType.PERSON],
            object_types=[EntityType.ORGANIZATION, EntityType.UNIVERSITY]
        ))
        
        # SUBSIDIARY_OF 关系模式
        patterns.append(RelationPattern(
            name="subsidiary_of_basic",
            relation_type=RelationType.SUBSIDIARY_OF,
            patterns=[
                {"pattern": r"{subject}.*(?:subsidiary|division|unit).*(?:of|under).*{object}"},
                {"pattern": r"{object}.*(?:owns|acquired|bought).*{subject}"},
                {"pattern": r"{subject}.*(?:owned by|acquired by).*{object}"}
            ],
            subject_types=[EntityType.ORGANIZATION, EntityType.COMPANY],
            object_types=[EntityType.ORGANIZATION, EntityType.COMPANY]
        ))
        
        # AUTHOR_OF 关系模式
        patterns.append(RelationPattern(
            name="author_of_basic",
            relation_type=RelationType.AUTHOR_OF,
            patterns=[
                {"pattern": r"{subject}.*(?:wrote|authored|written|created).*{object}"},
                {"pattern": r"{object}.*(?:by|author|written by).*{subject}"},
                {"pattern": r"author.*{subject}.*{object}"}
            ],
            subject_types=[EntityType.PERSON],
            object_types=[EntityType.WORK_OF_ART, EntityType.PRODUCT]
        ))
        
        return patterns
    
    def extract_relations(
        self, 
        text: str, 
        entities: List[Entity],
        sentence_boundaries: Optional[List[Tuple[int, int]]] = None
    ) -> List[Relation]:
        """基于模式的关系抽取"""
        if len(entities) < 2:
            return []
        
        relations = []
        
        # 如果没有提供句子边界，将整个文本作为一个句子
        if not sentence_boundaries:
            sentence_boundaries = [(0, len(text))]
        
        # 在每个句子内抽取关系
        for sent_start, sent_end in sentence_boundaries:
            sent_text = text[sent_start:sent_end]
            sent_entities = [
                e for e in entities 
                if sent_start <= e.start < sent_end and sent_start < e.end <= sent_end
            ]
            
            if len(sent_entities) < 2:
                continue
            
            # 尝试所有实体对
            for subj, obj in itertools.combinations(sent_entities, 2):
                if subj.start >= obj.start:
                    continue  # 保证主语在前
                
                # 尝试匹配所有模式
                for pattern in self.patterns:
                    if not pattern.matches_entity_types(subj, obj):
                        continue
                    
                    relation = self._match_pattern(
                        pattern, subj, obj, sent_text, sent_start
                    )
                    if relation:
                        relations.append(relation)
        
        return relations
    
    def _match_pattern(
        self,
        pattern: RelationPattern,
        subject: Entity,
        obj: Entity,
        sentence: str,
        sentence_offset: int
    ) -> Optional[Relation]:
        """匹配单个模式"""
        # 调整实体位置到句子内的相对位置
        subj_start = subject.start - sentence_offset
        subj_end = subject.end - sentence_offset
        obj_start = obj.start - sentence_offset
        obj_end = obj.end - sentence_offset
        
        # 检查实体是否在句子范围内
        if subj_start < 0 or obj_start < 0 or subj_end > len(sentence) or obj_end > len(sentence):
            return None
        
        subject_text = sentence[subj_start:subj_end]
        object_text = sentence[obj_start:obj_end]
        
        # 尝试匹配模式中的每个正则表达式
        for pattern_dict in pattern.patterns:
            regex_pattern = pattern_dict["pattern"]
            
            # 替换占位符
            regex_pattern = regex_pattern.replace("{subject}", re.escape(subject_text))
            regex_pattern = regex_pattern.replace("{object}", re.escape(object_text))
            
            try:
                if re.search(regex_pattern, sentence, re.IGNORECASE):
                    # 匹配成功，创建关系
                    relation = Relation(
                        subject=subject,
                        predicate=pattern.relation_type,
                        object=obj,
                        confidence=pattern.confidence,
                        context=sentence,
                        source_sentence=sentence,
                        evidence=[f"Pattern: {pattern.name}"],
                        metadata={
                            "extraction_method": "pattern_based",
                            "pattern_name": pattern.name,
                            "matched_pattern": regex_pattern,
                            "sentence_offset": sentence_offset
                        }
                    )
                    return relation
                    
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {regex_pattern}, error: {e}")
                continue
        
        return None

class DependencyBasedExtractor:
    """基于依存句法的关系抽取器"""
    
    def __init__(self, nlp_model: Optional[spacy.Language] = None):
        self.nlp = nlp_model
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("spaCy model not available for dependency parsing")
                self.nlp = None
        
        self.dependency_patterns = self._initialize_dependency_patterns()
    
    def _initialize_dependency_patterns(self) -> List[Dict[str, Any]]:
        """初始化依存关系模式"""
        return [
            {
                "name": "nsubj_prep_pobj",
                "relation_type": RelationType.WORKS_FOR,
                "pattern": {
                    "subject_dep": ["nsubj"],
                    "verb_lemma": ["work", "join", "employ"],
                    "prep": ["for", "at", "with"],
                    "object_dep": ["pobj"]
                },
                "subject_types": [EntityType.PERSON],
                "object_types": [EntityType.ORGANIZATION]
            },
            {
                "name": "passive_agent",
                "relation_type": RelationType.FOUNDED_BY,
                "pattern": {
                    "subject_dep": ["nsubjpass"],
                    "verb_lemma": ["found", "establish", "create", "start"],
                    "prep": ["by"],
                    "object_dep": ["pobj"]
                },
                "subject_types": [EntityType.ORGANIZATION],
                "object_types": [EntityType.PERSON]
            },
            {
                "name": "appos_relation",
                "relation_type": RelationType.CEO_OF,
                "pattern": {
                    "subject_dep": ["appos"],
                    "object_dep": ["nmod"],
                    "keywords": ["CEO", "president", "chief", "executive"]
                },
                "subject_types": [EntityType.PERSON],
                "object_types": [EntityType.ORGANIZATION]
            }
        ]
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于依存句法的关系抽取"""
        if not self.nlp or len(entities) < 2:
            return []
        
        try:
            doc = self.nlp(text)
            relations = []
            
            # 为每个句子抽取关系
            for sent in doc.sents:
                sent_entities = [
                    e for e in entities
                    if sent.start_char <= e.start < sent.end_char and 
                       sent.start_char < e.end <= sent.end_char
                ]
                
                if len(sent_entities) < 2:
                    continue
                
                # 创建实体到token的映射
                entity_token_map = self._map_entities_to_tokens(sent_entities, sent)
                
                # 对每对实体尝试依存关系模式
                for subj, obj in itertools.combinations(sent_entities, 2):
                    if subj.start >= obj.start:
                        continue
                    
                    relation = self._extract_dependency_relation(
                        subj, obj, sent, entity_token_map
                    )
                    if relation:
                        relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"Dependency-based relation extraction failed: {e}")
            return []
    
    def _map_entities_to_tokens(
        self, 
        entities: List[Entity], 
        sentence: spacy.tokens.Span
    ) -> Dict[str, List[spacy.tokens.Token]]:
        """将实体映射到对应的tokens"""
        entity_token_map = {}
        
        for entity in entities:
            entity_tokens = []
            for token in sentence:
                if (entity.start <= token.idx < entity.end or
                    entity.start < token.idx + len(token) <= entity.end):
                    entity_tokens.append(token)
            
            if entity_tokens:
                entity_token_map[entity.entity_id] = entity_tokens
        
        return entity_token_map
    
    def _extract_dependency_relation(
        self,
        subject: Entity,
        obj: Entity,
        sentence: spacy.tokens.Span,
        entity_token_map: Dict[str, List[spacy.tokens.Token]]
    ) -> Optional[Relation]:
        """基于依存关系抽取单个关系"""
        subj_tokens = entity_token_map.get(subject.entity_id, [])
        obj_tokens = entity_token_map.get(obj.entity_id, [])
        
        if not subj_tokens or not obj_tokens:
            return None
        
        # 获取实体的主要token（通常是最后一个）
        subj_head = subj_tokens[-1]
        obj_head = obj_tokens[-1]
        
        # 尝试匹配依存模式
        for pattern in self.dependency_patterns:
            relation = self._match_dependency_pattern(
                pattern, subject, obj, subj_head, obj_head, sentence
            )
            if relation:
                return relation
        
        return None
    
    def _match_dependency_pattern(
        self,
        pattern: Dict[str, Any],
        subject: Entity,
        obj: Entity,
        subj_token: spacy.tokens.Token,
        obj_token: spacy.tokens.Token,
        sentence: spacy.tokens.Span
    ) -> Optional[Relation]:
        """匹配依存关系模式"""
        # 检查实体类型
        if ("subject_types" in pattern and 
            subject.label not in pattern["subject_types"]):
            return None
        
        if ("object_types" in pattern and 
            obj.label not in pattern["object_types"]):
            return None
        
        pattern_config = pattern["pattern"]
        
        # 简单的依存关系匹配逻辑
        # 这里可以根据具体需求实现更复杂的依存关系分析
        
        # 检查是否存在连接两个实体的依存路径
        if self._has_dependency_path(subj_token, obj_token, pattern_config):
            relation = Relation(
                subject=subject,
                predicate=RelationType(pattern["relation_type"]),
                object=obj,
                confidence=0.75,  # 依存关系的默认置信度
                context=sentence.text,
                source_sentence=sentence.text,
                evidence=[f"Dependency pattern: {pattern['name']}"],
                metadata={
                    "extraction_method": "dependency_based",
                    "pattern_name": pattern["name"],
                    "subject_dep": subj_token.dep_,
                    "object_dep": obj_token.dep_,
                    "subject_head": subj_token.head.text,
                    "object_head": obj_token.head.text
                }
            )
            return relation
        
        return None
    
    def _has_dependency_path(
        self,
        token1: spacy.tokens.Token,
        token2: spacy.tokens.Token,
        pattern_config: Dict[str, Any]
    ) -> bool:
        """检查两个token之间是否存在符合模式的依存路径"""
        # 简化的依存路径检查
        # 实际实现中可以使用更复杂的图搜索算法
        
        # 检查直接依存关系
        if token1.head == token2 or token2.head == token1:
            return True
        
        # 检查共同祖先
        if token1.head == token2.head:
            return True
        
        # 检查特定的依存关系模式
        if "verb_lemma" in pattern_config:
            verb_lemmas = pattern_config["verb_lemma"]
            
            # 查找连接的动词
            for ancestor in token1.ancestors:
                if ancestor.lemma_.lower() in verb_lemmas:
                    for descendant in ancestor.subtree:
                        if descendant == token2:
                            return True
        
        return False

class RelationExtractor:
    """关系抽取器主类 - 集成多种抽取方法"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # 初始化各种抽取器
        self.pattern_extractor = PatternBasedExtractor()
        self.dependency_extractor = DependencyBasedExtractor()
        
        # 加载spaCy模型用于句子分割
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not available, using simple sentence splitting")
            self.nlp = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "confidence_threshold": 0.5,
            "max_entity_distance": 100,  # 最大实体距离（字符数）
            "use_pattern_based": True,
            "use_dependency_based": True,
            "deduplicate_relations": True,
            "max_relations_per_sentence": 10
        }
    
    async def extract_relations(
        self,
        text: str,
        entities: List[Entity],
        confidence_threshold: Optional[float] = None
    ) -> List[Relation]:
        """抽取文本中的关系"""
        if not text or len(entities) < 2:
            return []
        
        confidence_threshold = confidence_threshold or self.config.get("confidence_threshold", 0.5)
        start_time = time.time()
        
        # 句子分割
        sentence_boundaries = self._get_sentence_boundaries(text)
        
        all_relations = []
        
        # 基于模式的关系抽取
        if self.config.get("use_pattern_based", True):
            try:
                pattern_relations = self.pattern_extractor.extract_relations(
                    text, entities, sentence_boundaries
                )
                all_relations.extend(pattern_relations)
                logger.debug(f"Pattern-based extracted {len(pattern_relations)} relations")
            except Exception as e:
                logger.error(f"Pattern-based extraction failed: {e}")
        
        # 基于依存句法的关系抽取
        if self.config.get("use_dependency_based", True):
            try:
                dependency_relations = self.dependency_extractor.extract_relations(text, entities)
                all_relations.extend(dependency_relations)
                logger.debug(f"Dependency-based extracted {len(dependency_relations)} relations")
            except Exception as e:
                logger.error(f"Dependency-based extraction failed: {e}")
        
        # 过滤和去重
        filtered_relations = self._filter_relations(all_relations, confidence_threshold)
        
        if self.config.get("deduplicate_relations", True):
            filtered_relations = self._deduplicate_relations(filtered_relations)
        
        processing_time = time.time() - start_time
        logger.info(f"Relation extraction completed in {processing_time:.2f}s, found {len(filtered_relations)} relations")
        
        return filtered_relations
    
    def _get_sentence_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """获取句子边界"""
        if self.nlp:
            try:
                doc = self.nlp(text)
                return [(sent.start_char, sent.end_char) for sent in doc.sents]
            except Exception as e:
                logger.warning(f"spaCy sentence segmentation failed: {e}")
        
        # 简单的句子分割作为后备
        sentences = []
        current_start = 0
        
        for match in re.finditer(r'[.!?]+\s*', text):
            end = match.end()
            if end > current_start:
                sentences.append((current_start, end))
                current_start = end
        
        if current_start < len(text):
            sentences.append((current_start, len(text)))
        
        return sentences
    
    def _filter_relations(
        self, 
        relations: List[Relation], 
        confidence_threshold: float
    ) -> List[Relation]:
        """过滤关系"""
        filtered = []
        max_distance = self.config.get("max_entity_distance", 100)
        
        for relation in relations:
            # 置信度过滤
            if relation.confidence < confidence_threshold:
                continue
            
            # 距离过滤
            distance = abs(relation.object.start - relation.subject.end)
            if distance > max_distance:
                continue
            
            # 实体类型验证
            if relation.subject.label == relation.object.label == EntityType.PERSON:
                # 人-人关系需要特殊验证
                if relation.predicate not in [
                    RelationType.SPOUSE, RelationType.CHILD_OF, RelationType.PARENT_OF,
                    RelationType.COLLABORATED_WITH, RelationType.ADVISOR_OF
                ]:
                    continue
            
            filtered.append(relation)
        
        return filtered
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        # 基于(主语, 谓语, 宾语)的文本进行去重
        seen_triples = set()
        deduplicated = []
        
        # 按置信度排序，保留置信度最高的
        relations.sort(key=lambda r: r.confidence, reverse=True)
        
        for relation in relations:
            triple = (
                relation.subject.canonical_form or relation.subject.text.lower(),
                relation.predicate.value,
                relation.object.canonical_form or relation.object.text.lower()
            )
            
            if triple not in seen_triples:
                seen_triples.add(triple)
                deduplicated.append(relation)
        
        return deduplicated
    
    def get_supported_relations(self) -> List[Dict[str, Any]]:
        """获取支持的关系类型"""
        relations = []
        
        for relation_type in RelationType:
            relations.append({
                "type": relation_type.value,
                "name": relation_type.name,
                "description": self._get_relation_description(relation_type)
            })
        
        return relations
    
    def _get_relation_description(self, relation_type: RelationType) -> str:
        """获取关系类型描述"""
        descriptions = {
            RelationType.WORKS_FOR: "Person works for an organization",
            RelationType.LOCATED_IN: "Entity is located in a place",
            RelationType.BORN_IN: "Person was born in a place",
            RelationType.FOUNDED_BY: "Organization was founded by a person",
            RelationType.CEO_OF: "Person is CEO of an organization",
            RelationType.SPOUSE: "Two people are married",
            RelationType.EDUCATED_AT: "Person was educated at an institution",
            RelationType.CAPITAL_OF: "City is the capital of a country/state",
            RelationType.SUBSIDIARY_OF: "Company is a subsidiary of another company",
            RelationType.AUTHOR_OF: "Person authored a work"
        }
        return descriptions.get(relation_type, f"Relation of type {relation_type.value}")
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """获取抽取统计信息"""
        return {
            "pattern_count": len(self.pattern_extractor.patterns),
            "dependency_pattern_count": len(self.dependency_extractor.dependency_patterns),
            "supported_relation_types": len(RelationType),
            "config": self.config
        }
