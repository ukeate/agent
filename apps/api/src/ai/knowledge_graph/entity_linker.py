"""
实体链接器

实现实体规范化、去重、消歧和外部知识库链接
支持Wikidata/DBpedia链接，实体链接准确率≥80%
"""

import asyncio
import aiohttp
import logging
import time
import hashlib
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import quote
import difflib
import re
from collections import defaultdict

from .data_models import Entity, EntityType


logger = logging.getLogger(__name__)


@dataclass
class LinkedEntity:
    """链接后的实体"""
    original_entity: Entity
    canonical_form: str
    wikidata_id: Optional[str] = None
    dbpedia_uri: Optional[str] = None
    confidence: float = 0.0
    aliases: List[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.metadata is None:
            self.metadata = {}


class WikidataAPI:
    """Wikidata API 客户端"""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self.base_url = "https://www.wikidata.org/w/api.php"
        self.entity_url = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
        self.cache = {}  # 简单的内存缓存
    
    async def search_entities(
        self, 
        query: str, 
        entity_type: Optional[EntityType] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索Wikidata实体"""
        cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}_{entity_type}_{limit}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": query,
            "limit": limit,
            "type": "item"
        }
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._make_request(session, params, cache_key)
            else:
                return await self._make_request(self.session, params, cache_key)
                
        except Exception as e:
            logger.error(f"Wikidata search failed for '{query}': {e}")
            return []
    
    async def _make_request(
        self, 
        session: aiohttp.ClientSession, 
        params: Dict[str, Any], 
        cache_key: str
    ) -> List[Dict[str, Any]]:
        """执行API请求"""
        async with session.get(self.base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                entities = data.get("search", [])
                
                # 过滤和处理结果
                processed_entities = []
                for entity in entities:
                    processed_entity = {
                        "id": entity.get("id"),
                        "label": entity.get("label", ""),
                        "description": entity.get("description", ""),
                        "aliases": entity.get("aliases", []),
                        "uri": f"http://www.wikidata.org/entity/{entity.get('id')}",
                        "score": self._calculate_relevance_score(entity)
                    }
                    processed_entities.append(processed_entity)
                
                # 按相关性排序
                processed_entities.sort(key=lambda x: x["score"], reverse=True)
                
                # 缓存结果
                self.cache[cache_key] = processed_entities
                return processed_entities
            else:
                logger.error(f"Wikidata API error: {response.status}")
                return []
    
    def _calculate_relevance_score(self, entity: Dict[str, Any]) -> float:
        """计算实体相关性分数"""
        score = 0.0
        
        # 基础分数
        if entity.get("label"):
            score += 0.5
        
        if entity.get("description"):
            score += 0.3
        
        # 别名数量加分
        aliases_count = len(entity.get("aliases", []))
        score += min(aliases_count * 0.1, 0.2)
        
        return score
    
    async def get_entity_details(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取实体详细信息"""
        cache_key = f"details_{entity_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = self.entity_url.format(entity_id)
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            entity_data = data.get("entities", {}).get(entity_id, {})
                            self.cache[cache_key] = entity_data
                            return entity_data
            else:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        entity_data = data.get("entities", {}).get(entity_id, {})
                        self.cache[cache_key] = entity_data
                        return entity_data
                        
        except Exception as e:
            logger.error(f"Failed to get Wikidata entity details for {entity_id}: {e}")
        
        return None


class DBpediaAPI:
    """DBpedia API 客户端"""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self.lookup_url = "http://lookup.dbpedia.org/api/search.asmx/PrefixSearch"
        self.resource_url = "http://dbpedia.org/resource/{}"
        self.cache = {}
    
    async def search_entities(
        self, 
        query: str, 
        entity_type: Optional[EntityType] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索DBpedia实体"""
        cache_key = f"dbpedia_search_{hashlib.md5(query.encode()).hexdigest()}_{entity_type}_{limit}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        params = {
            "QueryString": query,
            "MaxHits": limit,
            "QueryClass": self._map_entity_type_to_dbpedia_class(entity_type)
        }
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._make_request(session, params, cache_key)
            else:
                return await self._make_request(self.session, params, cache_key)
                
        except Exception as e:
            logger.error(f"DBpedia search failed for '{query}': {e}")
            return []
    
    async def _make_request(
        self, 
        session: aiohttp.ClientSession, 
        params: Dict[str, Any], 
        cache_key: str
    ) -> List[Dict[str, Any]]:
        """执行DBpedia API请求"""
        async with session.get(self.lookup_url, params=params) as response:
            if response.status == 200:
                # DBpedia返回XML，这里简化处理
                text = await response.text()
                entities = self._parse_dbpedia_xml(text)
                self.cache[cache_key] = entities
                return entities
            else:
                logger.error(f"DBpedia API error: {response.status}")
                return []
    
    def _parse_dbpedia_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """解析DBpedia XML响应（简化实现）"""
        # 这里使用正则表达式简单解析XML
        # 生产环境中应该使用正确的XML解析器
        entities = []
        
        # 提取URI和标签
        uri_pattern = r'<URI>(.*?)</URI>'
        label_pattern = r'<Label>(.*?)</Label>'
        description_pattern = r'<Description>(.*?)</Description>'
        
        uris = re.findall(uri_pattern, xml_text)
        labels = re.findall(label_pattern, xml_text)
        descriptions = re.findall(description_pattern, xml_text)
        
        for i, uri in enumerate(uris):
            entity = {
                "uri": uri,
                "label": labels[i] if i < len(labels) else "",
                "description": descriptions[i] if i < len(descriptions) else "",
                "score": 1.0 / (i + 1)  # 简单的排序分数
            }
            entities.append(entity)
        
        return entities
    
    def _map_entity_type_to_dbpedia_class(self, entity_type: Optional[EntityType]) -> str:
        """将实体类型映射到DBpedia类"""
        if not entity_type:
            return ""
        
        mapping = {
            EntityType.PERSON: "Person",
            EntityType.ORGANIZATION: "Organisation",
            EntityType.LOCATION: "Place",
            EntityType.GPE: "Place",
            EntityType.COUNTRY: "Country",
            EntityType.CITY: "City"
        }
        
        return mapping.get(entity_type, "")


class EntityNormalizer:
    """实体规范化器"""
    
    def __init__(self):
        self.normalization_rules = self._load_normalization_rules()
        self.common_aliases = self._load_common_aliases()
    
    def _load_normalization_rules(self) -> Dict[str, Any]:
        """加载规范化规则"""
        return {
            "remove_patterns": [
                r"\s+",  # 多个空格合并为一个
                r"[^\w\s]",  # 移除特殊字符（保留字母数字和空格）
            ],
            "replace_patterns": [
                (r"&", "and"),
                (r"\bco\b", "company"),
                (r"\bcorp\b", "corporation"),
                (r"\binc\b", "incorporated"),
                (r"\bltd\b", "limited")
            ],
            "case_rules": {
                "person": "title",  # 人名首字母大写
                "organization": "title",  # 组织名首字母大写
                "location": "title"  # 地名首字母大写
            }
        }
    
    def _load_common_aliases(self) -> Dict[str, List[str]]:
        """加载常见别名映射"""
        return {
            "united states": ["usa", "us", "america", "united states of america"],
            "united kingdom": ["uk", "britain", "great britain"],
            "new york": ["nyc", "new york city"],
            "microsoft": ["microsoft corp", "microsoft corporation", "msft"],
            "apple": ["apple inc", "apple computer"],
            "google": ["alphabet", "alphabet inc"]
        }
    
    def normalize_entity(self, entity: Entity) -> str:
        """规范化实体文本"""
        text = entity.text.lower().strip()
        
        # 应用移除模式
        for pattern in self.normalization_rules["remove_patterns"]:
            text = re.sub(pattern, " ", text)
        
        # 应用替换模式
        for old_pattern, new_text in self.normalization_rules["replace_patterns"]:
            text = re.sub(old_pattern, new_text, text, flags=re.IGNORECASE)
        
        # 清理多余空格
        text = re.sub(r"\s+", " ", text).strip()
        
        # 应用大小写规则
        entity_type_str = entity.label.name.lower()
        case_rule = self.normalization_rules["case_rules"].get(entity_type_str)
        
        if case_rule == "title":
            text = text.title()
        elif case_rule == "upper":
            text = text.upper()
        elif case_rule == "lower":
            text = text.lower()
        
        return text
    
    def find_aliases(self, canonical_form: str) -> List[str]:
        """查找实体的别名"""
        canonical_lower = canonical_form.lower()
        
        # 直接查找
        if canonical_lower in self.common_aliases:
            return self.common_aliases[canonical_lower]
        
        # 模糊匹配
        aliases = []
        for key, values in self.common_aliases.items():
            if difflib.SequenceMatcher(None, canonical_lower, key).ratio() > 0.8:
                aliases.extend(values)
        
        return aliases


class EntityDeduplicator:
    """实体去重器"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体列表"""
        if len(entities) <= 1:
            return entities
        
        # 按实体类型分组
        type_groups = defaultdict(list)
        for entity in entities:
            type_groups[entity.label].append(entity)
        
        deduplicated = []
        
        for entity_type, type_entities in type_groups.items():
            deduplicated.extend(self._deduplicate_same_type_entities(type_entities))
        
        return deduplicated
    
    def _deduplicate_same_type_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重同类型实体"""
        if len(entities) <= 1:
            return entities
        
        # 按置信度排序
        entities.sort(key=lambda e: e.confidence, reverse=True)
        
        deduplicated = []
        processed_canonical_forms = set()
        
        for entity in entities:
            canonical_form = entity.canonical_form or entity.text.lower().strip()
            
            # 检查是否与已处理的实体相似
            is_duplicate = False
            for processed_form in processed_canonical_forms:
                similarity = difflib.SequenceMatcher(
                    None, canonical_form, processed_form
                ).ratio()
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(entity)
                processed_canonical_forms.add(canonical_form)
        
        return deduplicated
    
    def calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """计算两个实体的相似度"""
        if entity1.label != entity2.label:
            return 0.0
        
        text1 = entity1.canonical_form or entity1.text
        text2 = entity2.canonical_form or entity2.text
        
        # 文本相似度
        text_similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # 位置相似度（如果在同一文档中）
        position_similarity = 0.0
        if hasattr(entity1, 'start') and hasattr(entity2, 'start'):
            distance = abs(entity1.start - entity2.start)
            position_similarity = max(0, 1 - distance / 1000)  # 假设1000字符为最大距离
        
        # 综合相似度
        return 0.7 * text_similarity + 0.3 * position_similarity


class EntityLinker:
    """实体链接器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # 初始化组件
        self.normalizer = EntityNormalizer()
        self.deduplicator = EntityDeduplicator(
            similarity_threshold=self.config.get("similarity_threshold", 0.85)
        )
        
        # 初始化API客户端
        self.session = None
        self.wikidata_api = None
        self.dbpedia_api = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "similarity_threshold": 0.85,
            "linking_confidence_threshold": 0.7,
            "max_candidates": 5,
            "use_wikidata": True,
            "use_dbpedia": True,
            "cache_enabled": True,
            "timeout_seconds": 10
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.get("timeout_seconds", 10))
        )
        self.wikidata_api = WikidataAPI(self.session)
        self.dbpedia_api = DBpediaAPI(self.session)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def link_entities(self, entities: List[Entity]) -> List[LinkedEntity]:
        """链接实体到外部知识库"""
        if not entities:
            return []
        
        start_time = time.time()
        
        # 预处理：规范化和去重
        normalized_entities = await self._preprocess_entities(entities)
        
        # 并发链接所有实体
        tasks = []
        for entity in normalized_entities:
            task = self._link_single_entity(entity)
            tasks.append(task)
        
        linked_entities = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        valid_linked_entities = []
        for i, result in enumerate(linked_entities):
            if isinstance(result, Exception):
                logger.error(f"Failed to link entity {normalized_entities[i].text}: {result}")
                # 创建未链接的实体
                linked_entity = LinkedEntity(
                    original_entity=normalized_entities[i],
                    canonical_form=self.normalizer.normalize_entity(normalized_entities[i]),
                    confidence=0.0
                )
                valid_linked_entities.append(linked_entity)
            else:
                valid_linked_entities.append(result)
        
        processing_time = time.time() - start_time
        linked_count = sum(1 for le in valid_linked_entities if le.wikidata_id or le.dbpedia_uri)
        
        logger.info(
            f"Entity linking completed in {processing_time:.2f}s: "
            f"{linked_count}/{len(valid_linked_entities)} entities linked"
        )
        
        return valid_linked_entities
    
    async def _preprocess_entities(self, entities: List[Entity]) -> List[Entity]:
        """预处理实体：规范化和去重"""
        # 规范化实体
        for entity in entities:
            if not entity.canonical_form:
                entity.canonical_form = self.normalizer.normalize_entity(entity)
        
        # 去重
        deduplicated_entities = self.deduplicator.deduplicate_entities(entities)
        
        logger.debug(f"Preprocessed {len(entities)} -> {len(deduplicated_entities)} entities")
        return deduplicated_entities
    
    async def _link_single_entity(self, entity: Entity) -> LinkedEntity:
        """链接单个实体"""
        canonical_form = entity.canonical_form or self.normalizer.normalize_entity(entity)
        
        # 搜索候选实体
        candidates = await self._search_candidates(entity, canonical_form)
        
        # 选择最佳候选
        best_candidate = self._select_best_candidate(entity, candidates)
        
        # 创建链接结果
        linked_entity = LinkedEntity(
            original_entity=entity,
            canonical_form=canonical_form,
            aliases=self.normalizer.find_aliases(canonical_form)
        )
        
        if best_candidate:
            linked_entity.confidence = best_candidate["confidence"]
            
            if "wikidata_id" in best_candidate:
                linked_entity.wikidata_id = best_candidate["wikidata_id"]
                linked_entity.description = best_candidate.get("description")
                linked_entity.metadata["wikidata"] = best_candidate
            
            if "dbpedia_uri" in best_candidate:
                linked_entity.dbpedia_uri = best_candidate["dbpedia_uri"]
                if not linked_entity.description:
                    linked_entity.description = best_candidate.get("description")
                linked_entity.metadata["dbpedia"] = best_candidate
        
        return linked_entity
    
    async def _search_candidates(
        self, 
        entity: Entity, 
        canonical_form: str
    ) -> List[Dict[str, Any]]:
        """搜索候选实体"""
        candidates = []
        max_candidates = self.config.get("max_candidates", 5)
        
        # 并发搜索Wikidata和DBpedia
        tasks = []
        
        if self.config.get("use_wikidata", True) and self.wikidata_api:
            tasks.append(
                self._search_wikidata_candidates(entity, canonical_form, max_candidates)
            )
        
        if self.config.get("use_dbpedia", True) and self.dbpedia_api:
            tasks.append(
                self._search_dbpedia_candidates(entity, canonical_form, max_candidates)
            )
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Candidate search failed: {result}")
                else:
                    candidates.extend(result)
        
        return candidates
    
    async def _search_wikidata_candidates(
        self, 
        entity: Entity, 
        canonical_form: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """搜索Wikidata候选实体"""
        try:
            wikidata_entities = await self.wikidata_api.search_entities(
                canonical_form, entity.label, limit
            )
            
            candidates = []
            for wd_entity in wikidata_entities:
                candidate = {
                    "source": "wikidata",
                    "wikidata_id": wd_entity["id"],
                    "label": wd_entity["label"],
                    "description": wd_entity["description"],
                    "uri": wd_entity["uri"],
                    "confidence": self._calculate_linking_confidence(entity, wd_entity)
                }
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Wikidata search failed for '{canonical_form}': {e}")
            return []
    
    async def _search_dbpedia_candidates(
        self, 
        entity: Entity, 
        canonical_form: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """搜索DBpedia候选实体"""
        try:
            dbpedia_entities = await self.dbpedia_api.search_entities(
                canonical_form, entity.label, limit
            )
            
            candidates = []
            for db_entity in dbpedia_entities:
                candidate = {
                    "source": "dbpedia",
                    "dbpedia_uri": db_entity["uri"],
                    "label": db_entity["label"],
                    "description": db_entity["description"],
                    "confidence": self._calculate_linking_confidence(entity, db_entity)
                }
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"DBpedia search failed for '{canonical_form}': {e}")
            return []
    
    def _calculate_linking_confidence(
        self, 
        entity: Entity, 
        candidate: Dict[str, Any]
    ) -> float:
        """计算链接置信度"""
        confidence = 0.0
        
        entity_text = entity.canonical_form or entity.text
        candidate_label = candidate.get("label", "")
        
        # 文本相似度
        text_similarity = difflib.SequenceMatcher(
            None, 
            entity_text.lower(), 
            candidate_label.lower()
        ).ratio()
        
        confidence += 0.6 * text_similarity
        
        # 描述相关性
        description = candidate.get("description", "")
        if description:
            # 简单的关键词匹配
            entity_type_keywords = self._get_entity_type_keywords(entity.label)
            description_lower = description.lower()
            keyword_matches = sum(
                1 for keyword in entity_type_keywords 
                if keyword in description_lower
            )
            
            if entity_type_keywords:
                description_score = keyword_matches / len(entity_type_keywords)
                confidence += 0.3 * description_score
        
        # 来源权重
        source_weight = candidate.get("score", 0.0)
        confidence += 0.1 * source_weight
        
        return min(confidence, 1.0)
    
    def _get_entity_type_keywords(self, entity_type: EntityType) -> List[str]:
        """获取实体类型关键词"""
        keywords_map = {
            EntityType.PERSON: ["person", "human", "individual", "people"],
            EntityType.ORGANIZATION: ["organization", "company", "corporation", "business"],
            EntityType.LOCATION: ["place", "location", "area", "region"],
            EntityType.GPE: ["country", "state", "city", "nation", "government"],
            EntityType.COUNTRY: ["country", "nation", "state"],
            EntityType.CITY: ["city", "town", "municipality", "urban"]
        }
        
        return keywords_map.get(entity_type, [])
    
    def _select_best_candidate(
        self, 
        entity: Entity, 
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """选择最佳候选实体"""
        if not candidates:
            return None
        
        # 过滤低置信度候选
        threshold = self.config.get("linking_confidence_threshold", 0.7)
        filtered_candidates = [
            c for c in candidates if c["confidence"] >= threshold
        ]
        
        if not filtered_candidates:
            return None
        
        # 按置信度排序
        filtered_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return filtered_candidates[0]
    
    def get_linking_statistics(self) -> Dict[str, Any]:
        """获取链接统计信息"""
        return {
            "config": self.config,
            "similarity_threshold": self.deduplicator.similarity_threshold,
            "supported_sources": ["wikidata", "dbpedia"],
            "cache_enabled": self.config.get("cache_enabled", True)
        }