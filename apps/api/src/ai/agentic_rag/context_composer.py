"""
上下文相关的知识片段选择和组合

实现智能知识片段选择和组合功能，包括：
- 知识片段相关性评分和排序
- 片段去重和多样性控制机制
- 上下文长度优化和信息密度平衡
- 知识片段间逻辑关系分析
"""

import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import re
import json
from difflib import SequenceMatcher
from src.ai.rag.embeddings import embedding_service
from .query_analyzer import QueryAnalysis, QueryIntent
from .result_validator import ValidationResult, QualityScore

from src.core.logging import get_logger
logger = get_logger(__name__)

class FragmentType(str, Enum):
    """片段类型"""
    DEFINITION = "definition"        # 定义
    EXAMPLE = "example"              # 示例
    PROCEDURE = "procedure"          # 步骤
    CODE = "code"                    # 代码
    EXPLANATION = "explanation"      # 解释
    REFERENCE = "reference"          # 参考
    CONTEXT = "context"              # 上下文

class RelationshipType(str, Enum):
    """片段关系类型"""
    DEPENDENCY = "dependency"        # 依赖关系
    SEQUENCE = "sequence"           # 顺序关系
    SIMILARITY = "similarity"       # 相似关系
    COMPLEMENT = "complement"       # 互补关系
    CONTRAST = "contrast"           # 对比关系
    HIERARCHY = "hierarchy"         # 层次关系

class CompositionStrategy(str, Enum):
    """上下文组合策略"""
    BALANCED = "balanced"           # 平衡策略：相关性和多样性并重
    RELEVANCE_FIRST = "relevance"   # 相关性优先：优先选择最相关的片段
    DIVERSITY_FIRST = "diversity"   # 多样性优先：优先保证信息多样性
    HIERARCHICAL = "hierarchical"   # 层次化：按照逻辑层次组织
    SEQUENTIAL = "sequential"       # 顺序化：按照逻辑顺序组织

@dataclass
class KnowledgeFragment:
    """知识片段数据结构"""
    id: str
    content: str
    source: str
    fragment_type: FragmentType
    relevance_score: float
    quality_score: float
    information_density: float
    tokens: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FragmentRelationship:
    """片段关系数据结构"""
    fragment_a: str
    fragment_b: str
    relationship_type: RelationshipType
    strength: float
    explanation: str

@dataclass
class ComposedContext:
    """组合上下文结果"""
    query_id: str
    selected_fragments: List[KnowledgeFragment]
    total_tokens: int
    information_density: float
    diversity_score: float
    coherence_score: float
    relationships: List[FragmentRelationship]
    composition_strategy: str
    optimization_metrics: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ContextComposer:
    """上下文组合器"""
    
    def __init__(self):
        self.max_context_tokens = 4000  # 最大上下文长度
        self.min_fragment_tokens = 20   # 最小片段长度
        self.diversity_threshold = 0.3  # 多样性阈值
        self.coherence_weight = 0.4     # 连贯性权重
        self.relevance_weight = 0.4     # 相关性权重
        self.diversity_weight = 0.2     # 多样性权重
        
        # 片段类型权重
        self.fragment_type_weights = {
            FragmentType.DEFINITION: 1.2,
            FragmentType.EXAMPLE: 1.0,
            FragmentType.PROCEDURE: 1.1,
            FragmentType.CODE: 1.1,
            FragmentType.EXPLANATION: 1.0,
            FragmentType.REFERENCE: 0.8,
            FragmentType.CONTEXT: 0.9
        }
    
    async def compose_context(self,
                            query_analysis: QueryAnalysis,
                            validated_results: ValidationResult,
                            max_tokens: Optional[int] = None,
                            composition_strategy: str = "balanced") -> ComposedContext:
        """组合上下文"""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        # 1. 从验证结果中提取片段
        fragments = await self._extract_fragments(validated_results)
        
        # 2. 评估片段相关性和质量
        scored_fragments = await self._score_fragments(query_analysis, fragments)
        
        # 3. 分析片段关系
        relationships = await self._analyze_relationships(scored_fragments)
        
        # 4. 执行组合策略
        if composition_strategy == "balanced":
            selected_fragments = await self._balanced_composition(
                query_analysis, scored_fragments, relationships, max_tokens
            )
        elif composition_strategy == "relevance_first":
            selected_fragments = await self._relevance_first_composition(
                scored_fragments, max_tokens
            )
        elif composition_strategy == "diversity_first":
            selected_fragments = await self._diversity_first_composition(
                scored_fragments, relationships, max_tokens
            )
        else:
            selected_fragments = await self._balanced_composition(
                query_analysis, scored_fragments, relationships, max_tokens
            )
        
        # 5. 优化片段顺序
        optimized_fragments = await self._optimize_fragment_order(
            selected_fragments, relationships
        )
        
        # 6. 计算组合指标
        metrics = self._calculate_composition_metrics(
            optimized_fragments, relationships
        )
        
        return ComposedContext(
            query_id=getattr(validated_results, 'query_id', 'unknown'),
            selected_fragments=optimized_fragments,
            total_tokens=sum(f.tokens for f in optimized_fragments),
            information_density=metrics["information_density"],
            diversity_score=metrics["diversity_score"],
            coherence_score=metrics["coherence_score"],
            relationships=relationships,
            composition_strategy=composition_strategy,
            optimization_metrics=metrics,
            metadata={
                "original_fragments": len(fragments),
                "selected_fragments": len(optimized_fragments),
                "token_utilization": sum(f.tokens for f in optimized_fragments) / max_tokens
            }
        )
    
    async def _extract_fragments(self, validated_results: ValidationResult) -> List[KnowledgeFragment]:
        """从验证结果中提取知识片段"""
        fragments = []
        
        for retrieval_result in validated_results.retrieval_results:
            for idx, result_item in enumerate(retrieval_result.results):
                # 提取基本信息
                content = result_item.get("content", "")
                if len(content.strip()) < self.min_fragment_tokens:
                    continue
                
                # 估算token数量
                tokens = self._estimate_tokens(content)
                
                # 确定片段类型
                fragment_type = self._classify_fragment_type(content, result_item)
                
                # 计算信息密度
                info_density = self._calculate_information_density(content)
                
                fragment = KnowledgeFragment(
                    id=f"{retrieval_result.agent_type.value}_{idx}_{result_item.get('id', idx)}",
                    content=content,
                    source=result_item.get("file_path", "unknown"),
                    fragment_type=fragment_type,
                    relevance_score=result_item.get("score", 0.0),
                    quality_score=result_item.get("fused_score", result_item.get("score", 0.0)),
                    information_density=info_density,
                    tokens=tokens,
                    metadata={
                        "agent_type": retrieval_result.agent_type.value,
                        "file_type": result_item.get("file_type", ""),
                        "chunk_index": result_item.get("chunk_index", 0),
                        "original_item": result_item
                    }
                )
                fragments.append(fragment)
        
        return fragments
    
    async def _score_fragments(self, query_analysis: QueryAnalysis, fragments: List[KnowledgeFragment]) -> List[KnowledgeFragment]:
        """评估片段相关性和质量"""
        
        for fragment in fragments:
            # 基础得分来自检索结果
            base_score = fragment.relevance_score
            
            # 1. 基于查询意图调整分数
            intent_bonus = self._calculate_intent_bonus(query_analysis, fragment)
            
            # 2. 基于片段类型调整分数
            type_weight = self.fragment_type_weights.get(fragment.fragment_type, 1.0)
            
            # 3. 基于信息密度调整分数
            density_bonus = min(fragment.information_density, 0.3)  # 最多0.3的加成
            
            # 4. 基于内容质量调整分数
            quality_bonus = await self._calculate_content_quality_bonus(fragment)
            
            # 5. 基于关键词匹配调整分数
            keyword_bonus = self._calculate_keyword_match_bonus(query_analysis, fragment)
            
            # 综合计算最终相关性分数
            final_score = (
                base_score * type_weight * 0.6 +
                intent_bonus * 0.15 +
                density_bonus * 0.1 +
                quality_bonus * 0.1 +
                keyword_bonus * 0.05
            )
            
            fragment.relevance_score = min(final_score, 1.0)
        
        # 按相关性分数排序
        fragments.sort(key=lambda f: f.relevance_score, reverse=True)
        
        return fragments
    
    async def _analyze_relationships(self, fragments: List[KnowledgeFragment]) -> List[FragmentRelationship]:
        """分析片段间关系"""
        relationships = []
        
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                frag_a, frag_b = fragments[i], fragments[j]
                
                # 计算相似度
                similarity = self._calculate_content_similarity(frag_a.content, frag_b.content)
                
                # 检测依赖关系
                dependency_strength = self._detect_dependency(frag_a, frag_b)
                
                # 检测顺序关系
                sequence_strength = self._detect_sequence(frag_a, frag_b)
                
                # 检测层次关系
                hierarchy_strength = self._detect_hierarchy(frag_a, frag_b)
                
                # 检测对比关系
                contrast_strength = self._detect_contrast(frag_a, frag_b)
                
                # 确定主要关系类型
                relationship_scores = {
                    RelationshipType.SIMILARITY: similarity,
                    RelationshipType.DEPENDENCY: dependency_strength,
                    RelationshipType.SEQUENCE: sequence_strength,
                    RelationshipType.HIERARCHY: hierarchy_strength,
                    RelationshipType.CONTRAST: contrast_strength
                }
                
                # 只记录强关系
                max_relation = max(relationship_scores.items(), key=lambda x: x[1])
                if max_relation[1] > 0.3:
                    relationship = FragmentRelationship(
                        fragment_a=frag_a.id,
                        fragment_b=frag_b.id,
                        relationship_type=max_relation[0],
                        strength=max_relation[1],
                        explanation=self._generate_relationship_explanation(
                            max_relation[0], max_relation[1], frag_a, frag_b
                        )
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _balanced_composition(self,
                                  query_analysis: QueryAnalysis,
                                  fragments: List[KnowledgeFragment],
                                  relationships: List[FragmentRelationship],
                                  max_tokens: int) -> List[KnowledgeFragment]:
        """平衡组合策略"""
        selected = []
        used_tokens = 0
        selected_ids = set()
        
        # 按相关性排序的候选列表
        candidates = fragments.copy()
        
        while candidates and used_tokens < max_tokens:
            # 1. 选择最高相关性的片段
            if not selected:
                best_fragment = candidates[0]
            else:
                # 2. 在剩余片段中选择最佳组合片段
                best_fragment = self._select_best_complement(
                    candidates, selected, relationships, query_analysis
                )
            
            # 检查是否还有空间
            if used_tokens + best_fragment.tokens > max_tokens:
                # 尝试找更小的片段
                alternative = self._find_alternative_fragment(
                    candidates, max_tokens - used_tokens, selected_ids
                )
                if alternative:
                    best_fragment = alternative
                else:
                    break
            
            # 添加片段
            selected.append(best_fragment)
            selected_ids.add(best_fragment.id)
            used_tokens += best_fragment.tokens
            candidates.remove(best_fragment)
            
            # 去重检查 - 移除过于相似的片段
            candidates = self._remove_similar_fragments(
                candidates, best_fragment, threshold=0.8
            )
        
        return selected
    
    async def _relevance_first_composition(self, fragments: List[KnowledgeFragment], max_tokens: int) -> List[KnowledgeFragment]:
        """相关性优先组合策略"""
        selected = []
        used_tokens = 0
        
        # 严格按相关性排序选择
        for fragment in fragments:
            if used_tokens + fragment.tokens <= max_tokens:
                # 检查重复性
                if not self._is_too_similar_to_selected(fragment, selected):
                    selected.append(fragment)
                    used_tokens += fragment.tokens
        
        return selected
    
    async def _diversity_first_composition(self,
                                         fragments: List[KnowledgeFragment],
                                         relationships: List[FragmentRelationship],
                                         max_tokens: int) -> List[KnowledgeFragment]:
        """多样性优先组合策略"""
        selected = []
        used_tokens = 0
        remaining = fragments.copy()
        
        # 选择第一个最相关的片段
        if remaining:
            first = remaining.pop(0)
            selected.append(first)
            used_tokens += first.tokens
        
        # 基于多样性选择后续片段
        while remaining and used_tokens < max_tokens:
            best_fragment = None
            best_diversity_score = -1
            
            for fragment in remaining:
                if used_tokens + fragment.tokens > max_tokens:
                    continue
                
                # 计算添加此片段后的多样性分数
                diversity_score = self._calculate_diversity_with_fragment(
                    selected, fragment
                )
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_fragment = fragment
            
            if best_fragment:
                selected.append(best_fragment)
                used_tokens += best_fragment.tokens
                remaining.remove(best_fragment)
            else:
                break
        
        return selected
    
    async def _optimize_fragment_order(self,
                                     fragments: List[KnowledgeFragment],
                                     relationships: List[FragmentRelationship]) -> List[KnowledgeFragment]:
        """优化片段顺序"""
        if len(fragments) <= 1:
            return fragments
        
        # 构建关系图
        relation_graph = defaultdict(list)
        for rel in relationships:
            if rel.fragment_a in [f.id for f in fragments] and rel.fragment_b in [f.id for f in fragments]:
                relation_graph[rel.fragment_a].append((rel.fragment_b, rel.relationship_type, rel.strength))
                relation_graph[rel.fragment_b].append((rel.fragment_a, rel.relationship_type, rel.strength))
        
        # 寻找最佳排序
        ordered_fragments = self._find_optimal_order(fragments, relation_graph)
        
        return ordered_fragments
    
    def _classify_fragment_type(self, content: str, result_item: Dict[str, Any]) -> FragmentType:
        """分类片段类型"""
        content_lower = content.lower()
        file_type = result_item.get("file_type", "").lower()
        
        # 基于文件类型
        if file_type in ["py", "js", "java", "cpp", "c", "go", "rs", "python"]:
            return FragmentType.CODE
        
        # 基于内容模式 - 检查代码块
        if re.search(r'```|`[^`]+`', content):
            return FragmentType.CODE
        elif re.search(r'^(def |class |function |public |private )', content_lower):
            return FragmentType.CODE
        elif re.search(r'^(什么是|定义|概念)', content_lower):
            return FragmentType.DEFINITION
        elif re.search(r'^(例如|示例|举例)', content_lower):
            return FragmentType.EXAMPLE
        elif re.search(r'(步骤|第一步|第二步|首先|然后|最后)', content_lower):
            return FragmentType.PROCEDURE
        elif re.search(r'(参考|引用|来源)', content_lower):
            return FragmentType.REFERENCE
        elif len(content) > 500:
            return FragmentType.EXPLANATION
        else:
            return FragmentType.CONTEXT
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本token数量"""
        # 改进的token估算算法
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)
        english_word_count = len(english_words)
        
        # 计算数字和符号
        numbers = len(re.findall(r'\b\d+\b', text))
        punctuation = len(re.findall(r'[.,!?;:()\[\]{}"\'\-_`]', text))
        
        # 特殊处理代码块和技术字符
        code_blocks = len(re.findall(r'```[\s\S]*?```', text))
        inline_code = len(re.findall(r'`[^`]+`', text))
        special_chars = len(re.findall(r'[\n\t\r]', text))  # 换行符等
        
        # 基础token计算
        base_tokens = chinese_chars + english_word_count + numbers
        
        # 添加标点符号和特殊字符的token贡献
        punctuation_tokens = max(1, punctuation // 2)
        
        # 代码块通常包含更多的token
        code_tokens = code_blocks * 3 + inline_code * 2 + special_chars
        
        # 综合计算
        total_tokens = base_tokens + punctuation_tokens + code_tokens
        
        # 确保最小token数量（按空格分割的词数）
        word_count = len(text.split())
        total_tokens = max(total_tokens, word_count)
        
        return total_tokens
    
    def _calculate_information_density(self, content: str) -> float:
        """计算信息密度"""
        # 基于多个因子计算信息密度
        
        # 1. 词汇多样性 - 改进算法（支持中文）
        # 提取中文字符和英文单词
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', content.lower())
        english_words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        all_tokens = chinese_chars + english_words
        
        if len(all_tokens) == 0:
            vocabulary_diversity = 0.0
        else:
            unique_tokens = set(all_tokens)
            vocabulary_diversity = len(unique_tokens) / len(all_tokens)
            
            # 特殊处理纯重复内容
            if len(unique_tokens) == 1 and len(all_tokens) > 3:
                vocabulary_diversity = 0.1  # 纯重复内容给予很低分数
        
        # 2. 句子长度变化 - 改进算法
        sentences = re.split(r'[.!?。！？]', content)
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths:
                avg_length = sum(sentence_lengths) / len(sentence_lengths)
                if avg_length > 0:
                    length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
                    length_diversity = min(length_variance / (avg_length + 1), 1.0)
                else:
                    length_diversity = 0.0
            else:
                length_diversity = 0.0
        else:
            # 单句内容给予基础分数
            length_diversity = 0.3
        
        # 3. 结构化程度 - 改进评分
        structure_score = 0.0
        if re.search(r'[\*\-\+]\s|\d+\.\s', content):  # 列表项
            structure_score += 0.3
        if re.search(r'#+\s|##|###', content):  # 标题
            structure_score += 0.4
        if re.search(r'```[\s\S]*?```|`[^`]+`', content):  # 代码块
            structure_score += 0.5
        if re.search(r'\n\s*\n', content):  # 段落分隔
            structure_score += 0.2
        if re.search(r':|;|->|=>|\|', content):  # 结构化符号
            structure_score += 0.2
        
        # 综合计算信息密度 - 调整权重
        density = (
            vocabulary_diversity * 0.35 +
            length_diversity * 0.25 +
            min(structure_score, 1.0) * 0.4
        )
        
        # 确保最小密度值
        return max(min(density, 1.0), 0.1)
    
    def _calculate_intent_bonus(self, query_analysis: QueryAnalysis, fragment: KnowledgeFragment) -> float:
        """基于查询意图计算片段加成分数"""
        intent_type = query_analysis.intent_type
        fragment_type = fragment.fragment_type
        
        # 意图与片段类型匹配度
        match_scores = {
            QueryIntent.FACTUAL: {
                FragmentType.DEFINITION: 0.3,
                FragmentType.EXPLANATION: 0.2,
                FragmentType.EXAMPLE: 0.15,
                FragmentType.CONTEXT: 0.1
            },
            QueryIntent.PROCEDURAL: {
                FragmentType.PROCEDURE: 0.3,
                FragmentType.EXAMPLE: 0.2,
                FragmentType.CODE: 0.15,
                FragmentType.EXPLANATION: 0.1
            },
            QueryIntent.CODE: {
                FragmentType.CODE: 0.3,
                FragmentType.EXAMPLE: 0.2,
                FragmentType.PROCEDURE: 0.15,
                FragmentType.EXPLANATION: 0.1
            },
            QueryIntent.CREATIVE: {
                FragmentType.EXAMPLE: 0.2,
                FragmentType.EXPLANATION: 0.15,
                FragmentType.CONTEXT: 0.1,
                FragmentType.REFERENCE: 0.05
            },
            QueryIntent.EXPLORATORY: {
                FragmentType.EXPLANATION: 0.2,
                FragmentType.CONTEXT: 0.15,
                FragmentType.REFERENCE: 0.1,
                FragmentType.DEFINITION: 0.1
            }
        }
        
        return match_scores.get(intent_type, {}).get(fragment_type, 0.0)
    
    async def _calculate_content_quality_bonus(self, fragment: KnowledgeFragment) -> float:
        """计算内容质量加成"""
        content = fragment.content
        quality_bonus = 0.0
        
        # 1. 长度适中加成
        if 100 <= len(content) <= 800:
            quality_bonus += 0.05
        
        # 2. 包含结构化信息加成
        if re.search(r'```|`[^`]+`', content):  # 代码
            quality_bonus += 0.05
        if re.search(r'[\*\-]\s', content):    # 列表
            quality_bonus += 0.03
        if re.search(r'#+\s', content):        # 标题
            quality_bonus += 0.02
        
        # 3. 包含数字或具体数据加成
        if re.search(r'\d+%|\d+\.\d+|\d+年|\d+月', content):
            quality_bonus += 0.03
        
        # 4. 避免过短内容惩罚
        if len(content.strip()) < 50:
            quality_bonus -= 0.1
        
        return max(quality_bonus, 0.0)
    
    def _calculate_keyword_match_bonus(self, query_analysis: QueryAnalysis, fragment: KnowledgeFragment) -> float:
        """计算关键词匹配加成"""
        content_lower = fragment.content.lower()
        
        # 计算关键词匹配
        matched_keywords = 0
        for keyword in query_analysis.keywords:
            if keyword.lower() in content_lower:
                matched_keywords += 1
        
        keyword_bonus = (matched_keywords / max(len(query_analysis.keywords), 1)) * 0.1
        
        # 计算实体匹配
        matched_entities = 0
        for entity in query_analysis.entities:
            if entity.lower() in content_lower:
                matched_entities += 1
        
        entity_bonus = (matched_entities / max(len(query_analysis.entities), 1)) * 0.15
        
        return keyword_bonus + entity_bonus
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        # 使用序列匹配器计算相似度
        similarity = SequenceMatcher(None, content1.lower(), content2.lower()).ratio()
        return similarity
    
    def _detect_dependency(self, frag_a: KnowledgeFragment, frag_b: KnowledgeFragment) -> float:
        """检测依赖关系强度"""
        content_a = frag_a.content.lower()
        content_b = frag_b.content.lower()
        
        dependency_strength = 0.0
        
        # 检测引用模式
        if any(term in content_a for term in ["基于", "依赖", "需要", "要求"]):
            if any(term in content_b for term in frag_a.content.split()[:5]):
                dependency_strength += 0.3
        
        # 检测概念层次
        if frag_a.fragment_type == FragmentType.DEFINITION and frag_b.fragment_type == FragmentType.EXAMPLE:
            dependency_strength += 0.2
        
        return min(dependency_strength, 1.0)
    
    def _detect_sequence(self, frag_a: KnowledgeFragment, frag_b: KnowledgeFragment) -> float:
        """检测顺序关系强度"""
        content_a = frag_a.content.lower()
        content_b = frag_b.content.lower()
        
        sequence_strength = 0.0
        
        # 检测步骤模式 - 改进正则表达式和匹配逻辑
        step_patterns_a = re.findall(r'第(\d+)[步点项]|步骤\s*(\d+)|第(\d+)步|步骤(\d+)', content_a)
        step_patterns_b = re.findall(r'第(\d+)[步点项]|步骤\s*(\d+)|第(\d+)步|步骤(\d+)', content_b)
        
        # 提取数字的改进逻辑
        def extract_step_numbers(patterns):
            numbers = []
            for match in patterns:
                for group in match:
                    if group and group.isdigit():
                        numbers.append(int(group))
            return numbers
        
        nums_a = extract_step_numbers(step_patterns_a)
        nums_b = extract_step_numbers(step_patterns_b)
        
        if nums_a and nums_b:
            min_a, min_b = min(nums_a), min(nums_b)
            diff = abs(min_a - min_b)
            if diff == 1:
                sequence_strength = 0.8  # 提高连续步骤的强度
            elif diff <= 2:
                sequence_strength = 0.5
            elif diff <= 3:
                sequence_strength = 0.3
        
        # 检测中文数字步骤（一、二、三等）
        chinese_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        chinese_a = re.findall(r'第([一二三四五六七八九十]+)[步点项]', content_a)
        chinese_b = re.findall(r'第([一二三四五六七八九十]+)[步点项]', content_b)
        
        if chinese_a and chinese_b:
            try:
                num_a = chinese_num_map.get(chinese_a[0], 0)
                num_b = chinese_num_map.get(chinese_b[0], 0)
                if num_a and num_b and abs(num_a - num_b) == 1:
                    sequence_strength = max(sequence_strength, 0.8)
            except Exception:
                logger.exception("解析中文步骤失败", exc_info=True)
        
        # 检测时间顺序词
        time_words = ["首先", "然后", "接着", "最后", "之后", "先", "再", "finally", "first", "second", "next"]
        a_has_time = any(word in content_a for word in time_words)
        b_has_time = any(word in content_b for word in time_words)
        
        if a_has_time and b_has_time:
            sequence_strength = max(sequence_strength, 0.4)
        
        # 检测程序性片段的自然顺序
        if frag_a.fragment_type == FragmentType.PROCEDURE and frag_b.fragment_type == FragmentType.PROCEDURE:
            # 如果两个都是程序性片段且没有明确步骤标记，给予基础顺序分数
            if sequence_strength == 0.0:
                sequence_strength = 0.2
        
        return min(sequence_strength, 1.0)
    
    def _detect_hierarchy(self, frag_a: KnowledgeFragment, frag_b: KnowledgeFragment) -> float:
        """检测层次关系强度"""
        hierarchy_strength = 0.0
        
        # 基于片段类型的层次关系
        type_hierarchy = {
            (FragmentType.DEFINITION, FragmentType.EXPLANATION): 0.5,
            (FragmentType.DEFINITION, FragmentType.EXAMPLE): 0.4,
            (FragmentType.EXPLANATION, FragmentType.EXAMPLE): 0.3,
            (FragmentType.PROCEDURE, FragmentType.CODE): 0.4,
        }
        
        type_pair = (frag_a.fragment_type, frag_b.fragment_type)
        reverse_pair = (frag_b.fragment_type, frag_a.fragment_type)
        
        if type_pair in type_hierarchy:
            hierarchy_strength = type_hierarchy[type_pair]
        elif reverse_pair in type_hierarchy:
            hierarchy_strength = type_hierarchy[reverse_pair]
        
        return hierarchy_strength
    
    def _detect_contrast(self, frag_a: KnowledgeFragment, frag_b: KnowledgeFragment) -> float:
        """检测对比关系强度"""
        content_a = frag_a.content.lower()
        content_b = frag_b.content.lower()
        
        contrast_strength = 0.0
        
        # 检测对比词汇
        contrast_words = ["但是", "然而", "相反", "不同", "而", "相比", "对比", "vs", "versus"]
        
        a_has_contrast = any(word in content_a for word in contrast_words)
        b_has_contrast = any(word in content_b for word in contrast_words)
        
        if a_has_contrast or b_has_contrast:
            contrast_strength += 0.2
        
        # 检测否定词模式
        negative_patterns_a = len(re.findall(r'不|没有|无法|不能|不是', content_a))
        negative_patterns_b = len(re.findall(r'不|没有|无法|不能|不是', content_b))
        
        if negative_patterns_a > 0 and negative_patterns_b == 0:
            contrast_strength += 0.3
        elif negative_patterns_b > 0 and negative_patterns_a == 0:
            contrast_strength += 0.3
        
        return min(contrast_strength, 1.0)
    
    def _generate_relationship_explanation(self,
                                         relationship_type: RelationshipType,
                                         strength: float,
                                         frag_a: KnowledgeFragment,
                                         frag_b: KnowledgeFragment) -> str:
        """生成关系解释"""
        explanations = {
            RelationshipType.SIMILARITY: f"两个片段在内容上相似度为{strength:.2f}",
            RelationshipType.DEPENDENCY: f"片段存在依赖关系，强度为{strength:.2f}",
            RelationshipType.SEQUENCE: f"片段存在顺序关系，强度为{strength:.2f}",
            RelationshipType.HIERARCHY: f"片段存在层次关系，强度为{strength:.2f}",
            RelationshipType.CONTRAST: f"片段存在对比关系，强度为{strength:.2f}",
            RelationshipType.COMPLEMENT: f"片段互相补充，强度为{strength:.2f}"
        }
        
        return explanations.get(relationship_type, f"未知关系类型，强度为{strength:.2f}")
    
    def _select_best_complement(self,
                              candidates: List[KnowledgeFragment],
                              selected: List[KnowledgeFragment],
                              relationships: List[FragmentRelationship],
                              query_analysis: QueryAnalysis) -> KnowledgeFragment:
        """选择最佳补充片段"""
        best_fragment = None
        best_score = -1
        
        selected_ids = {f.id for f in selected}
        
        for candidate in candidates:
            if candidate.id in selected_ids:
                continue
            
            # 计算综合分数
            relevance_score = candidate.relevance_score
            
            # 计算与已选片段的关系分数
            relationship_score = 0.0
            relationship_count = 0
            
            for rel in relationships:
                if rel.fragment_a == candidate.id and rel.fragment_b in selected_ids:
                    relationship_score += rel.strength
                    relationship_count += 1
                elif rel.fragment_b == candidate.id and rel.fragment_a in selected_ids:
                    relationship_score += rel.strength
                    relationship_count += 1
            
            if relationship_count > 0:
                avg_relationship_score = relationship_score / relationship_count
            else:
                avg_relationship_score = 0.0
            
            # 计算多样性分数（避免过度相似）
            diversity_penalty = 0.0
            for selected_frag in selected:
                similarity = self._calculate_content_similarity(candidate.content, selected_frag.content)
                if similarity > 0.8:
                    diversity_penalty += (similarity - 0.8) * 0.5
            
            # 综合分数
            composite_score = (
                relevance_score * self.relevance_weight +
                avg_relationship_score * self.coherence_weight +
                max(0, self.diversity_weight - diversity_penalty)
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_fragment = candidate
        
        return best_fragment or candidates[0]
    
    def _find_alternative_fragment(self,
                                 candidates: List[KnowledgeFragment],
                                 remaining_tokens: int,
                                 selected_ids: Set[str]) -> Optional[KnowledgeFragment]:
        """寻找符合token限制的替代片段"""
        suitable_candidates = [
            c for c in candidates 
            if c.tokens <= remaining_tokens and c.id not in selected_ids
        ]
        
        if not suitable_candidates:
            return None
        
        # 按相关性排序，返回最佳的
        suitable_candidates.sort(key=lambda f: f.relevance_score, reverse=True)
        return suitable_candidates[0]
    
    def _remove_similar_fragments(self,
                                candidates: List[KnowledgeFragment],
                                reference_fragment: KnowledgeFragment,
                                threshold: float = 0.8) -> List[KnowledgeFragment]:
        """移除与参考片段过于相似的片段"""
        filtered = []
        
        for candidate in candidates:
            similarity = self._calculate_content_similarity(
                candidate.content, reference_fragment.content
            )
            if similarity < threshold:
                filtered.append(candidate)
        
        return filtered
    
    def _is_too_similar_to_selected(self,
                                  candidate: KnowledgeFragment,
                                  selected: List[KnowledgeFragment],
                                  threshold: float = 0.8) -> bool:
        """检查候选片段是否与已选片段过于相似"""
        for selected_frag in selected:
            similarity = self._calculate_content_similarity(
                candidate.content, selected_frag.content
            )
            if similarity > threshold:
                return True
        return False
    
    def _calculate_diversity_with_fragment(self,
                                         selected: List[KnowledgeFragment],
                                         candidate: KnowledgeFragment) -> float:
        """计算添加候选片段后的多样性分数"""
        if not selected:
            return 1.0
        
        # 计算内容多样性
        content_diversities = []
        for selected_frag in selected:
            similarity = self._calculate_content_similarity(candidate.content, selected_frag.content)
            diversity = 1.0 - similarity
            content_diversities.append(diversity)
        
        avg_content_diversity = sum(content_diversities) / len(content_diversities)
        
        # 计算类型多样性
        selected_types = {f.fragment_type for f in selected}
        type_diversity = 1.0 if candidate.fragment_type not in selected_types else 0.5
        
        # 计算来源多样性
        selected_sources = {f.source for f in selected}
        source_diversity = 1.0 if candidate.source not in selected_sources else 0.7
        
        # 综合多样性分数
        diversity_score = (
            avg_content_diversity * 0.6 +
            type_diversity * 0.3 +
            source_diversity * 0.1
        )
        
        return diversity_score
    
    def _find_optimal_order(self,
                          fragments: List[KnowledgeFragment],
                          relation_graph: Dict[str, List[Tuple[str, RelationshipType, float]]]) -> List[KnowledgeFragment]:
        """寻找最佳片段顺序"""
        if len(fragments) <= 2:
            return fragments
        
        fragment_dict = {f.id: f for f in fragments}
        
        # 简单启发式：按片段类型和关系强度排序
        
        # 1. 优先排序定义和解释类片段
        priority_order = [
            FragmentType.DEFINITION,
            FragmentType.EXPLANATION,
            FragmentType.PROCEDURE,
            FragmentType.EXAMPLE,
            FragmentType.CODE,
            FragmentType.CONTEXT,
            FragmentType.REFERENCE
        ]
        
        # 2. 按类型分组
        type_groups = defaultdict(list)
        for frag in fragments:
            type_groups[frag.fragment_type].append(frag)
        
        # 3. 构建有序列表
        ordered = []
        for frag_type in priority_order:
            if frag_type in type_groups:
                # 在同类型内按相关性排序
                group_fragments = sorted(type_groups[frag_type], key=lambda f: f.relevance_score, reverse=True)
                ordered.extend(group_fragments)
        
        # 4. 基于关系图进行局部调整
        if relation_graph:
            ordered = self._adjust_order_by_relationships(ordered, relation_graph)
        
        return ordered
    
    def _adjust_order_by_relationships(self,
                                     ordered_fragments: List[KnowledgeFragment],
                                     relation_graph: Dict[str, List[Tuple[str, RelationshipType, float]]]) -> List[KnowledgeFragment]:
        """基于关系图调整片段顺序"""
        # 简化实现：寻找强依赖和顺序关系，进行局部调整
        adjusted = ordered_fragments.copy()
        fragment_positions = {f.id: i for i, f in enumerate(adjusted)}
        
        # 处理依赖和顺序关系
        for frag_id, relations in relation_graph.items():
            if frag_id not in fragment_positions:
                continue
            
            current_pos = fragment_positions[frag_id]
            
            for related_id, rel_type, strength in relations:
                if related_id not in fragment_positions or strength < 0.5:
                    continue
                
                related_pos = fragment_positions[related_id]
                
                # 如果是依赖关系，被依赖的应该在前面
                if rel_type == RelationshipType.DEPENDENCY and current_pos < related_pos:
                    # 交换位置
                    adjusted[current_pos], adjusted[related_pos] = adjusted[related_pos], adjusted[current_pos]
                    fragment_positions[frag_id], fragment_positions[related_id] = related_pos, current_pos
                
                # 如果是顺序关系，按顺序排列
                elif rel_type == RelationshipType.SEQUENCE:
                    # 简化处理：如果发现逆序，进行调整
                    if current_pos > related_pos:
                        adjusted[current_pos], adjusted[related_pos] = adjusted[related_pos], adjusted[current_pos]
                        fragment_positions[frag_id], fragment_positions[related_id] = related_pos, current_pos
        
        return adjusted
    
    def _calculate_composition_metrics(self,
                                     fragments: List[KnowledgeFragment],
                                     relationships: List[FragmentRelationship]) -> Dict[str, float]:
        """计算组合指标"""
        if not fragments:
            return {
                "information_density": 0.0,
                "diversity_score": 0.0,
                "coherence_score": 0.0,
                "coverage_score": 0.0
            }
        
        # 1. 信息密度 - 所有片段的平均信息密度
        avg_info_density = sum(f.information_density for f in fragments) / len(fragments)
        
        # 2. 多样性分数 - 基于内容、类型、来源的多样性
        content_similarities = []
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                similarity = self._calculate_content_similarity(fragments[i].content, fragments[j].content)
                content_similarities.append(similarity)
        
        avg_similarity = sum(content_similarities) / max(len(content_similarities), 1)
        content_diversity = 1.0 - avg_similarity
        
        # 类型多样性
        unique_types = len(set(f.fragment_type for f in fragments))
        max_types = len(FragmentType)
        type_diversity = unique_types / max_types
        
        # 来源多样性
        unique_sources = len(set(f.source for f in fragments))
        source_diversity = min(unique_sources / max(len(fragments), 1), 1.0)
        
        diversity_score = (content_diversity * 0.6 + type_diversity * 0.3 + source_diversity * 0.1)
        
        # 3. 连贯性分数 - 基于片段间关系
        fragment_ids = {f.id for f in fragments}
        relevant_relationships = [r for r in relationships if r.fragment_a in fragment_ids and r.fragment_b in fragment_ids]
        
        if relevant_relationships:
            avg_relation_strength = sum(r.strength for r in relevant_relationships) / len(relevant_relationships)
            relation_density = len(relevant_relationships) / max((len(fragments) * (len(fragments) - 1)) // 2, 1)
            coherence_score = avg_relation_strength * 0.7 + relation_density * 0.3
        else:
            coherence_score = 0.0
        
        # 4. 覆盖度分数 - 基于片段类型的覆盖度
        coverage_score = type_diversity  # 简化为类型多样性
        
        return {
            "information_density": avg_info_density,
            "diversity_score": diversity_score,
            "coherence_score": coherence_score,
            "coverage_score": coverage_score
        }

# 全局上下文组合器实例
context_composer = ContextComposer()
