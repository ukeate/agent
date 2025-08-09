"""
查询扩展器模块

实现智能查询扩展和改写功能，包括：
- 同义词扩展和语义相似词生成
- 基于上下文的查询改写策略
- 查询分解和子问题生成
- 多语言查询扩展支持
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..openai_client import get_openai_client
from .query_analyzer import QueryAnalysis, QueryIntent


class ExpansionStrategy(str, Enum):
    """查询扩展策略"""
    SYNONYM = "synonym"           # 同义词扩展
    SEMANTIC = "semantic"         # 语义相似扩展
    CONTEXTUAL = "contextual"     # 上下文改写
    DECOMPOSITION = "decomposition"  # 查询分解
    MULTILINGUAL = "multilingual"    # 多语言扩展


@dataclass
class ExpandedQuery:
    """扩展后的查询结果"""
    original_query: str
    expanded_queries: List[str]
    strategy: ExpansionStrategy
    confidence: float  # 扩展质量置信度 0-1
    sub_questions: Optional[List[str]] = None  # 子问题（分解策略时使用）
    language_variants: Optional[Dict[str, str]] = None  # 多语言变体
    explanation: Optional[str] = None  # 扩展解释


class QueryExpander:
    """查询扩展器"""
    
    def __init__(self):
        self.client = None
        self._synonym_dict = self._build_synonym_dict()
        self._domain_terms = self._build_domain_terms()
    
    async def _get_client(self):
        """获取OpenAI客户端"""
        if self.client is None:
            self.client = await get_openai_client()
        return self.client
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """构建同义词词典"""
        return {
            # 技术术语同义词
            "机器学习": ["ML", "机器学习算法", "人工智能学习", "自动学习"],
            "人工智能": ["AI", "智能系统", "人工智慧", "智能算法"],
            "深度学习": ["深度神经网络", "DL", "深度学习算法"],
            "神经网络": ["NN", "人工神经网络", "神经元网络"],
            "数据库": ["DB", "数据存储", "数据仓库", "数据管理系统"],
            "接口": ["API", "应用程序接口", "编程接口", "服务接口"],
            "算法": ["计算方法", "解决方案", "处理方法", "计算程序"],
            
            # 动作同义词
            "实现": ["开发", "构建", "创建", "建立", "制作"],
            "优化": ["改进", "提升", "完善", "增强", "优化算法"],
            "分析": ["研究", "探讨", "解析", "调研", "检查"],
            "设计": ["规划", "构思", "架构", "布局", "策划"],
            
            # 疑问词同义词
            "如何": ["怎么", "怎样", "什么方法", "通过什么方式"],
            "什么": ["哪些", "什么样的", "何种", "什么类型的"],
            "为什么": ["为何", "什么原因", "出于什么考虑"],
        }
    
    def _build_domain_terms(self) -> Dict[str, List[str]]:
        """构建领域术语映射"""
        return {
            "技术": [
                "编程", "开发", "代码", "系统", "架构", "框架", "库", "工具",
                "数据库", "服务器", "网络", "安全", "性能", "测试", "部署"
            ],
            "业务": [
                "需求", "流程", "管理", "用户", "客户", "产品", "服务", "运营",
                "策略", "目标", "指标", "效果", "价值", "收益", "成本"
            ],
            "学术": [
                "研究", "理论", "方法", "实验", "数据", "分析", "结果", "结论",
                "文献", "论文", "假设", "验证", "模型", "评估", "应用"
            ]
        }

    async def expand_query(self, 
                          query_analysis: QueryAnalysis,
                          context_history: Optional[List[str]] = None,
                          strategies: Optional[List[ExpansionStrategy]] = None) -> List[ExpandedQuery]:
        """
        扩展查询，支持多种策略
        
        Args:
            query_analysis: 查询分析结果
            context_history: 历史对话上下文
            strategies: 指定使用的扩展策略，None时自动选择
            
        Returns:
            List[ExpandedQuery]: 扩展结果列表
        """
        if strategies is None:
            strategies = self._select_strategies(query_analysis)
        else:
            # 当明确指定策略时，确保同义词扩展总是包含在内
            if ExpansionStrategy.SYNONYM not in strategies:
                strategies = [ExpansionStrategy.SYNONYM] + list(strategies)
        
        results = []
        
        for strategy in strategies:
            try:
                if strategy == ExpansionStrategy.SYNONYM:
                    result = await self._expand_synonyms(query_analysis)
                elif strategy == ExpansionStrategy.SEMANTIC:
                    result = await self._expand_semantic(query_analysis, context_history)
                elif strategy == ExpansionStrategy.CONTEXTUAL:
                    result = await self._rewrite_contextual(query_analysis, context_history)
                elif strategy == ExpansionStrategy.DECOMPOSITION:
                    result = await self._decompose_query(query_analysis)
                elif strategy == ExpansionStrategy.MULTILINGUAL:
                    result = await self._expand_multilingual(query_analysis)
                else:
                    continue
                
                if result:
                    results.append(result)
                    
            except Exception as e:
                # 记录错误但继续其他策略
                print(f"扩展策略 {strategy} 失败: {e}")
                continue
        
        return results

    def _select_strategies(self, query_analysis: QueryAnalysis) -> List[ExpansionStrategy]:
        """根据查询分析自动选择扩展策略"""
        strategies = []
        
        # 基础策略：总是尝试同义词扩展
        strategies.append(ExpansionStrategy.SYNONYM)
        
        # 基于意图选择策略
        if query_analysis.intent_type == QueryIntent.FACTUAL:
            strategies.extend([ExpansionStrategy.SEMANTIC, ExpansionStrategy.MULTILINGUAL])
        elif query_analysis.intent_type == QueryIntent.PROCEDURAL:
            strategies.extend([ExpansionStrategy.DECOMPOSITION, ExpansionStrategy.CONTEXTUAL])
        elif query_analysis.intent_type == QueryIntent.CODE:
            strategies.append(ExpansionStrategy.SEMANTIC)
        elif query_analysis.intent_type in [QueryIntent.CREATIVE, QueryIntent.EXPLORATORY]:
            strategies.extend([ExpansionStrategy.SEMANTIC, ExpansionStrategy.CONTEXTUAL])
        
        # 基于复杂度选择策略
        if query_analysis.complexity_score > 0.5:
            if ExpansionStrategy.DECOMPOSITION not in strategies:
                strategies.append(ExpansionStrategy.DECOMPOSITION)
        
        # 基于语言选择策略
        if query_analysis.language == "zh" and len(query_analysis.entities) > 0:
            if ExpansionStrategy.MULTILINGUAL not in strategies:
                strategies.append(ExpansionStrategy.MULTILINGUAL)
        
        return strategies

    async def _expand_synonyms(self, query_analysis: QueryAnalysis) -> ExpandedQuery:
        """同义词扩展策略"""
        original_query = query_analysis.query_text
        expanded_queries = []
        
        # 使用内置同义词词典
        for term, synonyms in self._synonym_dict.items():
            if term in original_query:
                for synonym in synonyms[:2]:  # 限制每个术语最多2个同义词
                    expanded_query = original_query.replace(term, synonym)
                    if expanded_query != original_query:
                        expanded_queries.append(expanded_query)
        
        # 基于关键词生成同义词扩展
        for keyword in query_analysis.keywords:
            if keyword in self._synonym_dict:
                for synonym in self._synonym_dict[keyword][:1]:
                    expanded_query = original_query.replace(keyword, synonym)
                    if expanded_query != original_query and expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        # 去重并限制数量
        expanded_queries = list(set(expanded_queries))[:5]
        
        return ExpandedQuery(
            original_query=original_query,
            expanded_queries=expanded_queries,
            strategy=ExpansionStrategy.SYNONYM,
            confidence=0.8 if expanded_queries else 0.0,
            explanation="基于同义词词典进行术语替换扩展"
        )

    async def _expand_semantic(self, 
                             query_analysis: QueryAnalysis,
                             context_history: Optional[List[str]] = None) -> ExpandedQuery:
        """语义相似扩展策略"""
        context_str = ""
        if context_history:
            context_str = f"对话历史：{' | '.join(context_history[-3:])}\n"
        
        system_prompt = """你是一个查询语义扩展专家。请为给定查询生成语义相似的扩展查询。

要求：
1. 保持原始查询的核心意图不变
2. 使用不同的表达方式和词汇
3. 生成3-5个高质量的扩展查询
4. 确保扩展查询的语法正确和语义清晰

请以JSON格式返回：{"expanded_queries": ["扩展1", "扩展2", ...], "confidence": 0.85}"""

        user_prompt = f"""{context_str}原始查询：{query_analysis.query_text}
查询意图：{query_analysis.intent_type.value}
查询领域：{query_analysis.domain or "未知"}

请生成语义相似的扩展查询。"""

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            expanded_queries = result.get("expanded_queries", [])
            confidence = result.get("confidence", 0.5)
            
            return ExpandedQuery(
                original_query=query_analysis.query_text,
                expanded_queries=expanded_queries,
                strategy=ExpansionStrategy.SEMANTIC,
                confidence=min(max(confidence, 0.0), 1.0),
                explanation="基于语义理解生成相似表达的查询扩展"
            )
            
        except Exception:
            # 失败时使用规则方法作为后备
            return self._semantic_fallback(query_analysis)

    def _semantic_fallback(self, query_analysis: QueryAnalysis) -> ExpandedQuery:
        """语义扩展的后备方法"""
        original_query = query_analysis.query_text
        expanded_queries = []
        
        # 基于意图生成模板扩展
        if query_analysis.intent_type == QueryIntent.FACTUAL:
            templates = [
                f"什么是{original_query.replace('什么是', '').replace('是什么', '')}",
                f"关于{original_query}的信息",
                f"{original_query}的详细介绍"
            ]
        elif query_analysis.intent_type == QueryIntent.PROCEDURAL:
            templates = [
                f"步骤：{original_query}",
                f"教程：{original_query}",
                f"指南：{original_query}"
            ]
        else:
            templates = [f"详细说明：{original_query}"]
        
        expanded_queries = [t for t in templates if t != original_query][:3]
        
        return ExpandedQuery(
            original_query=original_query,
            expanded_queries=expanded_queries,
            strategy=ExpansionStrategy.SEMANTIC,
            confidence=0.4,
            explanation="基于规则模板生成的语义扩展（后备方案）"
        )

    async def _rewrite_contextual(self, 
                                query_analysis: QueryAnalysis,
                                context_history: Optional[List[str]] = None) -> ExpandedQuery:
        """上下文改写策略"""
        if not context_history:
            return ExpandedQuery(
                original_query=query_analysis.query_text,
                expanded_queries=[],
                strategy=ExpansionStrategy.CONTEXTUAL,
                confidence=0.0,
                explanation="缺少上下文历史，无法进行上下文改写"
            )
        
        context_str = " | ".join(context_history[-3:])
        
        system_prompt = """你是一个查询上下文改写专家。基于对话历史和当前查询，生成融合上下文信息的改写查询。

要求：
1. 结合历史对话信息，使查询更加完整和具体
2. 保持查询的原始意图
3. 生成2-4个高质量的上下文改写查询
4. 确保改写后的查询独立可理解

请以JSON格式返回：{"rewritten_queries": ["改写1", "改写2", ...], "confidence": 0.85}"""

        user_prompt = f"""对话历史：{context_str}

当前查询：{query_analysis.query_text}

请基于上下文生成改写查询。"""

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            rewritten_queries = result.get("rewritten_queries", [])
            confidence = result.get("confidence", 0.5)
            
            return ExpandedQuery(
                original_query=query_analysis.query_text,
                expanded_queries=rewritten_queries,
                strategy=ExpansionStrategy.CONTEXTUAL,
                confidence=min(max(confidence, 0.0), 1.0),
                explanation="基于对话上下文进行查询改写，融合历史信息"
            )
            
        except Exception:
            return self._contextual_fallback(query_analysis, context_history)

    def _contextual_fallback(self, 
                           query_analysis: QueryAnalysis,
                           context_history: List[str]) -> ExpandedQuery:
        """上下文改写的后备方法"""
        original_query = query_analysis.query_text
        
        # 简单的上下文融合
        last_context = context_history[-1] if context_history else ""
        
        # 提取上下文中的关键词
        context_keywords = []
        for ctx in context_history[-2:]:
            words = re.findall(r'[\u4e00-\u9fff]+|\w+', ctx)
            context_keywords.extend(words[:3])
        
        expanded_queries = []
        if context_keywords:
            # 将上下文关键词融入查询
            for keyword in context_keywords[:2]:
                if keyword not in original_query:
                    expanded_query = f"关于{keyword}的{original_query}"
                    expanded_queries.append(expanded_query)
        
        return ExpandedQuery(
            original_query=original_query,
            expanded_queries=expanded_queries,
            strategy=ExpansionStrategy.CONTEXTUAL,
            confidence=0.3,
            explanation="基于简单规则的上下文改写（后备方案）"
        )

    async def _decompose_query(self, query_analysis: QueryAnalysis) -> ExpandedQuery:
        """查询分解策略"""
        system_prompt = """你是一个查询分解专家。将复杂查询分解为多个简单的子问题。

要求：
1. 分解后的子问题应该涵盖原查询的所有关键方面
2. 每个子问题应该独立且具体
3. 生成3-6个高质量的子问题
4. 保持逻辑顺序和层次结构

请以JSON格式返回：{"sub_questions": ["子问题1", "子问题2", ...], "confidence": 0.85}"""

        user_prompt = f"""查询：{query_analysis.query_text}
查询意图：{query_analysis.intent_type.value}
复杂度：{query_analysis.complexity_score}

请将此查询分解为子问题。"""

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            result = json.loads(response.choices[0].message.content)
            sub_questions = result.get("sub_questions", [])
            confidence = result.get("confidence", 0.5)
            
            return ExpandedQuery(
                original_query=query_analysis.query_text,
                expanded_queries=sub_questions,
                strategy=ExpansionStrategy.DECOMPOSITION,
                confidence=min(max(confidence, 0.0), 1.0),
                sub_questions=sub_questions,
                explanation="将复杂查询分解为多个具体的子问题"
            )
            
        except Exception:
            return self._decomposition_fallback(query_analysis)

    def _decomposition_fallback(self, query_analysis: QueryAnalysis) -> ExpandedQuery:
        """查询分解的后备方法"""
        original_query = query_analysis.query_text
        sub_questions = []
        
        # 基于关键词生成子问题
        keywords = query_analysis.keywords[:3]
        entities = query_analysis.entities[:3]
        
        for keyword in keywords:
            sub_questions.append(f"什么是{keyword}")
        
        for entity in entities:
            sub_questions.append(f"关于{entity}的详细信息")
        
        # 基于意图生成通用子问题
        if query_analysis.intent_type == QueryIntent.PROCEDURAL:
            sub_questions.extend([
                f"需要什么前提条件",
                f"具体步骤是什么",
                f"可能遇到什么问题"
            ])
        elif query_analysis.intent_type == QueryIntent.CODE:
            sub_questions.extend([
                f"需要什么技术栈",
                f"核心实现逻辑",
                f"如何处理错误"
            ])
        
        # 去重并限制数量
        sub_questions = list(set(sub_questions))[:5]
        
        return ExpandedQuery(
            original_query=original_query,
            expanded_queries=sub_questions,
            strategy=ExpansionStrategy.DECOMPOSITION,
            confidence=0.3,
            sub_questions=sub_questions,
            explanation="基于关键词和意图的简单查询分解（后备方案）"
        )

    async def _expand_multilingual(self, query_analysis: QueryAnalysis) -> ExpandedQuery:
        """多语言扩展策略"""
        if query_analysis.language == "en":
            # 英文查询，生成中文变体
            target_language = "中文"
            source_language = "英文"
        else:
            # 中文查询，生成英文变体
            target_language = "英文"
            source_language = "中文"
        
        system_prompt = f"""你是一个多语言查询扩展专家。将{source_language}查询翻译为{target_language}，并生成语义等价的表达变体。

要求：
1. 保持原始查询的核心意图和语义
2. 生成地道的{target_language}表达
3. 提供2-3个不同的表达变体
4. 确保翻译准确和语法正确

请以JSON格式返回：{{"translations": ["翻译1", "翻译2", ...], "confidence": 0.85}}"""

        user_prompt = f"""原始查询（{source_language}）：{query_analysis.query_text}

请翻译为{target_language}并生成变体。"""

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            translations = result.get("translations", [])
            confidence = result.get("confidence", 0.5)
            
            language_variants = {
                target_language.lower(): translations[0] if translations else ""
            }
            
            return ExpandedQuery(
                original_query=query_analysis.query_text,
                expanded_queries=translations,
                strategy=ExpansionStrategy.MULTILINGUAL,
                confidence=min(max(confidence, 0.0), 1.0),
                language_variants=language_variants,
                explanation=f"生成{target_language}翻译和表达变体"
            )
            
        except Exception:
            return self._multilingual_fallback(query_analysis)

    def _multilingual_fallback(self, query_analysis: QueryAnalysis) -> ExpandedQuery:
        """多语言扩展的后备方法"""
        # 简单的翻译后备（基于规则）
        original_query = query_analysis.query_text
        translations = []
        
        if query_analysis.language == "zh":
            # 中文到英文的简单映射
            translation_map = {
                "什么": "what",
                "如何": "how",
                "为什么": "why",
                "机器学习": "machine learning",
                "人工智能": "artificial intelligence",
                "数据库": "database",
                "算法": "algorithm"
            }
            
            translated = original_query
            for zh, en in translation_map.items():
                translated = translated.replace(zh, en)
            
            if translated != original_query:
                translations = [translated]
        
        return ExpandedQuery(
            original_query=original_query,
            expanded_queries=translations,
            strategy=ExpansionStrategy.MULTILINGUAL,
            confidence=0.2,
            language_variants={"en" if query_analysis.language == "zh" else "zh": translations[0] if translations else ""},
            explanation="基于简单规则的多语言扩展（后备方案）"
        )

    def get_best_expansions(self, 
                           expansion_results: List[ExpandedQuery],
                           max_results: int = 10) -> List[str]:
        """
        从多个扩展结果中选择最佳的扩展查询
        
        Args:
            expansion_results: 扩展结果列表
            max_results: 最大返回结果数
            
        Returns:
            List[str]: 最佳扩展查询列表
        """
        all_queries = []
        
        # 收集所有扩展查询，按置信度加权
        for result in expansion_results:
            for query in result.expanded_queries:
                all_queries.append((query, result.confidence, result.strategy))
        
        # 去重
        unique_queries = {}
        for query, confidence, strategy in all_queries:
            if query not in unique_queries:
                unique_queries[query] = (confidence, strategy)
            else:
                # 如果重复，保留置信度更高的
                if confidence > unique_queries[query][0]:
                    unique_queries[query] = (confidence, strategy)
        
        # 排序并返回
        sorted_queries = sorted(
            unique_queries.items(),
            key=lambda x: x[1][0],  # 按置信度排序
            reverse=True
        )
        
        return [query for query, (confidence, strategy) in sorted_queries[:max_results]]