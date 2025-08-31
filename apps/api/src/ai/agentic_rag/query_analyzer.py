"""
查询分析器模块

实现查询理解和意图识别功能，包括：
- 查询意图分类（factual、procedural、code、creative）
- 查询复杂度评估
- 实体提取和关键词识别
- 上下文理解和历史记忆
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from src.ai.openai_client import get_openai_client


class QueryIntent(str, Enum):
    """查询意图类型"""
    FACTUAL = "factual"        # 事实性查询：寻找具体信息
    PROCEDURAL = "procedural"  # 程序性查询：如何做某事
    CODE = "code"             # 代码相关查询：编程问题
    CREATIVE = "creative"     # 创造性查询：需要生成内容
    EXPLORATORY = "exploratory"  # 探索性查询：开放性问题


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_text: str
    intent_type: QueryIntent
    confidence: float  # 意图分类置信度 0-1
    complexity_score: float  # 查询复杂度 0-1
    entities: List[str]  # 识别的实体
    keywords: List[str]  # 关键词
    domain: Optional[str]  # 领域分类
    sentiment: Optional[str]  # 情感倾向
    language: str = "zh"  # 查询语言
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = utc_now()


class QueryAnalyzer:
    """查询分析器"""
    
    def __init__(self):
        self.client = get_openai_client()
        self._entity_patterns = self._build_entity_patterns()
        self._stop_words = self._load_stop_words()
    
    def _build_entity_patterns(self) -> Dict[str, List[str]]:
        """构建实体识别模式"""
        return {
            "person": [r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', r'\b[A-Z][a-z]+\b'],
            "tech_term": [r'\b([A-Z]{2,}|[a-z]+[A-Z][a-zA-Z]*)\b'],
            "file_path": [r'\b[\w/\\]+\.(py|js|ts|java|cpp|c|h)\b'],
            "url": [r'https?://[\w\.-]+'],
            "number": [r'\b\d+\.?\d*\b'],
        }
    
    def _load_stop_words(self) -> set:
        """加载停用词"""
        chinese_stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", 
            "都", "一", "个", "上", "也", "很", "到", "说", "要", "去"
        }
        english_stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", 
            "to", "for", "of", "with", "by", "is", "are", "was", "were"
        }
        return chinese_stop_words | english_stop_words

    async def analyze_query(self, 
                           query: str, 
                           context_history: Optional[List[str]] = None) -> QueryAnalysis:
        """
        分析查询，返回完整的分析结果
        
        Args:
            query: 用户查询文本
            context_history: 历史对话上下文
            
        Returns:
            QueryAnalysis: 查询分析结果
        """
        # 清理和预处理查询
        cleaned_query = self._preprocess_query(query)
        
        # 分析查询意图
        intent, confidence = await self._classify_intent(cleaned_query, context_history)
        
        # 评估查询复杂度
        complexity = self._assess_complexity(cleaned_query)
        
        # 提取实体和关键词
        entities = self._extract_entities(cleaned_query)
        keywords = self._extract_keywords(cleaned_query)
        
        # 识别领域
        domain = await self._identify_domain(cleaned_query)
        
        # 分析情感
        sentiment = self._analyze_sentiment(cleaned_query)
        
        # 检测语言
        language = self._detect_language(cleaned_query)
        
        return QueryAnalysis(
            query_text=cleaned_query,
            intent_type=intent,
            confidence=confidence,
            complexity_score=complexity,
            entities=entities,
            keywords=keywords,
            domain=domain,
            sentiment=sentiment,
            language=language
        )

    def _preprocess_query(self, query: str) -> str:
        """预处理查询文本"""
        # 去除多余空白字符
        query = re.sub(r'\s+', ' ', query.strip())
        
        # 统一标点符号
        query = query.replace('？', '?').replace('！', '!')
        
        return query

    async def _classify_intent(self, 
                              query: str, 
                              context_history: Optional[List[str]] = None) -> Tuple[QueryIntent, float]:
        """
        使用LLM进行查询意图分类
        
        Returns:
            Tuple[QueryIntent, float]: (意图类型, 置信度)
        """
        context_str = ""
        if context_history:
            context_str = f"对话历史：{' | '.join(context_history[-3:])}\n"
        
        system_prompt = """你是一个查询意图分类专家。请分析用户查询的意图类型并给出置信度评分。

意图类型定义：
- factual: 事实性查询，寻找具体信息、数据或答案
- procedural: 程序性查询，询问如何完成某个任务或过程
- code: 代码相关查询，编程问题、调试、代码解释等
- creative: 创造性查询，需要生成创意内容、想法或解决方案
- exploratory: 探索性查询，开放性讨论、概念解释等

请以JSON格式返回结果：{"intent": "类型", "confidence": 0.85, "reasoning": "分类理由"}"""

        user_prompt = f"""{context_str}当前查询：{query}

请分析这个查询的意图类型。"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            intent_str = result.get("intent", "factual")
            confidence = result.get("confidence", 0.5)
            
            # 验证意图类型有效性
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.FACTUAL
                confidence = 0.3
            
            return intent, min(max(confidence, 0.0), 1.0)
            
        except Exception:
            # 失败时使用规则方法作为后备
            return self._rule_based_intent_classification(query)

    def _rule_based_intent_classification(self, query: str) -> Tuple[QueryIntent, float]:
        """基于规则的意图分类（后备方法）"""
        query_lower = query.lower()
        
        # 代码相关关键词
        code_keywords = ['代码', 'bug', 'error', '函数', '类', 'class', 'function', 
                        'python', 'javascript', 'java', 'api', '接口', '调试']
        
        # 程序性关键词
        procedural_keywords = ['怎么', '如何', 'how', '步骤', '方法', '教程', 
                              '实现', '安装', '配置', '设置']
        
        # 创造性关键词
        creative_keywords = ['设计', '创建', '生成', '写', '制作', '建议', 
                           '想法', '方案', '优化', '改进']
        
        # 事实性关键词
        factual_keywords = ['什么', 'what', '哪些', '多少', '谁', 'who', 
                          '定义', '含义', '原理', '区别']
        
        scores = {
            QueryIntent.CODE: sum(1 for kw in code_keywords if kw in query_lower),
            QueryIntent.PROCEDURAL: sum(1 for kw in procedural_keywords if kw in query_lower),
            QueryIntent.CREATIVE: sum(1 for kw in creative_keywords if kw in query_lower),
            QueryIntent.FACTUAL: sum(1 for kw in factual_keywords if kw in query_lower),
        }
        
        # 获取最高分的意图
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]
        
        if max_score == 0:
            return QueryIntent.EXPLORATORY, 0.4
        
        confidence = min(max_score / 3.0, 0.8)
        return max_intent, confidence

    def _assess_complexity(self, query: str) -> float:
        """评估查询复杂度 (0-1)"""
        # 基础长度分数（字符长度）
        char_length = len(query)
        length_score = min(char_length / 100.0, 1.0)  # 100字符为满分
        
        # 词汇数量分数
        words = query.split()
        word_count_score = min(len(words) / 15.0, 1.0)  # 15个词为满分
        
        # 标点符号复杂度（表示句子结构复杂）
        punctuation_count = len(re.findall(r'[,，;；.。?？!！:：]', query))
        punctuation_score = min(punctuation_count / 5.0, 1.0)  # 5个标点为满分
        
        # 复合句和连词
        compound_words = ['和', '而且', '并且', '或者', '但是', '因为', '所以', '如果', '虽然']
        compound_count = sum(1 for word in compound_words if word in query)
        compound_score = min(compound_count / 3.0, 1.0)  # 3个连词为满分
        
        # 技术术语（大写字母开头的词、驼峰命名等）
        tech_terms = re.findall(r'[A-Z][a-zA-Z]*|[a-z]+[A-Z][a-zA-Z]*|\b[A-Z]{2,}\b', query)
        tech_score = min(len(tech_terms) / 3.0, 1.0)  # 3个技术术语为满分
        
        # 特殊符号（代码、URL等）
        special_symbols = len(re.findall(r'[(){}[\]<>/@#$%^&*+=|\\~`]', query))
        special_score = min(special_symbols / 5.0, 1.0)  # 5个特殊符号为满分
        
        # 加权计算总复杂度
        weights = {
            "length": 0.2,
            "word_count": 0.2, 
            "punctuation": 0.15,
            "compound": 0.15,
            "technical": 0.2,
            "special": 0.1
        }
        
        complexity = (
            length_score * weights["length"] +
            word_count_score * weights["word_count"] +
            punctuation_score * weights["punctuation"] +
            compound_score * weights["compound"] +
            tech_score * weights["technical"] +
            special_score * weights["special"]
        )
        
        return min(complexity, 1.0)

    def _extract_entities(self, query: str) -> List[str]:
        """提取实体"""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query)
                if matches:
                    # 处理可能的元组结果
                    if isinstance(matches[0], tuple):
                        matches = [match[0] if match[0] else match[1] for match in matches]
                    entities.extend(matches)
        
        # 去重并过滤
        return list(set([e.strip() for e in entities if len(e.strip()) > 1]))

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 中英文分词处理
        # 对于中文，提取连续的中文字符作为词汇
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', query)
        # 对于英文，提取单词
        english_words = re.findall(r'\b[a-zA-Z]+\b', query)
        
        all_words = []
        
        # 处理中文词汇（简单分割）
        for word in chinese_words:
            if len(word) > 1:
                # 简单处理：按2-4字符长度切分
                for i in range(len(word)):
                    for length in [2, 3, 4]:
                        if i + length <= len(word):
                            sub_word = word[i:i+length]
                            if sub_word not in self._stop_words:
                                all_words.append(sub_word)
        
        # 处理英文词汇
        for word in english_words:
            if word.lower() not in self._stop_words and len(word) > 2:
                all_words.append(word.lower())
        
        # 去重并返回
        return list(set(all_words))

    async def _identify_domain(self, query: str) -> Optional[str]:
        """识别查询领域"""
        domain_keywords = {
            "技术": ["代码", "编程", "开发", "系统", "数据库", "API"],
            "商业": ["管理", "营销", "销售", "财务", "策略"],
            "教育": ["学习", "教学", "课程", "培训", "知识"],
            "生活": ["健康", "旅行", "美食", "购物", "娱乐"],
            "科学": ["研究", "实验", "理论", "算法", "数据"],
        }
        
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return None

    def _analyze_sentiment(self, query: str) -> Optional[str]:
        """简单情感分析"""
        positive_words = ["好", "棒", "优秀", "喜欢", "满意", "感谢"]
        negative_words = ["差", "坏", "问题", "错误", "失败", "讨厌", "不行"]
        
        query_lower = query.lower()
        positive_score = sum(1 for w in positive_words if w in query_lower)
        negative_score = sum(1 for w in negative_words if w in query_lower)
        
        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        return "neutral"

    def _detect_language(self, query: str) -> str:
        """检测查询语言"""
        # 简单的中英文检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > 0:
            return "en"
        return "zh"  # 默认中文


class QueryContext:
    """查询上下文管理器"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.query_history: List[QueryAnalysis] = []
        self.session_start = utc_now()
    
    def add_query(self, analysis: QueryAnalysis):
        """添加查询到历史记录"""
        self.query_history.append(analysis)
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
    
    def get_context_for_query(self, current_query: str) -> List[str]:
        """获取与当前查询相关的上下文"""
        if not self.query_history:
            return []
        
        # 返回最近的查询文本
        return [q.query_text for q in self.query_history[-3:]]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        if not self.query_history:
            return {"total_queries": 0, "session_duration": 0}
        
        intent_counts = {}
        for q in self.query_history:
            intent_counts[q.intent_type.value] = intent_counts.get(q.intent_type.value, 0) + 1
        
        avg_complexity = sum(q.complexity_score for q in self.query_history) / len(self.query_history)
        session_duration = (utc_now() - self.session_start).total_seconds()
        
        return {
            "total_queries": len(self.query_history),
            "session_duration": session_duration,
            "intent_distribution": intent_counts,
            "average_complexity": avg_complexity,
            "languages": list(set(q.language for q in self.query_history))
        }