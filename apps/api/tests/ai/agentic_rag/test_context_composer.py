"""
上下文组合器单元测试

测试上下文组合器的各项功能：
- 知识片段提取和分类
- 片段相关性评分和排序  
- 片段去重和多样性控制
- 上下文长度优化和信息密度平衡
- 知识片段间逻辑关系分析
- 不同组合策略的效果
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.ai.agentic_rag.context_composer import (
    ContextComposer, KnowledgeFragment, FragmentRelationship,
    ComposedContext, FragmentType, RelationshipType
)
from src.ai.agentic_rag.query_analyzer import QueryAnalysis, QueryIntent
from src.ai.agentic_rag.result_validator import ValidationResult, QualityScore, QualityDimension
from src.ai.agentic_rag.retrieval_agents import RetrievalResult, RetrievalStrategy


@pytest.fixture
def context_composer():
    """创建上下文组合器实例"""
    return ContextComposer()


@pytest.fixture
def sample_query_analysis():
    """创建示例查询分析"""
    return QueryAnalysis(
        query_text="Python机器学习库的使用方法",
        intent_type=QueryIntent.CODE,
        confidence=0.8,
        complexity_score=0.7,
        entities=["Python", "机器学习"],
        keywords=["Python", "机器学习", "库", "使用", "方法"],
        domain="技术",
        sentiment="neutral",
        language="zh"
    )


@pytest.fixture
def sample_validation_result():
    """创建示例验证结果"""
    retrieval_results = [
        RetrievalResult(
            agent_type=RetrievalStrategy.SEMANTIC,
            query="Python机器学习库的使用方法",
            results=[
                {
                    "id": "ml_guide_1",
                    "score": 0.9,
                    "content": """# 机器学习基础概念

机器学习是人工智能的一个重要分支，它让计算机能够从数据中学习模式和规律。Python作为最受欢迎的机器学习编程语言，提供了丰富的库和工具。

主要的机器学习库包括：
- scikit-learn：通用机器学习库
- TensorFlow：深度学习框架  
- PyTorch：动态深度学习框架
- pandas：数据处理库
- numpy：数值计算库""",
                    "file_path": "/docs/ml_guide.md",
                    "file_type": "markdown",
                    "chunk_index": 0
                },
                {
                    "id": "sklearn_tutorial_1", 
                    "score": 0.85,
                    "content": """```python
# 使用scikit-learn进行机器学习的基本步骤

# 1. 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. 加载数据
data = load_iris()
X, y = data.data, data.target

# 3. 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 创建模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. 训练模型
model.fit(X_train, y_train)

# 6. 预测和评估
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"模型准确率: {accuracy:.2f}")
```""",
                    "file_path": "/code/sklearn_example.py",
                    "file_type": "python",
                    "chunk_index": 0
                },
                {
                    "id": "data_preprocessing_1",
                    "score": 0.8,
                    "content": """数据预处理是机器学习中的关键步骤，通常包括以下几个方面：

1. **数据清洗**：处理缺失值、异常值和重复数据
2. **特征工程**：创建新特征、选择重要特征  
3. **数据标准化**：将数据缩放到相同的范围
4. **数据编码**：处理分类变量

使用pandas进行数据预处理的示例：
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv('data.csv')

# 处理缺失值
df = df.fillna(df.mean())

# 特征标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))
```""",
                    "file_path": "/docs/data_preprocessing.md",
                    "file_type": "markdown",
                    "chunk_index": 0
                }
            ],
            score=0.85,
            confidence=0.9,
            processing_time=0.1,
            explanation="语义检索结果"
        ),
        RetrievalResult(
            agent_type=RetrievalStrategy.KEYWORD,
            query="Python机器学习库的使用方法",
            results=[
                {
                    "id": "python_ml_basics",
                    "score": 0.75,
                    "content": """Python机器学习的优势包括：
- 简单易学的语法
- 丰富的第三方库生态
- 强大的数据处理能力
- 优秀的可视化工具
- 活跃的社区支持

常用的机器学习工作流程：
1. 问题定义和数据收集
2. 数据探索和可视化
3. 数据预处理和特征工程
4. 模型选择和训练
5. 模型评估和优化
6. 模型部署和监控""",
                    "file_path": "/docs/python_ml_overview.md",
                    "file_type": "markdown",
                    "bm25_score": 2.5,
                    "keyword_matches": 6
                }
            ],
            score=0.75,
            confidence=0.8,
            processing_time=0.2,
            explanation="关键词检索结果"
        )
    ]
    
    quality_scores = {
        QualityDimension.RELEVANCE: QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=0.85,
            confidence=0.9,
            explanation="高相关性"
        )
    }
    
    return ValidationResult(
        query_id="test_query_1",
        retrieval_results=retrieval_results,
        quality_scores=quality_scores,
        conflicts=[],
        overall_quality=0.85,
        overall_confidence=0.85,
        recommendations=[],
        validation_time=0.5
    )


@pytest.fixture
def sample_fragments():
    """创建示例知识片段"""
    return [
        KnowledgeFragment(
            id="frag_1",
            content="机器学习是人工智能的一个重要分支",
            source="/docs/ml_intro.md",
            fragment_type=FragmentType.DEFINITION,
            relevance_score=0.9,
            quality_score=0.8,
            information_density=0.7,
            tokens=50
        ),
        KnowledgeFragment(
            id="frag_2", 
            content="```python\nfrom sklearn import datasets\ndata = datasets.load_iris()\n```",
            source="/code/example.py",
            fragment_type=FragmentType.CODE,
            relevance_score=0.85,
            quality_score=0.9,
            information_density=0.8,
            tokens=30
        ),
        KnowledgeFragment(
            id="frag_3",
            content="例如，可以使用scikit-learn库进行分类任务",
            source="/docs/examples.md",
            fragment_type=FragmentType.EXAMPLE,
            relevance_score=0.8,
            quality_score=0.75,
            information_density=0.6,
            tokens=40
        )
    ]


class TestContextComposer:
    """上下文组合器基础功能测试"""

    def test_composer_initialization(self, context_composer):
        """测试组合器初始化"""
        assert context_composer.max_context_tokens == 4000
        assert context_composer.min_fragment_tokens == 20
        assert context_composer.diversity_threshold == 0.3
        assert len(context_composer.fragment_type_weights) == 7

    @pytest.mark.asyncio
    async def test_compose_context_success(self, context_composer, sample_query_analysis, sample_validation_result):
        """测试成功的上下文组合"""
        result = await context_composer.compose_context(
            sample_query_analysis,
            sample_validation_result,
            max_tokens=2000,
            composition_strategy="balanced"
        )
        
        assert isinstance(result, ComposedContext)
        assert result.query_id == "test_query_1"
        assert len(result.selected_fragments) > 0
        assert result.total_tokens <= 2000
        assert 0.0 <= result.information_density <= 1.0
        assert 0.0 <= result.diversity_score <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert result.composition_strategy == "balanced"
        assert isinstance(result.optimization_metrics, dict)

    @pytest.mark.asyncio
    async def test_compose_context_different_strategies(self, context_composer, sample_query_analysis, sample_validation_result):
        """测试不同的组合策略"""
        strategies = ["balanced", "relevance_first", "diversity_first"]
        
        for strategy in strategies:
            result = await context_composer.compose_context(
                sample_query_analysis,
                sample_validation_result,
                max_tokens=1500,
                composition_strategy=strategy
            )
            
            assert result.composition_strategy == strategy
            assert len(result.selected_fragments) > 0
            assert result.total_tokens <= 1500

    @pytest.mark.asyncio
    async def test_compose_context_small_token_limit(self, context_composer, sample_query_analysis, sample_validation_result):
        """测试小token限制下的组合"""
        result = await context_composer.compose_context(
            sample_query_analysis,
            sample_validation_result,
            max_tokens=200,  # 很小的限制
            composition_strategy="balanced"
        )
        
        assert result.total_tokens <= 200
        assert len(result.selected_fragments) >= 1  # 至少选择一个片段


class TestFragmentExtraction:
    """片段提取测试"""

    @pytest.mark.asyncio
    async def test_extract_fragments_success(self, context_composer, sample_validation_result):
        """测试成功提取片段"""
        fragments = await context_composer._extract_fragments(sample_validation_result)
        
        assert len(fragments) == 4  # 3个语义检索结果 + 1个关键词检索结果
        
        for fragment in fragments:
            assert isinstance(fragment, KnowledgeFragment)
            assert len(fragment.content) > 0
            assert fragment.tokens > 0
            assert fragment.fragment_type in FragmentType
            assert 0.0 <= fragment.relevance_score <= 1.0
            assert 0.0 <= fragment.information_density <= 1.0

    def test_classify_fragment_type_code(self, context_composer):
        """测试代码片段分类"""
        result_item = {
            "content": "```python\ndef hello():\n    print('Hello')\n```",
            "file_type": "python"
        }
        
        fragment_type = context_composer._classify_fragment_type(
            result_item["content"], result_item
        )
        
        assert fragment_type == FragmentType.CODE

    def test_classify_fragment_type_definition(self, context_composer):
        """测试定义片段分类"""
        result_item = {
            "content": "什么是机器学习？机器学习是一种人工智能技术...",
            "file_type": "markdown"
        }
        
        fragment_type = context_composer._classify_fragment_type(
            result_item["content"], result_item
        )
        
        assert fragment_type == FragmentType.DEFINITION

    def test_classify_fragment_type_procedure(self, context_composer):
        """测试步骤片段分类"""
        result_item = {
            "content": "第一步：安装必要的库\n第二步：加载数据\n第三步：训练模型",
            "file_type": "markdown"
        }
        
        fragment_type = context_composer._classify_fragment_type(
            result_item["content"], result_item
        )
        
        assert fragment_type == FragmentType.PROCEDURE

    def test_estimate_tokens(self, context_composer):
        """测试token数量估算"""
        test_cases = [
            ("Hello world", 2),  # 英文
            ("你好世界", 4),      # 中文
            ("Hello 世界 test", 5),  # 混合
            ("```python\nprint('hello')\n```", 8)  # 代码
        ]
        
        for text, expected_min in test_cases:
            tokens = context_composer._estimate_tokens(text)
            assert tokens >= expected_min
            assert isinstance(tokens, int)

    def test_calculate_information_density(self, context_composer):
        """测试信息密度计算"""
        test_cases = [
            ("短文本", 0.2, 0.8),  # 简单文本，密度较低
            ("这是一个包含多种结构的文档：\n\n# 标题\n\n- 列表项1\n- 列表项2\n\n```code```", 0.4, 1.0),  # 结构化文档，密度较高
            ("重复重复重复重复重复", 0.0, 0.3),  # 重复内容，密度很低
        ]
        
        for content, min_density, max_density in test_cases:
            density = context_composer._calculate_information_density(content)
            assert min_density <= density <= max_density
            assert 0.0 <= density <= 1.0


class TestFragmentScoring:
    """片段评分测试"""

    @pytest.mark.asyncio
    async def test_score_fragments(self, context_composer, sample_query_analysis, sample_fragments):
        """测试片段评分"""
        scored_fragments = await context_composer._score_fragments(sample_query_analysis, sample_fragments)
        
        assert len(scored_fragments) == len(sample_fragments)
        
        # 验证排序（按相关性分数降序）
        for i in range(1, len(scored_fragments)):
            assert scored_fragments[i-1].relevance_score >= scored_fragments[i].relevance_score

    def test_calculate_intent_bonus_code_query(self, context_composer):
        """测试代码查询的意图加成"""
        query_analysis = QueryAnalysis(
            query_text="Python函数示例",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.5,
            entities=[],
            keywords=["Python", "函数"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        code_fragment = KnowledgeFragment(
            id="test",
            content="def example(): pass",
            source="test.py",
            fragment_type=FragmentType.CODE,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=20
        )
        
        bonus = context_composer._calculate_intent_bonus(query_analysis, code_fragment)
        assert bonus == 0.3  # CODE意图对CODE片段的最高加成

    def test_calculate_intent_bonus_factual_query(self, context_composer):
        """测试事实查询的意图加成"""
        query_analysis = QueryAnalysis(
            query_text="什么是机器学习",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=["机器学习"],
            keywords=["什么是", "机器学习"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        definition_fragment = KnowledgeFragment(
            id="test",
            content="机器学习是一种人工智能技术",
            source="test.md",
            fragment_type=FragmentType.DEFINITION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=30
        )
        
        bonus = context_composer._calculate_intent_bonus(query_analysis, definition_fragment)
        assert bonus == 0.3  # FACTUAL意图对DEFINITION片段的最高加成

    @pytest.mark.asyncio
    async def test_calculate_content_quality_bonus(self, context_composer):
        """测试内容质量加成计算"""
        # 高质量内容（包含代码、列表、适中长度）
        high_quality_fragment = KnowledgeFragment(
            id="test1",
            content="""# 示例标题

这是一个包含多种元素的高质量片段：

- 列表项1
- 列表项2

```python
def example():
    return "Hello"
```

包含具体数据：准确率达到95%。""",
            source="test.md",
            fragment_type=FragmentType.EXPLANATION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=150
        )
        
        bonus = await context_composer._calculate_content_quality_bonus(high_quality_fragment)
        assert bonus > 0.1  # 应该有较高的质量加成
        
        # 低质量内容（过短）
        low_quality_fragment = KnowledgeFragment(
            id="test2",
            content="短",
            source="test.md",
            fragment_type=FragmentType.CONTEXT,
            relevance_score=0.5,
            quality_score=0.5,
            information_density=0.3,
            tokens=5
        )
        
        bonus = await context_composer._calculate_content_quality_bonus(low_quality_fragment)
        assert bonus <= 0.0  # 应该有惩罚

    def test_calculate_keyword_match_bonus(self, context_composer, sample_query_analysis):
        """测试关键词匹配加成"""
        fragment_with_keywords = KnowledgeFragment(
            id="test",
            content="Python机器学习库的使用方法和最佳实践",
            source="test.md",
            fragment_type=FragmentType.EXPLANATION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=50
        )
        
        bonus = context_composer._calculate_keyword_match_bonus(sample_query_analysis, fragment_with_keywords)
        assert bonus > 0.0  # 应该有加成
        
        fragment_without_keywords = KnowledgeFragment(
            id="test2",
            content="这是完全不相关的内容",
            source="test.md",
            fragment_type=FragmentType.CONTEXT,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=30
        )
        
        bonus_no_match = context_composer._calculate_keyword_match_bonus(sample_query_analysis, fragment_without_keywords)
        assert bonus_no_match < bonus  # 无匹配的加成应该更低


class TestRelationshipAnalysis:
    """关系分析测试"""

    @pytest.mark.asyncio
    async def test_analyze_relationships(self, context_composer, sample_fragments):
        """测试关系分析"""
        relationships = await context_composer._analyze_relationships(sample_fragments)
        
        assert isinstance(relationships, list)
        
        for relationship in relationships:
            assert isinstance(relationship, FragmentRelationship)
            assert relationship.fragment_a in [f.id for f in sample_fragments]
            assert relationship.fragment_b in [f.id for f in sample_fragments]
            assert relationship.relationship_type in RelationshipType
            assert 0.0 <= relationship.strength <= 1.0
            assert len(relationship.explanation) > 0

    def test_calculate_content_similarity(self, context_composer):
        """测试内容相似度计算"""
        content1 = "机器学习是人工智能的重要分支"
        content2 = "机器学习属于人工智能领域"
        content3 = "天气今天很好"
        
        similarity_high = context_composer._calculate_content_similarity(content1, content2)
        similarity_low = context_composer._calculate_content_similarity(content1, content3)
        
        assert similarity_high > similarity_low
        assert 0.0 <= similarity_high <= 1.0
        assert 0.0 <= similarity_low <= 1.0

    def test_detect_dependency(self, context_composer):
        """测试依赖关系检测"""
        definition_frag = KnowledgeFragment(
            id="def1",
            content="机器学习是一种人工智能技术",
            source="test.md",
            fragment_type=FragmentType.DEFINITION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=30
        )
        
        example_frag = KnowledgeFragment(
            id="ex1", 
            content="例如，可以使用机器学习进行图像识别",
            source="test.md",
            fragment_type=FragmentType.EXAMPLE,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=40
        )
        
        dependency = context_composer._detect_dependency(definition_frag, example_frag)
        assert dependency > 0.0  # 定义到示例应该有依赖关系

    def test_detect_sequence(self, context_composer):
        """测试顺序关系检测"""
        step1_frag = KnowledgeFragment(
            id="step1",
            content="第一步：准备数据和环境",
            source="test.md",
            fragment_type=FragmentType.PROCEDURE,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=30
        )
        
        step2_frag = KnowledgeFragment(
            id="step2",
            content="第二步：训练机器学习模型",
            source="test.md",
            fragment_type=FragmentType.PROCEDURE,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=40
        )
        
        sequence = context_composer._detect_sequence(step1_frag, step2_frag)
        assert sequence > 0.5  # 连续步骤应该有强顺序关系

    def test_detect_hierarchy(self, context_composer):
        """测试层次关系检测"""
        definition_frag = KnowledgeFragment(
            id="def1",
            content="机器学习定义",
            source="test.md",
            fragment_type=FragmentType.DEFINITION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=20
        )
        
        explanation_frag = KnowledgeFragment(
            id="exp1",
            content="机器学习的详细解释",
            source="test.md",
            fragment_type=FragmentType.EXPLANATION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=30
        )
        
        hierarchy = context_composer._detect_hierarchy(definition_frag, explanation_frag)
        assert hierarchy == 0.5  # 定义到解释有层次关系

    def test_detect_contrast(self, context_composer):
        """测试对比关系检测"""
        positive_frag = KnowledgeFragment(
            id="pos1",
            content="Python运行速度很快，是高效的语言",
            source="test.md",
            fragment_type=FragmentType.EXPLANATION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=40
        )
        
        negative_frag = KnowledgeFragment(
            id="neg1",
            content="Python运行速度不快，性能相对较低",
            source="test.md",
            fragment_type=FragmentType.EXPLANATION,
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=40
        )
        
        contrast = context_composer._detect_contrast(positive_frag, negative_frag)
        assert contrast > 0.2  # 对比内容应该有对比关系


class TestCompositionStrategies:
    """组合策略测试"""

    @pytest.mark.asyncio
    async def test_balanced_composition(self, context_composer, sample_query_analysis, sample_fragments):
        """测试平衡组合策略"""
        relationships = []  # 空关系列表
        
        selected = await context_composer._balanced_composition(
            sample_query_analysis, sample_fragments, relationships, max_tokens=200
        )
        
        assert len(selected) > 0
        assert sum(f.tokens for f in selected) <= 200
        
        # 验证去重 - 不应该有重复的片段
        selected_ids = [f.id for f in selected]
        assert len(selected_ids) == len(set(selected_ids))

    @pytest.mark.asyncio
    async def test_relevance_first_composition(self, context_composer, sample_fragments):
        """测试相关性优先组合策略"""
        # 确保片段按相关性排序
        sample_fragments.sort(key=lambda f: f.relevance_score, reverse=True)
        
        selected = await context_composer._relevance_first_composition(
            sample_fragments, max_tokens=150
        )
        
        assert len(selected) > 0
        assert sum(f.tokens for f in selected) <= 150
        
        # 验证选择了最高相关性的片段
        if len(selected) > 1:
            for i in range(1, len(selected)):
                assert selected[i-1].relevance_score >= selected[i].relevance_score

    @pytest.mark.asyncio
    async def test_diversity_first_composition(self, context_composer, sample_fragments):
        """测试多样性优先组合策略"""
        relationships = []  # 空关系列表
        
        selected = await context_composer._diversity_first_composition(
            sample_fragments, relationships, max_tokens=150
        )
        
        assert len(selected) > 0
        assert sum(f.tokens for f in selected) <= 150
        
        # 验证类型多样性
        if len(selected) > 1:
            selected_types = [f.fragment_type for f in selected]
            unique_types = set(selected_types)
            assert len(unique_types) >= min(len(selected), len(set(f.fragment_type for f in sample_fragments)))


class TestOptimization:
    """优化功能测试"""

    @pytest.mark.asyncio
    async def test_optimize_fragment_order(self, context_composer, sample_fragments):
        """测试片段顺序优化"""
        # 创建一些关系
        relationships = [
            FragmentRelationship(
                fragment_a=sample_fragments[0].id,
                fragment_b=sample_fragments[1].id,
                relationship_type=RelationshipType.DEPENDENCY,
                strength=0.7,
                explanation="依赖关系"
            )
        ]
        
        optimized = await context_composer._optimize_fragment_order(sample_fragments, relationships)
        
        assert len(optimized) == len(sample_fragments)
        assert set(f.id for f in optimized) == set(f.id for f in sample_fragments)

    def test_find_optimal_order_by_type(self, context_composer, sample_fragments):
        """测试基于类型的最优排序"""
        relation_graph = {}
        
        ordered = context_composer._find_optimal_order(sample_fragments, relation_graph)
        
        assert len(ordered) == len(sample_fragments)
        
        # 验证定义类型的片段排在前面
        definition_indices = [i for i, f in enumerate(ordered) if f.fragment_type == FragmentType.DEFINITION]
        if definition_indices:
            assert min(definition_indices) < len(ordered) / 2  # 定义应该在前半部分

    def test_remove_similar_fragments(self, context_composer):
        """测试移除相似片段"""
        fragments = [
            KnowledgeFragment(
                id="frag1",
                content="机器学习是人工智能的分支",
                source="test1.md",
                fragment_type=FragmentType.DEFINITION,
                relevance_score=0.9,
                quality_score=0.8,
                information_density=0.7,
                tokens=40
            ),
            KnowledgeFragment(
                id="frag2",
                content="机器学习属于人工智能领域",  # 相似内容
                source="test2.md", 
                fragment_type=FragmentType.DEFINITION,
                relevance_score=0.85,
                quality_score=0.8,
                information_density=0.7,
                tokens=40
            ),
            KnowledgeFragment(
                id="frag3",
                content="深度学习是机器学习的子集",  # 不同内容
                source="test3.md",
                fragment_type=FragmentType.DEFINITION,
                relevance_score=0.8,
                quality_score=0.8,
                information_density=0.7,
                tokens=40
            )
        ]
        
        reference = fragments[0]
        filtered = context_composer._remove_similar_fragments(fragments[1:], reference, threshold=0.5)
        
        # 应该移除相似的frag2，保留不同的frag3
        assert len(filtered) <= 1
        if filtered:
            assert filtered[0].id == "frag3"

    def test_is_too_similar_to_selected(self, context_composer):
        """测试相似度检查"""
        selected = [
            KnowledgeFragment(
                id="selected1",
                content="机器学习是人工智能技术",
                source="test.md",
                fragment_type=FragmentType.DEFINITION,
                relevance_score=0.9,
                quality_score=0.8,
                information_density=0.7,
                tokens=40
            )
        ]
        
        similar_candidate = KnowledgeFragment(
            id="candidate1",
            content="机器学习属于人工智能领域",  # 相似内容
            source="test2.md",
            fragment_type=FragmentType.DEFINITION,
            relevance_score=0.85,
            quality_score=0.8,
            information_density=0.7,
            tokens=40
        )
        
        different_candidate = KnowledgeFragment(
            id="candidate2", 
            content="天气预报显示明天会下雨",  # 不同内容
            source="test3.md",
            fragment_type=FragmentType.CONTEXT,
            relevance_score=0.3,
            quality_score=0.5,
            information_density=0.4,
            tokens=30
        )
        
        assert context_composer._is_too_similar_to_selected(similar_candidate, selected, threshold=0.3)
        assert not context_composer._is_too_similar_to_selected(different_candidate, selected, threshold=0.8)

    def test_calculate_diversity_with_fragment(self, context_composer, sample_fragments):
        """测试多样性分数计算"""
        selected = [sample_fragments[0]]  # 选择第一个片段
        
        # 测试与选择片段相似的候选
        similar_candidate = KnowledgeFragment(
            id="similar",
            content=sample_fragments[0].content,  # 相同内容
            source=sample_fragments[0].source,    # 相同来源
            fragment_type=sample_fragments[0].fragment_type,  # 相同类型
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=40
        )
        
        diversity_low = context_composer._calculate_diversity_with_fragment(selected, similar_candidate)
        
        # 测试与选择片段不同的候选
        different_candidate = KnowledgeFragment(
            id="different",
            content="完全不同的内容",
            source="/different/path.md",
            fragment_type=FragmentType.CODE,  # 不同类型
            relevance_score=0.8,
            quality_score=0.8,
            information_density=0.7,
            tokens=50
        )
        
        diversity_high = context_composer._calculate_diversity_with_fragment(selected, different_candidate)
        
        assert diversity_high > diversity_low
        assert 0.0 <= diversity_low <= 1.0
        assert 0.0 <= diversity_high <= 1.0


class TestMetricsCalculation:
    """指标计算测试"""

    def test_calculate_composition_metrics(self, context_composer, sample_fragments):
        """测试组合指标计算"""
        # 创建一些关系
        relationships = [
            FragmentRelationship(
                fragment_a=sample_fragments[0].id,
                fragment_b=sample_fragments[1].id,
                relationship_type=RelationshipType.SIMILARITY,
                strength=0.6,
                explanation="相似关系"
            )
        ]
        
        metrics = context_composer._calculate_composition_metrics(sample_fragments, relationships)
        
        assert isinstance(metrics, dict)
        assert "information_density" in metrics
        assert "diversity_score" in metrics
        assert "coherence_score" in metrics
        assert "coverage_score" in metrics
        
        # 验证指标范围
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0

    def test_calculate_composition_metrics_empty_fragments(self, context_composer):
        """测试空片段列表的指标计算"""
        metrics = context_composer._calculate_composition_metrics([], [])
        
        expected_keys = ["information_density", "diversity_score", "coherence_score", "coverage_score"]
        for key in expected_keys:
            assert key in metrics
            assert metrics[key] == 0.0

    def test_calculate_composition_metrics_no_relationships(self, context_composer, sample_fragments):
        """测试无关系情况下的指标计算"""
        metrics = context_composer._calculate_composition_metrics(sample_fragments, [])
        
        assert metrics["coherence_score"] == 0.0  # 无关系时连贯性为0
        assert metrics["information_density"] > 0.0  # 信息密度应该大于0
        assert metrics["diversity_score"] > 0.0     # 多样性应该大于0


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_compose_context_empty_results(self, context_composer, sample_query_analysis):
        """测试空检索结果的组合"""
        empty_validation_result = ValidationResult(
            query_id="empty_query",
            retrieval_results=[],
            quality_scores={},
            conflicts=[],
            overall_quality=0.0,
            overall_confidence=0.0,
            recommendations=[],
            validation_time=0.1
        )
        
        result = await context_composer.compose_context(
            sample_query_analysis,
            empty_validation_result
        )
        
        assert isinstance(result, ComposedContext)
        assert len(result.selected_fragments) == 0
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_compose_context_single_large_fragment(self, context_composer, sample_query_analysis):
        """测试单个超大片段的处理"""
        large_content = "Large content. " * 1000  # 很长的内容
        
        large_retrieval_result = RetrievalResult(
            agent_type=RetrievalStrategy.SEMANTIC,
            query="test",
            results=[{
                "id": "large_1",
                "score": 0.9,
                "content": large_content,
                "file_path": "/large.md",
                "file_type": "markdown"
            }],
            score=0.9,
            confidence=0.9,
            processing_time=0.1
        )
        
        validation_result = ValidationResult(
            query_id="large_query",
            retrieval_results=[large_retrieval_result],
            quality_scores={},
            conflicts=[],
            overall_quality=0.8,
            overall_confidence=0.8,
            recommendations=[],
            validation_time=0.1
        )
        
        result = await context_composer.compose_context(
            sample_query_analysis,
            validation_result,
            max_tokens=500  # 小于大片段的token数
        )
        
        assert isinstance(result, ComposedContext)
        # 应该仍然能处理，即使超过token限制


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])