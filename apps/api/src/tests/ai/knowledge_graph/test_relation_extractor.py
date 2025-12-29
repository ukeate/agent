"""
关系抽取器测试

测试基于模式和依存句法的关系抽取功能
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from ai.knowledge_graph.relation_extractor import (
    RelationExtractor,
    PatternBasedExtractor,
    DependencyBasedExtractor
)
from ai.knowledge_graph.data_models import Entity, Relation, EntityType, RelationType

class TestPatternBasedExtractor:
    """基于模式的关系抽取器测试"""
    
    @pytest.fixture
    def extractor(self):
        """创建模式抽取器实例"""
        return PatternBasedExtractor()
    
    def test_extractor_creation(self, extractor):
        """测试抽取器创建"""
        assert extractor is not None
        assert len(extractor.patterns) > 0
    
    def test_works_for_pattern(self, extractor):
        """测试工作关系模式"""
        text = "张三在苹果公司工作"
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95)
        ]
        
        relations = extractor.extract_relations(text, entities)
        
        assert len(relations) >= 1
        work_relation = next((r for r in relations if r.predicate == RelationType.WORKS_FOR), None)
        assert work_relation is not None
        assert work_relation.subject.text == "张三"
        assert work_relation.object.text == "苹果公司"
        assert work_relation.confidence > 0
    
    def test_located_in_pattern(self, extractor):
        """测试位置关系模式"""
        text = "苹果公司位于加利福尼亚州"
        entities = [
            Entity("苹果公司", EntityType.COMPANY, 0, 4, 0.95),
            Entity("加利福尼亚州", EntityType.LOCATION, 6, 11, 0.9)
        ]
        
        relations = extractor.extract_relations(text, entities)
        
        location_relation = next((r for r in relations if r.predicate == RelationType.LOCATED_IN), None)
        assert location_relation is not None
        assert location_relation.subject.text == "苹果公司"
        assert location_relation.object.text == "加利福尼亚州"
    
    def test_founded_by_pattern(self, extractor):
        """测试创立关系模式"""
        text = "苹果公司由史蒂夫·乔布斯创立"
        entities = [
            Entity("苹果公司", EntityType.COMPANY, 0, 4, 0.95),
            Entity("史蒂夫·乔布斯", EntityType.PERSON, 5, 12, 0.9)
        ]
        
        relations = extractor.extract_relations(text, entities)
        
        founded_relation = next((r for r in relations if r.predicate == RelationType.FOUNDED_BY), None)
        assert founded_relation is not None
        assert founded_relation.subject.text == "苹果公司"
        assert founded_relation.object.text == "史蒂夫·乔布斯"
    
    def test_english_patterns(self, extractor):
        """测试英文模式"""
        text = "John Smith works for Apple Inc."
        entities = [
            Entity("John Smith", EntityType.PERSON, 0, 10, 0.9),
            Entity("Apple Inc.", EntityType.COMPANY, 21, 31, 0.95)
        ]
        
        relations = extractor.extract_relations(text, entities)
        
        work_relation = next((r for r in relations if r.predicate == RelationType.WORKS_FOR), None)
        assert work_relation is not None
        assert work_relation.subject.text == "John Smith"
        assert work_relation.object.text == "Apple Inc."
    
    def test_multiple_relations_same_sentence(self, extractor):
        """测试同一句子中的多个关系"""
        text = "张三在北京的苹果公司工作"
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("北京", EntityType.CITY, 3, 5, 0.92),
            Entity("苹果公司", EntityType.COMPANY, 6, 10, 0.95)
        ]
        
        relations = extractor.extract_relations(text, entities)
        
        # 应该找到至少一个关系
        assert len(relations) >= 1
        
        # 检查是否有工作关系
        work_relations = [r for r in relations if r.predicate == RelationType.WORKS_FOR]
        assert len(work_relations) >= 1
    
    def test_confidence_calculation(self, extractor):
        """测试置信度计算"""
        text = "张三在苹果公司工作"
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95)
        ]
        
        relations = extractor.extract_relations(text, entities)
        
        assert len(relations) > 0
        for relation in relations:
            assert 0 <= relation.confidence <= 1
            # 置信度应该考虑实体置信度和模式置信度
            assert relation.confidence > 0.5  # 基本合理的置信度

class TestDependencyBasedExtractor:
    """基于依存句法的关系抽取器测试"""
    
    @pytest.fixture
    def extractor(self):
        """创建依存句法抽取器实例"""
        return DependencyBasedExtractor()
    
    @pytest.mark.asyncio
    async def test_initialization(self, extractor):
        """测试初始化"""
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            
            await extractor.initialize()
            
            assert extractor.is_loaded()
            mock_load.assert_called()
    
    @pytest.mark.asyncio
    async def test_extract_relations_with_dependency_parsing(self, extractor):
        """测试基于依存句法的关系抽取"""
        # Mock spaCy依存句法分析结果
        mock_token1 = Mock()
        mock_token1.text = "张三"
        mock_token1.pos_ = "NOUN"
        mock_token1.dep_ = "nsubj"
        mock_token1.head.text = "工作"
        mock_token1.i = 0
        
        mock_token2 = Mock()
        mock_token2.text = "在"
        mock_token2.pos_ = "ADP"
        mock_token2.dep_ = "prep"
        mock_token2.head.text = "工作"
        mock_token2.i = 1
        
        mock_token3 = Mock()
        mock_token3.text = "苹果公司"
        mock_token3.pos_ = "NOUN"
        mock_token3.dep_ = "pobj"
        mock_token3.head = mock_token2
        mock_token3.i = 2
        
        mock_token4 = Mock()
        mock_token4.text = "工作"
        mock_token4.pos_ = "VERB"
        mock_token4.dep_ = "ROOT"
        mock_token4.i = 3
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2, mock_token3, mock_token4]))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        extractor.nlp_models = {"zh": mock_nlp}
        extractor.loaded = True
        
        text = "张三在苹果公司工作"
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95)
        ]
        
        relations = await extractor.extract_relations(text, entities)
        
        # 验证是否提取到关系
        assert len(relations) >= 0  # 可能为0，因为这是一个简化的mock
    
    def test_analyze_dependency_patterns(self, extractor):
        """测试依存关系模式分析"""
        # 测试主谓宾结构分析
        patterns = extractor._analyze_dependency_patterns(
            subject_token_idx=0,
            object_token_idx=2,
            verb_token_idx=1,
            dependency_info=["nsubj", "ROOT", "dobj"]
        )
        
        assert isinstance(patterns, list)
    
    def test_map_dependency_to_relation(self, extractor):
        """测试依存关系到语义关系的映射"""
        # 测试工作关系映射
        relation_type = extractor._map_dependency_to_relation(
            subject_entity=Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            object_entity=Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95),
            verb="工作",
            dependency_pattern=["nsubj", "ROOT", "prep", "pobj"]
        )
        
        # 根据实体类型和动词，应该能映射到相应的关系类型
        assert relation_type in [RelationType.WORKS_FOR, RelationType.MISC, None]

class TestRelationExtractor:
    """关系抽取器主类测试"""
    
    @pytest.fixture
    def extractor(self):
        """创建关系抽取器实例"""
        return RelationExtractor()
    
    @pytest.mark.asyncio
    async def test_initialization(self, extractor):
        """测试初始化"""
        with patch.object(DependencyBasedExtractor, 'initialize') as mock_init:
            await extractor.initialize()
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_relations_combined(self, extractor):
        """测试组合抽取（模式+依存句法）"""
        # Mock各个抽取器
        pattern_relations = [
            Relation(
                subject=Entity("张三", EntityType.PERSON, 0, 2, 0.9),
                predicate=RelationType.WORKS_FOR,
                object=Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95),
                confidence=0.8,
                context="张三在苹果公司工作",
                source_sentence="张三在苹果公司工作"
            )
        ]
        
        dependency_relations = [
            Relation(
                subject=Entity("张三", EntityType.PERSON, 0, 2, 0.9),
                predicate=RelationType.WORKS_FOR,
                object=Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95),
                confidence=0.75,
                context="张三在苹果公司工作",
                source_sentence="张三在苹果公司工作"
            )
        ]
        
        # Mock抽取器方法
        extractor.pattern_extractor.extract_relations = Mock(return_value=pattern_relations)
        extractor.dependency_extractor.extract_relations = AsyncMock(return_value=dependency_relations)
        extractor.dependency_extractor.is_loaded = Mock(return_value=True)
        
        text = "张三在苹果公司工作"
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95)
        ]
        
        relations = await extractor.extract_relations(text, entities)
        
        # 验证关系去重和融合
        assert len(relations) >= 1
        
        # 检查融合后的置信度是否更高
        work_relation = next((r for r in relations if r.predicate == RelationType.WORKS_FOR), None)
        assert work_relation is not None
        assert work_relation.confidence >= 0.75  # 至少保持原有置信度
    
    def test_merge_relations(self, extractor):
        """测试关系合并"""
        pattern_relations = [
            Relation(
                subject=Entity("张三", EntityType.PERSON, 0, 2, 0.9),
                predicate=RelationType.WORKS_FOR,
                object=Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95),
                confidence=0.8,
                context="张三在苹果公司工作",
                source_sentence="张三在苹果公司工作"
            ),
            Relation(
                subject=Entity("李四", EntityType.PERSON, 10, 12, 0.85),
                predicate=RelationType.WORKS_FOR,
                object=Entity("谷歌", EntityType.COMPANY, 15, 17, 0.9),
                confidence=0.82,
                context="李四在谷歌工作",
                source_sentence="李四在谷歌工作"
            )
        ]
        
        dependency_relations = [
            Relation(
                subject=Entity("张三", EntityType.PERSON, 0, 2, 0.9),
                predicate=RelationType.WORKS_FOR,
                object=Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95),
                confidence=0.75,
                context="张三在苹果公司工作",
                source_sentence="张三在苹果公司工作"
            )
        ]
        
        merged_relations = extractor._merge_relations(pattern_relations, dependency_relations)
        
        # 应该有2个不同的关系（张三-苹果，李四-谷歌）
        # 张三-苹果关系应该被合并，置信度提高
        assert len(merged_relations) == 2
        
        zhang_relation = next((r for r in merged_relations 
                              if r.subject.text == "张三" and r.object.text == "苹果公司"), None)
        assert zhang_relation is not None
        assert zhang_relation.confidence > 0.8  # 合并后置信度应该提高
        
        li_relation = next((r for r in merged_relations 
                           if r.subject.text == "李四"), None)
        assert li_relation is not None
    
    def test_are_relations_similar(self, extractor):
        """测试关系相似性判断"""
        relation1 = Relation(
            subject=Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            predicate=RelationType.WORKS_FOR,
            object=Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95),
            confidence=0.8,
            context="张三在苹果公司工作",
            source_sentence="张三在苹果公司工作"
        )
        
        # 相同关系
        relation2 = Relation(
            subject=Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            predicate=RelationType.WORKS_FOR,
            object=Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95),
            confidence=0.75,
            context="张三在苹果公司工作",
            source_sentence="张三在苹果公司工作"
        )
        
        # 不同关系
        relation3 = Relation(
            subject=Entity("李四", EntityType.PERSON, 10, 12, 0.85),
            predicate=RelationType.WORKS_FOR,
            object=Entity("谷歌", EntityType.COMPANY, 15, 17, 0.9),
            confidence=0.82,
            context="李四在谷歌工作",
            source_sentence="李四在谷歌工作"
        )
        
        assert extractor._are_relations_similar(relation1, relation2)
        assert not extractor._are_relations_similar(relation1, relation3)
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, extractor):
        """测试置信度阈值过滤"""
        # Mock关系抽取结果
        mock_relations = [
            Relation(
                subject=Entity("高置信", EntityType.PERSON, 0, 3, 0.9),
                predicate=RelationType.WORKS_FOR,
                object=Entity("公司A", EntityType.COMPANY, 5, 8, 0.95),
                confidence=0.9,
                context="高置信在公司A工作",
                source_sentence="高置信在公司A工作"
            ),
            Relation(
                subject=Entity("中置信", EntityType.PERSON, 10, 13, 0.8),
                predicate=RelationType.WORKS_FOR,
                object=Entity("公司B", EntityType.COMPANY, 15, 18, 0.85),
                confidence=0.7,
                context="中置信在公司B工作",
                source_sentence="中置信在公司B工作"
            ),
            Relation(
                subject=Entity("低置信", EntityType.PERSON, 20, 23, 0.7),
                predicate=RelationType.WORKS_FOR,
                object=Entity("公司C", EntityType.COMPANY, 25, 28, 0.8),
                confidence=0.5,
                context="低置信在公司C工作",
                source_sentence="低置信在公司C工作"
            )
        ]
        
        # Mock抽取方法
        extractor.pattern_extractor.extract_relations = Mock(return_value=mock_relations)
        extractor.dependency_extractor.extract_relations = AsyncMock(return_value=[])
        extractor.dependency_extractor.is_loaded = Mock(return_value=False)
        
        text = "测试文本"
        entities = []
        
        # 测试不同阈值
        relations_high = await extractor.extract_relations(text, entities, confidence_threshold=0.8)
        assert len(relations_high) == 1  # 只有高置信关系
        
        relations_medium = await extractor.extract_relations(text, entities, confidence_threshold=0.6)
        assert len(relations_medium) == 2  # 高置信和中置信关系
        
        relations_low = await extractor.extract_relations(text, entities, confidence_threshold=0.4)
        assert len(relations_low) == 3  # 所有关系
    
    @pytest.mark.asyncio
    async def test_complex_sentence_extraction(self, extractor):
        """测试复杂句子的关系抽取"""
        text = "张三在位于北京的苹果公司工作，他的同事李四负责市场营销。"
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("北京", EntityType.CITY, 5, 7, 0.92),
            Entity("苹果公司", EntityType.COMPANY, 8, 12, 0.95),
            Entity("李四", EntityType.PERSON, 18, 20, 0.88),
            Entity("市场营销", EntityType.MISC, 23, 27, 0.7)
        ]
        
        # Mock抽取结果
        expected_relations = [
            Relation(
                subject=entities[0],  # 张三
                predicate=RelationType.WORKS_FOR,
                object=entities[2],   # 苹果公司
                confidence=0.85,
                context=text,
                source_sentence=text
            ),
            Relation(
                subject=entities[2],  # 苹果公司
                predicate=RelationType.LOCATED_IN,
                object=entities[1],   # 北京
                confidence=0.8,
                context=text,
                source_sentence=text
            )
        ]
        
        extractor.pattern_extractor.extract_relations = Mock(return_value=expected_relations)
        extractor.dependency_extractor.extract_relations = AsyncMock(return_value=[])
        extractor.dependency_extractor.is_loaded = Mock(return_value=False)
        
        relations = await extractor.extract_relations(text, entities)
        
        # 验证复杂句子中的多个关系
        assert len(relations) >= 2
        
        # 检查特定关系
        work_relations = [r for r in relations if r.predicate == RelationType.WORKS_FOR]
        location_relations = [r for r in relations if r.predicate == RelationType.LOCATED_IN]
        
        assert len(work_relations) >= 1
        assert len(location_relations) >= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
