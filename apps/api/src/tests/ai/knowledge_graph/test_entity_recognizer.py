"""
实体识别器测试

测试多模型实体识别功能和模型融合逻辑
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from ai.knowledge_graph.entity_recognizer import (
    MultiModelEntityRecognizer,
    SpacyEntityRecognizer,
    TransformersEntityRecognizer,
    StanzaEntityRecognizer
)
from ai.knowledge_graph.data_models import Entity, EntityType

class TestSpacyEntityRecognizer:
    """spaCy实体识别器测试"""
    
    @pytest.fixture
    def recognizer(self):
        """创建spaCy识别器实例"""
        return SpacyEntityRecognizer()
    
    def test_recognizer_creation(self, recognizer):
        """测试识别器创建"""
        assert recognizer is not None
        assert not recognizer.is_loaded()
    
    @pytest.mark.asyncio
    async def test_initialization(self, recognizer):
        """测试初始化"""
        # Mock spacy加载
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            
            await recognizer.initialize()
            
            assert recognizer.is_loaded()
            mock_load.assert_called()
    
    @pytest.mark.asyncio
    async def test_extract_entities_english(self, recognizer):
        """测试英文实体识别"""
        # Mock spacy模型
        mock_doc = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "Apple Inc."
        mock_ent1.label_ = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 10
        
        mock_ent2 = Mock()
        mock_ent2.text = "John Smith"
        mock_ent2.label_ = "PERSON"
        mock_ent2.start_char = 15
        mock_ent2.end_char = 25
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        recognizer.nlp_models = {"en": mock_nlp}
        recognizer.loaded = True
        
        text = "Apple Inc. and John Smith"
        entities = await recognizer.extract_entities(text, "en")
        
        assert len(entities) == 2
        
        # 检查第一个实体
        assert entities[0].text == "Apple Inc."
        assert entities[0].label == EntityType.ORGANIZATION
        assert entities[0].start == 0
        assert entities[0].end == 10
        assert entities[0].confidence > 0
        
        # 检查第二个实体
        assert entities[1].text == "John Smith"
        assert entities[1].label == EntityType.PERSON
        assert entities[1].start == 15
        assert entities[1].end == 25
    
    @pytest.mark.asyncio
    async def test_extract_entities_chinese(self, recognizer):
        """测试中文实体识别"""
        # Mock中文实体
        mock_doc = Mock()
        mock_ent = Mock()
        mock_ent.text = "苹果公司"
        mock_ent.label_ = "ORG"
        mock_ent.start_char = 0
        mock_ent.end_char = 4
        
        mock_doc.ents = [mock_ent]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        recognizer.nlp_models = {"zh": mock_nlp}
        recognizer.loaded = True
        
        text = "苹果公司在美国"
        entities = await recognizer.extract_entities(text, "zh")
        
        assert len(entities) == 1
        assert entities[0].text == "苹果公司"
        assert entities[0].label == EntityType.ORGANIZATION
    
    def test_map_spacy_label(self, recognizer):
        """测试spaCy标签映射"""
        assert recognizer._map_spacy_label("PERSON") == EntityType.PERSON
        assert recognizer._map_spacy_label("ORG") == EntityType.ORGANIZATION
        assert recognizer._map_spacy_label("GPE") == EntityType.GPE
        assert recognizer._map_spacy_label("UNKNOWN") == EntityType.MISC

class TestTransformersEntityRecognizer:
    """Transformers实体识别器测试"""
    
    @pytest.fixture
    def recognizer(self):
        """创建Transformers识别器实例"""
        return TransformersEntityRecognizer()
    
    @pytest.mark.asyncio
    async def test_initialization(self, recognizer):
        """测试初始化"""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModelForTokenClassification.from_pretrained') as mock_model, \
             patch('transformers.pipeline') as mock_pipeline:
            
            mock_pipeline.return_value = Mock()
            
            await recognizer.initialize()
            
            assert recognizer.is_loaded()
            mock_tokenizer.assert_called()
            mock_model.assert_called()
            mock_pipeline.assert_called()
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, recognizer):
        """测试实体抽取"""
        # Mock pipeline结果
        mock_result = [
            {
                "entity": "B-PER",
                "score": 0.9998,
                "index": 1,
                "word": "John",
                "start": 0,
                "end": 4
            },
            {
                "entity": "I-PER", 
                "score": 0.9997,
                "index": 2,
                "word": "Smith",
                "start": 5,
                "end": 10
            },
            {
                "entity": "B-ORG",
                "score": 0.9995,
                "index": 4,
                "word": "Apple",
                "start": 20,
                "end": 25
            }
        ]
        
        mock_pipeline = Mock()
        mock_pipeline.return_value = mock_result
        recognizer.pipelines = {"default": mock_pipeline}
        recognizer.loaded = True
        
        text = "John Smith works at Apple"
        entities = await recognizer.extract_entities(text)
        
        assert len(entities) == 2  # John Smith 和 Apple
        
        # 检查聚合的实体
        person_entity = next(e for e in entities if e.label == EntityType.PERSON)
        assert person_entity.text == "John Smith"
        assert person_entity.start == 0
        assert person_entity.end == 10
        
        org_entity = next(e for e in entities if e.label == EntityType.ORGANIZATION)
        assert org_entity.text == "Apple"
        assert org_entity.start == 20
        assert org_entity.end == 25
    
    def test_aggregate_tokens(self, recognizer):
        """测试token聚合"""
        tokens = [
            {"entity": "B-PER", "word": "John", "start": 0, "end": 4, "score": 0.99},
            {"entity": "I-PER", "word": "Smith", "start": 5, "end": 10, "score": 0.98},
            {"entity": "B-ORG", "word": "Apple", "start": 15, "end": 20, "score": 0.97}
        ]
        
        entities = recognizer._aggregate_tokens(tokens, "John Smith works at Apple")
        
        assert len(entities) == 2
        
        # 检查聚合的人名
        person = entities[0]
        assert person.text == "John Smith"
        assert person.start == 0
        assert person.end == 10
        assert person.label == EntityType.PERSON
        
        # 检查组织名
        org = entities[1]
        assert org.text == "Apple"
        assert org.start == 15
        assert org.end == 20
        assert org.label == EntityType.ORGANIZATION

class TestStanzaEntityRecognizer:
    """Stanza实体识别器测试"""
    
    @pytest.fixture
    def recognizer(self):
        """创建Stanza识别器实例"""
        return StanzaEntityRecognizer()
    
    @pytest.mark.asyncio
    async def test_initialization(self, recognizer):
        """测试初始化"""
        with patch('stanza.Pipeline') as mock_pipeline:
            mock_nlp = Mock()
            mock_pipeline.return_value = mock_nlp
            
            await recognizer.initialize()
            
            assert recognizer.is_loaded()
            mock_pipeline.assert_called()
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, recognizer):
        """测试实体抽取"""
        # Mock Stanza文档结构
        mock_token1 = Mock()
        mock_token1.text = "Apple"
        mock_token1.ner = "B-ORG"
        mock_token1.start_char = 0
        mock_token1.end_char = 5
        
        mock_token2 = Mock()
        mock_token2.text = "Inc"
        mock_token2.ner = "I-ORG"
        mock_token2.start_char = 6
        mock_token2.end_char = 9
        
        mock_token3 = Mock()
        mock_token3.text = "John"
        mock_token3.ner = "B-PERSON"
        mock_token3.start_char = 15
        mock_token3.end_char = 19
        
        mock_sentence = Mock()
        mock_sentence.tokens = [
            Mock(words=[mock_token1]),
            Mock(words=[mock_token2]),
            Mock(words=[mock_token3])
        ]
        
        mock_doc = Mock()
        mock_doc.sentences = [mock_sentence]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        recognizer.pipelines = {"en": mock_nlp}
        recognizer.loaded = True
        
        text = "Apple Inc and John"
        entities = await recognizer.extract_entities(text, "en")
        
        assert len(entities) == 2
        
        # 检查组织实体
        org_entity = next(e for e in entities if e.label == EntityType.ORGANIZATION)
        assert org_entity.text == "Apple Inc"
        assert org_entity.start == 0
        assert org_entity.end == 9
        
        # 检查人物实体
        person_entity = next(e for e in entities if e.label == EntityType.PERSON)
        assert person_entity.text == "John"
        assert person_entity.start == 15
        assert person_entity.end == 19

class TestMultiModelEntityRecognizer:
    """多模型实体识别器测试"""
    
    @pytest.fixture
    def recognizer(self):
        """创建多模型识别器实例"""
        return MultiModelEntityRecognizer()
    
    @pytest.mark.asyncio
    async def test_initialization(self, recognizer):
        """测试初始化"""
        with patch.object(SpacyEntityRecognizer, 'initialize') as mock_spacy_init, \
             patch.object(TransformersEntityRecognizer, 'initialize') as mock_trans_init, \
             patch.object(StanzaEntityRecognizer, 'initialize') as mock_stanza_init:
            
            await recognizer.initialize()
            
            assert len(recognizer.recognizers) > 0
    
    @pytest.mark.asyncio
    async def test_extract_entities_with_fusion(self, recognizer):
        """测试多模型融合实体抽取"""
        # Mock各个识别器的结果
        spacy_entities = [
            Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.9),
            Entity("John Smith", EntityType.PERSON, 15, 25, 0.85)
        ]
        
        transformers_entities = [
            Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.95),
            Entity("John Smith", EntityType.PERSON, 15, 25, 0.88),
            Entity("California", EntityType.LOCATION, 30, 40, 0.82)
        ]
        
        stanza_entities = [
            Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.92),
            Entity("California", EntityType.GPE, 30, 40, 0.89)
        ]
        
        # Mock识别器
        mock_spacy = Mock()
        mock_spacy.is_loaded.return_value = True
        mock_spacy.extract_entities = AsyncMock(return_value=spacy_entities)
        
        mock_transformers = Mock()
        mock_transformers.is_loaded.return_value = True
        mock_transformers.extract_entities = AsyncMock(return_value=transformers_entities)
        
        mock_stanza = Mock()
        mock_stanza.is_loaded.return_value = True
        mock_stanza.extract_entities = AsyncMock(return_value=stanza_entities)
        
        recognizer.recognizers = [mock_spacy, mock_transformers, mock_stanza]
        
        text = "Apple Inc. and John Smith in California"
        entities = await recognizer.extract_entities(text)
        
        # 验证融合结果
        assert len(entities) >= 3  # 至少包含3个不同的实体
        
        # 检查Apple Inc.的融合结果（所有模型都识别到）
        apple_entities = [e for e in entities if e.text == "Apple Inc."]
        assert len(apple_entities) == 1
        apple_entity = apple_entities[0]
        assert apple_entity.confidence > 0.9  # 融合后置信度应该更高
        
        # 检查California的融合结果（两个模型识别到，但类型不同）
        california_entities = [e for e in entities if "California" in e.text]
        assert len(california_entities) >= 1
    
    def test_merge_entities(self, recognizer):
        """测试实体合并逻辑"""
        model_results = [
            [
                Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.9),
                Entity("John Smith", EntityType.PERSON, 15, 25, 0.85)
            ],
            [
                Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.95),
                Entity("John Smith", EntityType.PERSON, 15, 25, 0.88),
                Entity("California", EntityType.LOCATION, 30, 40, 0.82)
            ],
            [
                Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.92)
            ]
        ]
        
        merged_entities = recognizer._merge_entities(model_results, 0.5)
        
        # 验证合并结果
        entity_texts = [e.text for e in merged_entities]
        assert "Apple Inc." in entity_texts
        assert "John Smith" in entity_texts
        assert "California" in entity_texts
        
        # 检查Apple Inc.的融合置信度
        apple_entity = next(e for e in merged_entities if e.text == "Apple Inc.")
        assert apple_entity.confidence > 0.9  # 应该是多个模型结果的加权平均
    
    def test_calculate_overlap(self, recognizer):
        """测试重叠计算"""
        entity1 = Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.9)
        entity2 = Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 0.95)
        entity3 = Entity("Apple", EntityType.ORGANIZATION, 0, 5, 0.8)
        entity4 = Entity("Google", EntityType.ORGANIZATION, 20, 26, 0.9)
        
        # 完全重叠
        overlap1 = recognizer._calculate_overlap(entity1, entity2)
        assert overlap1 > 0.9
        
        # 部分重叠
        overlap2 = recognizer._calculate_overlap(entity1, entity3)
        assert 0.3 < overlap2 < 0.7
        
        # 无重叠
        overlap3 = recognizer._calculate_overlap(entity1, entity4)
        assert overlap3 == 0.0
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, recognizer):
        """测试置信度阈值过滤"""
        # Mock识别器结果
        mock_entities = [
            Entity("High Conf", EntityType.PERSON, 0, 9, 0.95),
            Entity("Medium Conf", EntityType.ORGANIZATION, 10, 21, 0.75),
            Entity("Low Conf", EntityType.LOCATION, 22, 30, 0.45)
        ]
        
        mock_recognizer = Mock()
        mock_recognizer.is_loaded.return_value = True
        mock_recognizer.extract_entities = AsyncMock(return_value=mock_entities)
        
        recognizer.recognizers = [mock_recognizer]
        
        # 测试不同阈值
        entities_high = await recognizer.extract_entities("test", confidence_threshold=0.8)
        assert len(entities_high) == 1  # 只有High Conf满足
        
        entities_medium = await recognizer.extract_entities("test", confidence_threshold=0.7)
        assert len(entities_medium) == 2  # High Conf和Medium Conf满足
        
        entities_low = await recognizer.extract_entities("test", confidence_threshold=0.4)
        assert len(entities_low) == 3  # 所有实体都满足

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
