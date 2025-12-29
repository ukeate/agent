"""
训练数据预处理系统

这个模块包含数据预处理的所有组件：
- 文本清理
- 数据去重
- 格式标准化
- 质量评估
- 数据丰富化
"""

import re
import hashlib
import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .core import DataRecord, ProcessingRule, QualityAssessor
from src.core.utils.timezone_utils import utc_now, utc_factory

from src.core.logging import get_logger
logger = get_logger(__name__)

class TextCleaningRule(ProcessingRule):
    """文本清理规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.remove_special_chars = self.config.get('remove_special_chars', True)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        self.fix_encoding = self.config.get('fix_encoding', True)
        self.custom_patterns = self.config.get('custom_patterns', [])
    
    async def apply(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """应用文本清理规则"""
        cleaned_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 10:  # 只清理长文本
                cleaned_data[key] = self._clean_text(value)
            elif isinstance(value, list):
                cleaned_data[key] = [
                    self._clean_text(str(item)) if isinstance(item, str) else item 
                    for item in value
                ]
            else:
                cleaned_data[key] = value
        
        return cleaned_data
    
    def _clean_text(self, text: str) -> str:
        """清理单个文本字段"""
        if not text or not isinstance(text, str):
            return str(text) if text is not None else ''
        
        # 修正编码问题
        if self.fix_encoding:
            text = text.replace('\ufffd', '')  # 移除替换字符
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # 标准化空白字符
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # 移除特殊字符(保留中文、英文、数字和常用标点)
        if self.remove_special_chars:
            # 保留中文、英文、数字、空格和常用标点符号
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()[\]{}"\'@#$%^&*+=<>/\-]', '', text)
        
        # 应用自定义清理模式
        for pattern, replacement in self.custom_patterns:
            text = re.sub(pattern, replacement, text)
        
        # 移除多余的空白
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 规范化段落分隔
        text = text.strip()
        
        return text

class DeduplicationRule(ProcessingRule):
    """数据去重规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.hash_fields = self.config.get('hash_fields', None)  # 指定用于计算哈希的字段
        self.similarity_threshold = self.config.get('similarity_threshold', 0.95)
        self.use_fuzzy_matching = self.config.get('use_fuzzy_matching', False)
    
    async def apply(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """应用去重规则"""
        # 计算内容哈希
        content_hash = self._calculate_content_hash(data)
        metadata['content_hash'] = content_hash
        metadata['deduplication_applied'] = True
        
        # 如果启用模糊匹配，计算文本相似度特征
        if self.use_fuzzy_matching:
            text_features = self._extract_text_features(data)
            metadata['text_features'] = text_features
        
        return data
    
    def _calculate_content_hash(self, data: Dict[str, Any]) -> str:
        """计算内容哈希值"""
        # 如果指定了哈希字段，只使用这些字段
        if self.hash_fields:
            hash_data = {k: v for k, v in data.items() if k in self.hash_fields}
        else:
            hash_data = data
        
        # 排序并序列化
        content = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_text_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取文本特征用于相似度匹配"""
        features = {}
        
        # 提取主要文本内容
        text_content = ""
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 20:
                text_content += f" {value}"
        
        if text_content:
            # 基本统计特征
            features['char_count'] = len(text_content)
            features['word_count'] = len(text_content.split())
            
            # 简单的n-gram特征
            words = text_content.lower().split()
            if len(words) >= 3:
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                features['trigram_sample'] = trigrams[:10]  # 只保留前10个
        
        return features

class FormatStandardizationRule(ProcessingRule):
    """格式标准化规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.field_mapping = self.config.get('field_mapping', {
            'title': ['title', 'name', 'subject', 'headline', 'header'],
            'content': ['content', 'text', 'body', 'description', 'message'],
            'author': ['author', 'creator', 'user', 'username', 'by'],
            'timestamp': ['timestamp', 'created_at', 'date', 'time', 'published'],
            'url': ['url', 'link', 'href', 'source_url'],
            'category': ['category', 'type', 'class', 'label', 'tag']
        })
        self.normalize_dates = self.config.get('normalize_dates', True)
        self.normalize_urls = self.config.get('normalize_urls', True)
    
    async def apply(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """应用格式标准化规则"""
        standardized_data = {}
        original_fields = set(data.keys())
        mapped_fields = set()
        
        # 映射标准字段
        for standard_field, possible_fields in self.field_mapping.items():
            for field in possible_fields:
                if field in data and standard_field not in standardized_data:
                    value = data[field]
                    
                    # 应用字段特定的标准化
                    if standard_field == 'timestamp' and self.normalize_dates:
                        value = self._normalize_timestamp(value)
                    elif standard_field == 'url' and self.normalize_urls:
                        value = self._normalize_url(value)
                    
                    standardized_data[standard_field] = value
                    mapped_fields.add(field)
                    break
        
        # 保留未映射的原始字段
        for key, value in data.items():
            if key not in mapped_fields:
                standardized_data[key] = value
        
        # 记录标准化信息
        metadata['standardization_applied'] = True
        metadata['mapped_fields'] = list(mapped_fields)
        metadata['original_field_count'] = len(original_fields)
        metadata['standardized_field_count'] = len(standardized_data)
        
        return standardized_data
    
    def _normalize_timestamp(self, timestamp: Any) -> Optional[str]:
        """标准化时间戳"""
        if not timestamp:
            return None
        
        try:
            if isinstance(timestamp, str):
                # 尝试解析常见的时间戳格式
                from datetime import datetime
                import dateutil.parser
                
                # 清理时间戳字符串
                timestamp = timestamp.strip()
                if timestamp.endswith('Z'):
                    timestamp = timestamp[:-1] + '+00:00'
                
                # 使用dateutil解析
                dt = dateutil.parser.parse(timestamp)
                return dt.isoformat()
            elif isinstance(timestamp, (int, float)):
                # 假设是Unix时间戳
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                return dt.isoformat()
        except Exception as e:
            logger.warning(f"Failed to normalize timestamp '{timestamp}': {e}")
        
        return str(timestamp)
    
    def _normalize_url(self, url: Any) -> Optional[str]:
        """标准化URL"""
        if not url or not isinstance(url, str):
            return str(url) if url else None
        
        url = url.strip()
        
        # 添加协议前缀
        if url and not url.startswith(('http://', 'https://', 'ftp://')):
            url = 'https://' + url
        
        # 移除尾部斜杠
        if url.endswith('/'):
            url = url[:-1]
        
        return url

class QualityFilteringRule(ProcessingRule):
    """质量过滤规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.required_fields = self.config.get('required_fields', [])
        self.min_content_length = self.config.get('min_content_length', 10)
        self.max_content_length = self.config.get('max_content_length', 50000)
        self.forbidden_patterns = self.config.get('forbidden_patterns', [])
        self.quality_thresholds = self.config.get('quality_thresholds', {})
    
    async def apply(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """应用质量过滤规则"""
        # 检查必要字段
        for field in self.required_fields:
            if field not in data or not data[field]:
                raise ValueError(f"Required field '{field}' is missing or empty")
        
        # 检查内容长度
        content_field = self._find_content_field(data)
        if content_field:
            content = data[content_field]
            if isinstance(content, str):
                if len(content) < self.min_content_length:
                    raise ValueError(f"Content too short: {len(content)} < {self.min_content_length}")
                
                if len(content) > self.max_content_length:
                    # 截断内容而不是拒绝
                    data[content_field] = content[:self.max_content_length] + '...[truncated]'
                    metadata['content_truncated'] = True
                    metadata['original_content_length'] = len(content)
        
        # 检查禁止的模式
        for pattern in self.forbidden_patterns:
            for key, value in data.items():
                if isinstance(value, str) and re.search(pattern, value, re.IGNORECASE):
                    raise ValueError(f"Forbidden pattern '{pattern}' found in field '{key}'")
        
        # 应用质量阈值检查
        for field, threshold in self.quality_thresholds.items():
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)) and value < threshold:
                    raise ValueError(f"Quality threshold not met for field '{field}': {value} < {threshold}")
        
        metadata['quality_filtering_applied'] = True
        return data
    
    def _find_content_field(self, data: Dict[str, Any]) -> Optional[str]:
        """查找主要内容字段"""
        content_candidates = ['content', 'text', 'body', 'description', 'message']
        for field in content_candidates:
            if field in data:
                return field
        return None

class DataEnrichmentRule(ProcessingRule):
    """数据丰富化规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.add_metadata = self.config.get('add_metadata', True)
        self.analyze_content = self.config.get('analyze_content', True)
        self.extract_keywords = self.config.get('extract_keywords', False)
        self.detect_language = self.config.get('detect_language', True)
        self.analyze_sentiment = self.config.get('analyze_sentiment', True)
    
    async def apply(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """应用数据丰富化规则"""
        enriched_data = data.copy()
        
        # 添加处理元数据
        if self.add_metadata:
            enriched_data['_metadata'] = {
                'processed_at': utc_now().isoformat(),
                'processing_version': '1.0',
                'enrichment_applied': True
            }
        
        # 内容分析
        if self.analyze_content:
            content_field = self._find_content_field(data)
            if content_field and isinstance(data[content_field], str):
                content = data[content_field]
                analysis = await self._analyze_content(content)
                enriched_data['_analysis'] = analysis
        
        return enriched_data
    
    def _find_content_field(self, data: Dict[str, Any]) -> Optional[str]:
        """查找主要内容字段"""
        content_candidates = ['content', 'text', 'body', 'description', 'message']
        for field in content_candidates:
            if field in data:
                return field
        return None
    
    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """分析文本内容"""
        analysis = {
            'character_count': len(content),
            'word_count': len(content.split()),
            'sentence_count': len(re.split(r'[.!?]+', content)),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
        }
        
        # 语言检测
        if self.detect_language:
            analysis['language'] = self._detect_language(content)
        
        # 情感分析
        if self.analyze_sentiment:
            analysis['sentiment'] = self._analyze_sentiment(content)
        
        # 关键词提取
        if self.extract_keywords:
            analysis['keywords'] = self._extract_keywords(content)
        
        return analysis
    
    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.3:
            return 'zh'
        elif english_ratio > 0.5:
            return 'en'
        else:
            return 'mixed'
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """简单的情感分析"""
        positive_words = [
            '好', '棒', '优秀', '喜欢', '满意', '开心', '高兴', '赞',
            'good', 'great', 'excellent', 'love', 'like', 'happy', 'satisfied', 'awesome'
        ]
        negative_words = [
            '坏', '差', '讨厌', '糟糕', '失望', '愤怒', '难过', '垃圾',
            'bad', 'terrible', 'hate', 'awful', 'disappointed', 'angry', 'sad', 'horrible'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        positive_ratio = positive_count / total_words if total_words > 0 else 0
        negative_ratio = negative_count / total_words if total_words > 0 else 0
        
        if positive_count > negative_count:
            polarity = 'positive'
        elif negative_count > positive_count:
            polarity = 'negative'
        else:
            polarity = 'neutral'
        
        return {
            'polarity': polarity,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'confidence': abs(positive_count - negative_count) / max(total_words, 1)
        }
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """简单的关键词提取"""
        # 移除标点符号并分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤常见停用词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            '的', '是', '在', '了', '和', '与', '或', '但', '不', '也', '就', '都', '很',
            '更', '最', '可以', '能够', '应该', '必须', '这', '那', '这个', '那个', '我',
            '你', '他', '她', '它', '我们', '你们', '他们'
        }
        
        # 过滤停用词并计算词频
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # 计算词频
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序并返回前N个关键词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]

class DataQualityAssessor(QualityAssessor):
    """数据质量评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.weights = self.config.get('weights', {
            'completeness': 0.3,
            'content_quality': 0.3,
            'structure': 0.2,
            'consistency': 0.2
        })
    
    def assess(self, data: Dict[str, Any]) -> float:
        """评估数据质量"""
        scores = {
            'completeness': self._assess_completeness(data),
            'content_quality': self._assess_content_quality(data),
            'structure': self._assess_structure(data),
            'consistency': self._assess_consistency(data)
        }
        
        # 计算加权平均分
        weighted_score = sum(
            scores[metric] * self.weights.get(metric, 0) 
            for metric in scores
        )
        
        return min(max(weighted_score, 0.0), 1.0)
    
    def _assess_completeness(self, data: Dict[str, Any]) -> float:
        """评估数据完整性"""
        important_fields = ['title', 'content', 'text', 'body', 'description']
        
        total_fields = len(important_fields)
        complete_fields = 0
        
        for field in important_fields:
            if field in data and data[field]:
                complete_fields += 1
        
        return complete_fields / total_fields if total_fields > 0 else 0.0
    
    def _assess_content_quality(self, data: Dict[str, Any]) -> float:
        """评估内容质量"""
        content_field = self._find_content_field(data)
        if not content_field:
            return 0.0
        
        content = data[content_field]
        if not isinstance(content, str):
            return 0.0
        
        length = len(content)
        word_count = len(content.split())
        
        # 基于长度的质量评分
        length_score = 0.0
        if 50 <= length <= 5000:
            length_score = 1.0
        elif 20 <= length <= 10000:
            length_score = 0.7
        elif length > 10:
            length_score = 0.3
        
        # 基于单词数的评分
        word_score = 0.0
        if 10 <= word_count <= 1000:
            word_score = 1.0
        elif 5 <= word_count <= 2000:
            word_score = 0.7
        elif word_count > 2:
            word_score = 0.3
        
        # 内容复杂度评分
        complexity_score = 0.0
        sentence_count = len(re.split(r'[.!?]+', content))
        if sentence_count >= 2:
            complexity_score = min(sentence_count / 10, 1.0)
        
        return (length_score + word_score + complexity_score) / 3
    
    def _assess_structure(self, data: Dict[str, Any]) -> float:
        """评估数据结构化程度"""
        total_fields = len(data)
        non_empty_fields = len([k for k, v in data.items() if v])
        
        # 结构化字段评分
        structured_score = 0.0
        if total_fields >= 5:
            structured_score = 1.0
        elif total_fields >= 3:
            structured_score = 0.7
        elif total_fields >= 1:
            structured_score = 0.3
        
        # 字段完整性评分
        completeness_score = non_empty_fields / total_fields if total_fields > 0 else 0.0
        
        return (structured_score + completeness_score) / 2
    
    def _assess_consistency(self, data: Dict[str, Any]) -> float:
        """评估数据一致性"""
        consistency_score = 1.0
        
        # 检查数据类型一致性
        for key, value in data.items():
            if key.endswith('_id') and not isinstance(value, (str, int)):
                consistency_score -= 0.1
            elif key.endswith('_count') and not isinstance(value, int):
                consistency_score -= 0.1
            elif key.endswith('_date') or key.endswith('_time'):
                if not isinstance(value, str) or not re.match(r'\d{4}-\d{2}-\d{2}', str(value)):
                    consistency_score -= 0.1
        
        return max(consistency_score, 0.0)
    
    def _find_content_field(self, data: Dict[str, Any]) -> Optional[str]:
        """查找主要内容字段"""
        content_candidates = ['content', 'text', 'body', 'description', 'message']
        for field in content_candidates:
            if field in data:
                return field
        return None

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # 初始化处理规则
        self.rules = {
            'text_cleaning': TextCleaningRule(self.config.get('text_cleaning', {})),
            'deduplication': DeduplicationRule(self.config.get('deduplication', {})),
            'format_standardization': FormatStandardizationRule(self.config.get('format_standardization', {})),
            'quality_filtering': QualityFilteringRule(self.config.get('quality_filtering', {})),
            'data_enrichment': DataEnrichmentRule(self.config.get('data_enrichment', {}))
        }
        
        # 初始化质量评估器
        self.quality_assessor = DataQualityAssessor(self.config.get('quality_assessment', {}))
    
    async def preprocess_records(
        self, 
        records: List[DataRecord],
        rules: List[str] = None
    ) -> List[DataRecord]:
        """预处理数据记录"""
        
        if rules is None:
            rules = list(self.rules.keys())
        
        self.logger.info(f"Preprocessing {len(records)} records with rules: {rules}")
        
        processed_records = []
        
        for record in records:
            try:
                processed_data = record.raw_data.copy()
                processing_metadata = record.metadata.copy()
                
                # 应用预处理规则
                for rule_name in rules:
                    if rule_name in self.rules:
                        self.logger.debug(f"Applying rule '{rule_name}' to record {record.record_id}")
                        processed_data = await self.rules[rule_name].apply(processed_data, processing_metadata)
                
                # 计算质量分数
                quality_score = self.quality_assessor.assess(processed_data)
                
                # 更新记录
                record.processed_data = processed_data
                record.metadata.update(processing_metadata)
                record.quality_score = quality_score
                record.status = record.status.PROCESSED if hasattr(record.status, 'PROCESSED') else 'processed'
                record.processed_at = utc_now()
                
                processed_records.append(record)
                
            except ValueError as e:
                # 质量过滤失败，标记为拒绝
                self.logger.warning(f"Record {record.record_id} rejected: {e}")
                record.status = record.status.REJECTED if hasattr(record.status, 'REJECTED') else 'rejected'
                record.metadata['rejection_reason'] = str(e)
                processed_records.append(record)
                
            except Exception as e:
                # 其他处理错误
                self.logger.error(f"Error preprocessing record {record.record_id}: {e}")
                record.status = record.status.ERROR if hasattr(record.status, 'ERROR') else 'error'
                record.metadata['processing_error'] = str(e)
                processed_records.append(record)
        
        success_count = len([r for r in processed_records if r.status == 'processed'])
        self.logger.info(f"Preprocessing completed. {success_count}/{len(records)} records processed successfully.")
        
        return processed_records
    
    def add_rule(self, name: str, rule: ProcessingRule):
        """添加自定义处理规则"""
        self.rules[name] = rule
    
    def remove_rule(self, name: str):
        """移除处理规则"""
        if name in self.rules:
            del self.rules[name]
    
    def get_available_rules(self) -> List[str]:
        """获取可用的处理规则"""
        return list(self.rules.keys())
    
    def set_quality_assessor(self, assessor: QualityAssessor):
        """设置质量评估器"""
        self.quality_assessor = assessor
