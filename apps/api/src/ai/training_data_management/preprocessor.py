"""
数据预处理模块

提供数据清洗、去重、格式标准化、质量评估等功能
"""

import asyncio
import json
import re
import hashlib
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging

from .models import DataRecord


@dataclass
class PreprocessingRule:
    """预处理规则定义"""
    name: str
    function: Callable
    description: str
    enabled: bool = True


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 预处理规则映射
        self.preprocessing_rules = {
            'text_cleaning': PreprocessingRule(
                name='text_cleaning',
                function=self._clean_text,
                description='清理文本数据，移除多余空白字符和特殊字符'
            ),
            'deduplication': PreprocessingRule(
                name='deduplication',
                function=self._deduplicate,
                description='数据去重，生成内容哈希'
            ),
            'format_standardization': PreprocessingRule(
                name='format_standardization',
                function=self._standardize_format,
                description='标准化数据格式，统一字段名称'
            ),
            'quality_filtering': PreprocessingRule(
                name='quality_filtering',
                function=self._filter_quality,
                description='质量过滤，检查必要字段和内容长度'
            ),
            'data_enrichment': PreprocessingRule(
                name='data_enrichment',
                function=self._enrich_data,
                description='数据丰富化，添加元数据和分析信息'
            ),
            'normalization': PreprocessingRule(
                name='normalization',
                function=self._normalize_data,
                description='数据标准化，统一数值格式'
            ),
            'validation': PreprocessingRule(
                name='validation',
                function=self._validate_data,
                description='数据验证，确保数据完整性'
            )
        }
    
    async def preprocess_data(
        self, 
        records: List[DataRecord],
        rules: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> List[DataRecord]:
        """预处理数据记录"""
        
        if rules is None:
            rules = ['text_cleaning', 'deduplication', 'format_standardization', 'quality_filtering', 'data_enrichment']
        
        if custom_config is None:
            custom_config = {}
        
        self.logger.info(f"Preprocessing {len(records)} records with rules: {rules}")
        
        processed_records = []
        error_count = 0
        
        for record in records:
            try:
                processed_data = record.raw_data.copy()
                processing_metadata = record.metadata.copy() if record.metadata else {}
                processing_metadata.update(custom_config)
                
                # 应用预处理规则
                for rule_name in rules:
                    if rule_name in self.preprocessing_rules:
                        rule = self.preprocessing_rules[rule_name]
                        if rule.enabled:
                            processed_data = await rule.function(processed_data, processing_metadata)
                    else:
                        self.logger.warning(f"Unknown preprocessing rule: {rule_name}")
                
                # 计算质量分数
                quality_score = self._calculate_quality_score(processed_data)
                
                # 更新记录
                record.processed_data = processed_data
                record.quality_score = quality_score
                record.status = 'processed'
                record.processed_at = utc_now()
                
                processed_records.append(record)
                
            except Exception as e:
                self.logger.error(f"Error preprocessing record {record.record_id}: {e}")
                record.status = 'error'
                if not record.metadata:
                    record.metadata = {}
                record.metadata['error'] = str(e)
                record.metadata['error_timestamp'] = utc_now().isoformat()
                processed_records.append(record)
                error_count += 1
        
        self.logger.info(f"Preprocessing completed. {len(processed_records)} records processed, {error_count} errors.")
        
        return processed_records
    
    async def _clean_text(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理文本数据"""
        
        def clean_text_field(text: str) -> str:
            if not isinstance(text, str):
                return str(text) if text is not None else ''
            
            # 移除多余空白字符
            text = re.sub(r'\s+', ' ', text)
            
            # 移除特殊字符(保留中文、英文、数字和常用标点)
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()[\]{}"\'@#$%^&*+=<>/-]', '', text)
            
            # 修正常见的编码问题
            text = text.replace('\ufffd', '')  # 移除替换字符
            text = text.replace('\u200b', '')  # 移除零宽度空格
            text = text.replace('\u00a0', ' ')  # 替换非断行空格
            
            return text.strip()
        
        cleaned_data = {}
        text_fields = metadata.get('text_fields', ['text', 'content', 'description', 'title', 'body'])
        
        for key, value in data.items():
            if key in text_fields and isinstance(value, str) and len(value) > 10:
                cleaned_data[key] = clean_text_field(value)
            elif isinstance(value, list):
                cleaned_data[key] = [
                    clean_text_field(item) if isinstance(item, str) and key in text_fields
                    else item
                    for item in value
                ]
            else:
                cleaned_data[key] = value
        
        return cleaned_data
    
    async def _deduplicate(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """数据去重"""
        
        # 生成内容哈希
        content_fields = metadata.get('content_fields', ['text', 'content'])
        content_for_hash = {}
        
        for field in content_fields:
            if field in data:
                content_for_hash[field] = data[field]
        
        if content_for_hash:
            content_hash = hashlib.md5(
                json.dumps(content_for_hash, sort_keys=True).encode()
            ).hexdigest()
        else:
            content_hash = hashlib.md5(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
        
        # 添加哈希到元数据
        if '_metadata' not in data:
            data['_metadata'] = {}
        data['_metadata']['content_hash'] = content_hash
        
        return data
    
    async def _standardize_format(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """标准化数据格式"""
        
        standardized_data = {}
        
        # 字段映射配置
        field_mapping = metadata.get('field_mapping', {
            'title': ['title', 'name', 'subject', 'headline', 'heading'],
            'content': ['content', 'text', 'body', 'description', 'message'],
            'author': ['author', 'creator', 'user', 'username', 'writer'],
            'timestamp': ['timestamp', 'created_at', 'date', 'time', 'publish_date'],
            'url': ['url', 'link', 'href', 'source_url', 'web_url'],
            'category': ['category', 'type', 'class', 'label', 'tag'],
            'id': ['id', 'uid', 'identifier', 'record_id']
        })
        
        # 映射字段名
        mapped_fields = set()
        for standard_field, possible_fields in field_mapping.items():
            for field in possible_fields:
                if field in data and standard_field not in standardized_data:
                    standardized_data[standard_field] = data[field]
                    mapped_fields.add(field)
                    break
        
        # 保留未映射的原始字段
        for key, value in data.items():
            if key not in mapped_fields:
                standardized_data[key] = value
        
        # 标准化数据类型
        if 'timestamp' in standardized_data:
            standardized_data['timestamp'] = self._normalize_timestamp(
                standardized_data['timestamp']
            )
        
        # 标准化URL
        if 'url' in standardized_data:
            standardized_data['url'] = self._normalize_url(
                standardized_data['url']
            )
        
        return standardized_data
    
    async def _filter_quality(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """质量过滤"""
        
        # 检查必要字段
        required_fields = metadata.get('required_fields', [])
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValueError(f"Required field '{field}' is missing or empty")
        
        # 检查内容长度
        content_fields = ['content', 'text', 'body', 'description']
        content_field = next((field for field in content_fields if field in data), None)
        
        if content_field:
            content = data[content_field]
            if isinstance(content, str):
                min_length = metadata.get('min_content_length', 10)
                max_length = metadata.get('max_content_length', 50000)
                
                if len(content) < min_length:
                    raise ValueError(f"Content too short: {len(content)} < {min_length}")
                
                if len(content) > max_length:
                    self.logger.info(f"Truncating content from {len(content)} to {max_length} characters")
                    data[content_field] = content[:max_length] + '...[truncated]'
        
        # 检查重复字段
        if metadata.get('check_duplicates', False):
            seen_values = {}
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    if value in seen_values:
                        self.logger.warning(f"Duplicate value '{value}' found in fields '{seen_values[value]}' and '{key}'")
                    else:
                        seen_values[value] = key
        
        return data
    
    async def _enrich_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """数据丰富化"""
        
        enriched_data = data.copy()
        
        # 添加处理元数据
        if '_metadata' not in enriched_data:
            enriched_data['_metadata'] = {}
        
        enriched_data['_metadata'].update({
            'processed_at': utc_now().isoformat(),
            'source_id': metadata.get('source_id'),
            'processing_version': '1.0',
            'enrichment_timestamp': utc_now().isoformat()
        })
        
        # 内容分析
        content_fields = ['content', 'text', 'body', 'description']
        content_field = next((field for field in content_fields if field in data), None)
        
        if content_field and isinstance(data[content_field], str):
            content = data[content_field]
            
            enriched_data['_analysis'] = {
                'character_count': len(content),
                'word_count': len(content.split()),
                'sentence_count': len(re.findall(r'[.!?]+', content)),
                'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
                'language': self._detect_language(content),
                'sentiment': self._analyze_sentiment(content),
                'contains_urls': bool(re.search(r'https?://\S+', content)),
                'contains_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
            }
        
        # 数据结构分析
        enriched_data['_structure'] = {
            'field_count': len([k for k in data.keys() if not k.startswith('_')]),
            'has_metadata': '_metadata' in data,
            'field_types': {k: type(v).__name__ for k, v in data.items() if not k.startswith('_')},
            'nested_levels': self._calculate_nesting_depth(data)
        }
        
        return enriched_data
    
    async def _normalize_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """数据标准化"""
        
        normalized_data = data.copy()
        
        # 标准化数值字段
        numeric_fields = metadata.get('numeric_fields', [])
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    # 尝试转换为数值
                    if isinstance(data[field], str):
                        # 清理数值字符串
                        cleaned = re.sub(r'[^\d.-]', '', str(data[field]))
                        if cleaned:
                            normalized_data[field] = float(cleaned)
                except (ValueError, TypeError):
                    self.logger.warning(f"Cannot normalize numeric field '{field}': {data[field]}")
        
        # 标准化布尔字段
        boolean_fields = metadata.get('boolean_fields', [])
        for field in boolean_fields:
            if field in data and data[field] is not None:
                normalized_data[field] = self._normalize_boolean(data[field])
        
        # 标准化列表字段
        list_fields = metadata.get('list_fields', [])
        for field in list_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], list):
                    # 尝试将字符串转换为列表
                    if isinstance(data[field], str):
                        # 按逗号分割
                        normalized_data[field] = [item.strip() for item in data[field].split(',') if item.strip()]
                    else:
                        normalized_data[field] = [data[field]]
        
        return normalized_data
    
    async def _validate_data(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """数据验证"""
        
        validation_rules = metadata.get('validation_rules', {})
        
        for field, rules in validation_rules.items():
            if field in data:
                value = data[field]
                
                # 类型验证
                if 'type' in rules:
                    expected_type = rules['type']
                    if expected_type == 'string' and not isinstance(value, str):
                        raise ValueError(f"Field '{field}' must be string, got {type(value).__name__}")
                    elif expected_type == 'number' and not isinstance(value, (int, float)):
                        raise ValueError(f"Field '{field}' must be number, got {type(value).__name__}")
                    elif expected_type == 'boolean' and not isinstance(value, bool):
                        raise ValueError(f"Field '{field}' must be boolean, got {type(value).__name__}")
                
                # 长度验证
                if 'min_length' in rules and hasattr(value, '__len__'):
                    if len(value) < rules['min_length']:
                        raise ValueError(f"Field '{field}' length {len(value)} < {rules['min_length']}")
                
                if 'max_length' in rules and hasattr(value, '__len__'):
                    if len(value) > rules['max_length']:
                        raise ValueError(f"Field '{field}' length {len(value)} > {rules['max_length']}")
                
                # 值范围验证
                if 'min_value' in rules and isinstance(value, (int, float)):
                    if value < rules['min_value']:
                        raise ValueError(f"Field '{field}' value {value} < {rules['min_value']}")
                
                if 'max_value' in rules and isinstance(value, (int, float)):
                    if value > rules['max_value']:
                        raise ValueError(f"Field '{field}' value {value} > {rules['max_value']}")
                
                # 正则表达式验证
                if 'pattern' in rules and isinstance(value, str):
                    if not re.match(rules['pattern'], value):
                        raise ValueError(f"Field '{field}' value '{value}' does not match pattern '{rules['pattern']}'")
        
        return data
    
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        
        score = 0.0
        max_score = 0.0
        
        # 完整性评分 (40%)
        completeness_weight = 0.4
        important_fields = ['title', 'content', 'text', 'body']
        completeness_score = 0.0
        
        for field in important_fields:
            max_score += completeness_weight / len(important_fields)
            if field in data and data[field] and str(data[field]).strip():
                completeness_score += completeness_weight / len(important_fields)
        
        score += completeness_score
        
        # 内容质量评分 (30%)
        content_weight = 0.3
        max_score += content_weight
        content_field = next((field for field in ['content', 'text', 'body'] if field in data), None)
        
        if content_field and isinstance(data[content_field], str):
            content = data[content_field].strip()
            content_length = len(content)
            
            if 50 <= content_length <= 5000:
                score += content_weight
            elif 20 <= content_length <= 10000:
                score += content_weight * 0.7
            elif content_length > 10:
                score += content_weight * 0.3
        
        # 结构化程度评分 (20%)
        structure_weight = 0.2
        max_score += structure_weight
        non_meta_fields = [k for k in data.keys() if not k.startswith('_')]
        field_count = len(non_meta_fields)
        
        if field_count >= 5:
            score += structure_weight
        elif field_count >= 3:
            score += structure_weight * 0.7
        elif field_count >= 1:
            score += structure_weight * 0.3
        
        # 数据一致性评分 (10%)
        consistency_weight = 0.1
        max_score += consistency_weight
        
        # 检查是否有明显的数据问题
        consistency_issues = 0
        
        # 检查空值比例
        total_fields = len(data)
        empty_fields = sum(1 for v in data.values() if not v)
        if total_fields > 0 and empty_fields / total_fields < 0.5:
            score += consistency_weight * 0.5
        
        # 检查数据类型一致性
        if '_analysis' in data and isinstance(data['_analysis'], dict):
            score += consistency_weight * 0.5
        
        return min(score / max_score, 1.0) if max_score > 0 else 0.0
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
        
        total_chars = max(chinese_chars + english_chars + japanese_chars + korean_chars, 1)
        
        if chinese_chars / total_chars > 0.3:
            return 'zh'
        elif japanese_chars / total_chars > 0.1:
            return 'ja'
        elif korean_chars / total_chars > 0.1:
            return 'ko'
        elif english_chars / total_chars > 0.5:
            return 'en'
        else:
            return 'unknown'
    
    def _analyze_sentiment(self, text: str) -> str:
        """简单的情感分析"""
        positive_words = [
            '好', '棒', '优秀', '喜欢', '成功', '满意', '开心', '高兴',
            'good', 'great', 'excellent', 'love', 'amazing', 'wonderful', 'fantastic', 'awesome'
        ]
        negative_words = [
            '坏', '差', '讨厌', '糟糕', '失败', '不满', '难过', '生气',
            'bad', 'terrible', 'hate', 'awful', 'horrible', 'disappointing', 'sad', 'angry'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count + 1:
            return 'positive'
        elif negative_count > positive_count + 1:
            return 'negative'
        else:
            return 'neutral'
    
    def _normalize_timestamp(self, timestamp: Any) -> str:
        """标准化时间戳"""
        if isinstance(timestamp, str):
            try:
                # 尝试解析ISO格式
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.isoformat()
            except ValueError:
                # 尝试其他常见格式
                common_formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%Y/%m/%d %H:%M:%S',
                    '%Y/%m/%d',
                    '%d-%m-%Y %H:%M:%S',
                    '%d-%m-%Y'
                ]
                
                for fmt in common_formats:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        return dt.replace(tzinfo=timezone.utc).isoformat()
                    except ValueError:
                        continue
                        
                return timestamp  # 无法解析，返回原值
        elif isinstance(timestamp, (int, float)):
            # Unix时间戳
            try:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                return dt.isoformat()
            except (ValueError, OSError):
                return str(timestamp)
        else:
            return str(timestamp)
    
    def _normalize_url(self, url: Any) -> str:
        """标准化URL"""
        if not isinstance(url, str):
            return str(url)
        
        url = url.strip()
        if not url:
            return url
        
        # 添加协议前缀
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
            elif '.' in url:
                url = 'https://' + url
        
        return url
    
    def _normalize_boolean(self, value: Any) -> bool:
        """标准化布尔值"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            return bool(value)
    
    def _calculate_nesting_depth(self, obj: Any, depth: int = 0) -> int:
        """计算数据嵌套深度"""
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(self._calculate_nesting_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(self._calculate_nesting_depth(item, depth + 1) for item in obj)
        else:
            return depth