"""
结构化数据提取器
"""

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

class StructuredDataExtractor:
    """结构化数据提取器"""
    
    @staticmethod
    def extract_from_image(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """从图像分析结果中提取结构化数据"""
        structured = {
            "type": "image",
            "entities": [],
            "text": [],
            "metadata": {}
        }
        
        # 提取对象/实体
        if "objects" in raw_data:
            for obj in raw_data.get("objects", []):
                structured["entities"].append({
                    "type": "object",
                    "name": obj,
                    "confidence": raw_data.get("confidence", 0.8)
                })
        
        # 提取文本
        if "text_content" in raw_data:
            text_content = raw_data.get("text_content", "")
            if text_content:
                # 尝试提取结构化文本元素
                structured["text"] = StructuredDataExtractor._extract_text_elements(text_content)
        
        # 提取情感
        if "sentiment" in raw_data:
            structured["metadata"]["sentiment"] = raw_data["sentiment"]
        
        # 添加描述
        if "description" in raw_data:
            structured["metadata"]["description"] = raw_data["description"]
        
        return structured
    
    @staticmethod
    def extract_from_document(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """从文档分析结果中提取结构化数据"""
        structured = {
            "type": "document",
            "summary": raw_data.get("summary", ""),
            "sections": [],
            "key_points": [],
            "entities": [],
            "metadata": {}
        }
        
        # 提取关键点
        if "key_points" in raw_data:
            structured["key_points"] = raw_data["key_points"]
        
        # 提取文档结构
        if "structure" in raw_data:
            structure = raw_data["structure"]
            if isinstance(structure, str):
                # 尝试解析结构描述
                sections = StructuredDataExtractor._parse_document_structure(structure)
                structured["sections"] = sections
            elif isinstance(structure, list):
                structured["sections"] = structure
        
        # 提取数据元素
        if "data_elements" in raw_data:
            data_elements = raw_data["data_elements"]
            if isinstance(data_elements, dict):
                # 提取表格数据
                if "tables" in data_elements:
                    structured["entities"].extend([
                        {"type": "table", "data": table}
                        for table in data_elements.get("tables", [])
                    ])
                
                # 提取数字/统计
                if "statistics" in data_elements:
                    structured["entities"].extend([
                        {"type": "statistic", "value": stat}
                        for stat in data_elements.get("statistics", [])
                    ])
        
        # 添加文档类型
        if "document_type" in raw_data:
            structured["metadata"]["document_type"] = raw_data["document_type"]
        
        return structured
    
    @staticmethod
    def extract_from_video(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """从视频分析结果中提取结构化数据"""
        structured = {
            "type": "video",
            "frames": [],
            "timeline": [],
            "entities": [],
            "metadata": {}
        }
        
        # 提取关键帧信息
        if "key_frames" in raw_data:
            for frame in raw_data["key_frames"]:
                frame_data = {
                    "index": frame.get("frame_index", 0),
                    "timestamp": frame.get("timestamp", 0),
                    "description": frame.get("analysis", ""),
                    "entities": []
                }
                
                # 从帧描述中提取实体
                if frame_data["description"]:
                    entities = StructuredDataExtractor._extract_entities_from_text(
                        frame_data["description"]
                    )
                    frame_data["entities"] = entities
                    structured["entities"].extend(entities)
                
                structured["frames"].append(frame_data)
                
                # 添加到时间线
                structured["timeline"].append({
                    "time": frame_data["timestamp"],
                    "event": frame_data["description"][:100]  # 简短描述
                })
        
        # 添加整体摘要
        if "overall_summary" in raw_data:
            structured["metadata"]["summary"] = raw_data["overall_summary"]
        
        # 添加帧数信息
        if "frame_count" in raw_data:
            structured["metadata"]["total_frames"] = raw_data["frame_count"]
        
        return structured
    
    @staticmethod
    def extract_from_audio(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """从音频分析结果中提取结构化数据"""
        structured = {
            "type": "audio",
            "transcription": raw_data.get("transcription", ""),
            "segments": [],
            "entities": [],
            "metadata": {}
        }
        
        # 提取转录文本
        if "transcription" in raw_data:
            text = raw_data["transcription"]
            # 提取句子
            sentences = StructuredDataExtractor._split_into_sentences(text)
            structured["segments"] = [
                {"type": "sentence", "text": sent} 
                for sent in sentences
            ]
            
            # 提取实体
            structured["entities"] = StructuredDataExtractor._extract_entities_from_text(text)
        
        # 添加语言信息
        if "language" in raw_data:
            structured["metadata"]["language"] = raw_data["language"]
        
        # 添加说话人信息
        if "speakers" in raw_data:
            structured["metadata"]["speakers"] = raw_data["speakers"]
        
        return structured
    
    @staticmethod
    def _extract_text_elements(text: str) -> List[Dict[str, Any]]:
        """从文本中提取结构化元素"""
        elements = []
        
        # 提取电子邮件
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        for email in emails:
            elements.append({"type": "email", "value": email})
        
        # 提取电话号码
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        for phone in phones:
            elements.append({"type": "phone", "value": phone})
        
        # 提取URL
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        for url in urls:
            elements.append({"type": "url", "value": url})
        
        # 提取日期
        dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', text)
        for date in dates:
            elements.append({"type": "date", "value": date})
        
        # 提取数字/金额
        numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', text)
        for number in numbers:
            if '$' in number:
                elements.append({"type": "money", "value": number})
            elif len(number) > 3:
                elements.append({"type": "number", "value": number})
        
        return elements
    
    @staticmethod
    def _extract_entities_from_text(text: str) -> List[Dict[str, str]]:
        """从文本中提取命名实体"""
        entities = []
        
        # 简单的实体提取（实际应用中可以使用NER模型）
        # 提取大写词（可能是人名、地名、组织名）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for noun in proper_nouns:
            if len(noun) > 2:  # 过滤太短的词
                entities.append({
                    "type": "proper_noun",
                    "value": noun
                })
        
        # 提取引号中的内容
        quotes = re.findall(r'"([^"]*)"', text)
        for quote in quotes:
            if quote:
                entities.append({
                    "type": "quote",
                    "value": quote
                })
        
        return entities
    
    @staticmethod
    def _parse_document_structure(structure_text: str) -> List[Dict[str, Any]]:
        """解析文档结构描述"""
        sections = []
        
        # 尝试识别章节标题
        lines = structure_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 识别编号的章节
            if re.match(r'^\d+\.', line) or re.match(r'^[A-Z]\.', line):
                sections.append({
                    "type": "numbered_section",
                    "title": line
                })
            # 识别标题（以大写开头）
            elif line[0].isupper() and len(line) < 100:
                sections.append({
                    "type": "heading",
                    "title": line
                })
        
        return sections
    
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """将文本分割成句子"""
        # 简单的句子分割
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def merge_extracted_data(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个提取结果"""
        merged = {
            "entities": [],
            "key_points": [],
            "metadata": {},
            "summary": ""
        }
        
        for data in data_list:
            # 合并实体
            if "entities" in data:
                merged["entities"].extend(data["entities"])
            
            # 合并关键点
            if "key_points" in data:
                merged["key_points"].extend(data["key_points"])
            
            # 合并元数据
            if "metadata" in data:
                merged["metadata"].update(data["metadata"])
            
            # 合并摘要
            if "summary" in data and data["summary"]:
                if merged["summary"]:
                    merged["summary"] += " " + data["summary"]
                else:
                    merged["summary"] = data["summary"]
        
        # 去重实体
        seen_entities = set()
        unique_entities = []
        for entity in merged["entities"]:
            entity_key = f"{entity.get('type', '')}:{entity.get('value', '')}"
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_entities.append(entity)
        merged["entities"] = unique_entities
        
        return merged
