"""
知识图谱数据导入器

支持多种格式的数据导入：
- RDF/XML格式
- Turtle格式  
- JSON-LD格式
- N-Triples格式
- CSV格式
- Excel格式
- 增量导入和冲突解决
- 进度跟踪和错误处理
"""

import asyncio
import csv
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO, BytesIO
import uuid
import time
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import re

from src.core.security.expression import safe_eval_bool
from src.core.logging import get_logger
logger = get_logger(__name__)

try:
    import rdflib
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.plugins.parsers.notation3 import BadSyntax
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class ImportFormat(str, Enum):
    """导入数据格式"""
    RDF_XML = "rdf_xml"
    TURTLE = "turtle"
    JSON_LD = "json_ld"
    N_TRIPLES = "n_triples"
    CSV = "csv"
    EXCEL = "excel"
    TSV = "tsv"
    AUTO = "auto"  # 自动检测

class ImportMode(str, Enum):
    """导入模式"""
    FULL = "full"
    INCREMENTAL = "incremental"
    REPLACE = "replace"
    MERGE = "merge"
    VALIDATE_ONLY = "validate_only"

class ConflictResolution(str, Enum):
    """冲突解决策略"""
    SKIP = "skip"
    OVERWRITE = "overwrite"
    MERGE = "merge"
    ERROR = "error"
    PROMPT = "prompt"

class ImportStatus(str, Enum):
    """导入状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ImportJob:
    """导入任务"""
    job_id: str
    source_format: ImportFormat
    import_mode: ImportMode
    source_data: Union[str, bytes, Dict[str, Any]]
    source_file_name: Optional[str] = None
    source_url: Optional[str] = None
    mapping_rules: Optional[Dict[str, str]] = None
    validation_config: Dict[str, Any] = field(default_factory=dict)
    conflict_resolution: ConflictResolution = ConflictResolution.SKIP
    progress_callback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_size: int = 1000
    timeout_seconds: int = 3600
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())

@dataclass
class ImportResult:
    """导入结果"""
    job_id: str
    status: ImportStatus
    total_records: int = 0
    processed_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    created_entities: List[str] = field(default_factory=list)
    created_relations: List[str] = field(default_factory=list)
    updated_entities: List[str] = field(default_factory=list)
    updated_relations: List[str] = field(default_factory=list)
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error_type: str, message: str, record_index: int = None, details: Dict[str, Any] = None):
        """添加错误"""
        self.errors.append({
            "type": error_type,
            "message": message,
            "record_index": record_index,
            "details": details or {},
            "timestamp": utc_now().isoformat()
        })
        self.failed_records += 1
    
    def add_warning(self, warning_type: str, message: str, record_index: int = None, details: Dict[str, Any] = None):
        """添加警告"""
        self.warnings.append({
            "type": warning_type,
            "message": message,
            "record_index": record_index,
            "details": details or {},
            "timestamp": utc_now().isoformat()
        })

@dataclass
class ValidationRule:
    """数据验证规则"""
    rule_id: str
    rule_type: str  # required, format, constraint, custom
    field_name: Optional[str] = None
    rule_expression: Optional[str] = None
    error_message: str = ""
    warning_only: bool = False
    enabled: bool = True

class FormatDetector:
    """格式检测器"""
    
    @staticmethod
    def detect_format(data: Union[str, bytes], filename: str = None) -> ImportFormat:
        """检测数据格式"""
        try:
            # 基于文件名检测
            if filename:
                ext = filename.lower().split('.')[-1]
                if ext == 'ttl':
                    return ImportFormat.TURTLE
                elif ext == 'xml':
                    return ImportFormat.RDF_XML
                elif ext == 'jsonld':
                    return ImportFormat.JSON_LD
                elif ext == 'nt':
                    return ImportFormat.N_TRIPLES
                elif ext == 'csv':
                    return ImportFormat.CSV
                elif ext in ['xls', 'xlsx']:
                    return ImportFormat.EXCEL
                elif ext == 'tsv':
                    return ImportFormat.TSV
            
            # 基于内容检测
            if isinstance(data, bytes):
                try:
                    content = data.decode('utf-8')
                except UnicodeDecodeError:
                    content = data.decode('latin-1')
            else:
                content = data
            
            content_stripped = content.strip()
            
            # XML格式检测
            if content_stripped.startswith('<?xml') or content_stripped.startswith('<rdf:'):
                return ImportFormat.RDF_XML
            
            # JSON-LD格式检测
            if content_stripped.startswith('{') or content_stripped.startswith('['):
                try:
                    json.loads(content)
                    return ImportFormat.JSON_LD
                except json.JSONDecodeError:
                    logger.debug("JSON解析失败，无法判定为JSON-LD", exc_info=True)
            
            # Turtle格式检测
            if '@prefix' in content or '@base' in content:
                return ImportFormat.TURTLE
            
            # N-Triples格式检测
            lines = content.split('\n')[:10]  # 检查前10行
            if all(line.strip().endswith('.') or not line.strip() for line in lines if line.strip()):
                return ImportFormat.N_TRIPLES
            
            # CSV/TSV格式检测
            try:
                # 检测分隔符
                comma_count = content[:1000].count(',')
                tab_count = content[:1000].count('\t')
                
                if tab_count > comma_count:
                    return ImportFormat.TSV
                elif comma_count > 0:
                    return ImportFormat.CSV
            except Exception:
                logger.exception("CSV/TSV格式检测失败", exc_info=True)
            
            # 默认返回Turtle
            return ImportFormat.TURTLE
            
        except Exception as e:
            logger.warning(f"格式检测失败: {e}")
            return ImportFormat.TURTLE

class BaseFormatProcessor:
    """格式处理器基类"""
    
    def __init__(self):
        self.format_type = None
    
    async def parse(
        self, 
        data: Union[str, bytes, Dict], 
        mapping_rules: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """解析数据，返回标准化的三元组列表"""
        raise NotImplementedError
    
    async def validate(
        self, 
        data: Union[str, bytes, Dict], 
        validation_rules: List[ValidationRule] = None
    ) -> List[Dict[str, Any]]:
        """验证数据格式和内容"""
        errors = []
        
        try:
            await self.parse(data)
        except Exception as e:
            errors.append({
                "type": "format_error",
                "message": f"数据格式解析失败: {str(e)}",
                "severity": "error"
            })
        
        return errors

class RDFXMLProcessor(BaseFormatProcessor):
    """RDF/XML格式处理器"""
    
    def __init__(self):
        super().__init__()
        self.format_type = ImportFormat.RDF_XML
    
    async def parse(
        self, 
        data: Union[str, bytes], 
        mapping_rules: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """解析RDF/XML数据"""
        try:
            if not RDFLIB_AVAILABLE:
                return await self._parse_xml_manually(data)
            
            # 使用RDFLib解析
            g = rdflib.Graph()
            
            if isinstance(data, str):
                g.parse(data=data, format="xml")
            else:
                g.parse(data=BytesIO(data), format="xml")
            
            # 转换为标准格式
            triples = []
            for subject, predicate, obj in g:
                triple = {
                    "subject": str(subject),
                    "predicate": str(predicate),
                    "object": str(obj),
                    "object_type": self._get_object_type(obj)
                }
                triples.append(triple)
            
            return triples
            
        except Exception as e:
            logger.error(f"RDF/XML解析失败: {e}")
            raise ValueError(f"RDF/XML格式错误: {str(e)}")
    
    async def _parse_xml_manually(self, data: Union[str, bytes]) -> List[Dict[str, Any]]:
        """手动解析XML（当RDFLib不可用时）"""
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            root = ET.fromstring(data)
            triples = []
            
            # 简化的RDF/XML解析
            namespaces = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
            }
            
            # 查找所有Description元素
            for desc in root.findall('.//rdf:Description', namespaces):
                subject = desc.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', 
                                  desc.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID'))
                
                if not subject:
                    subject = f"_:bnode{uuid.uuid4().hex[:8]}"
                
                # 处理子元素作为属性
                for child in desc:
                    predicate = child.tag
                    obj = child.text or ""
                    
                    resource = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                    if resource:
                        obj = resource
                        obj_type = "uri"
                    else:
                        obj_type = "literal"
                    
                    triple = {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "object_type": obj_type
                    }
                    triples.append(triple)
            
            return triples
            
        except ET.ParseError as e:
            raise ValueError(f"XML格式错误: {str(e)}")
    
    def _get_object_type(self, obj) -> str:
        """获取对象类型"""
        if hasattr(obj, 'toPython'):
            if isinstance(obj, rdflib.URIRef):
                return "uri"
            elif isinstance(obj, rdflib.BNode):
                return "bnode"
            else:
                return "literal"
        else:
            return "literal"

class TurtleProcessor(BaseFormatProcessor):
    """Turtle格式处理器"""
    
    def __init__(self):
        super().__init__()
        self.format_type = ImportFormat.TURTLE
    
    async def parse(
        self, 
        data: Union[str, bytes], 
        mapping_rules: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """解析Turtle数据"""
        try:
            if not RDFLIB_AVAILABLE:
                return await self._parse_turtle_manually(data)
            
            # 使用RDFLib解析
            g = rdflib.Graph()
            
            if isinstance(data, str):
                g.parse(data=data, format="turtle")
            else:
                g.parse(data=BytesIO(data), format="turtle")
            
            # 转换为标准格式
            triples = []
            for subject, predicate, obj in g:
                triple = {
                    "subject": str(subject),
                    "predicate": str(predicate),
                    "object": str(obj),
                    "object_type": self._get_object_type(obj)
                }
                triples.append(triple)
            
            return triples
            
        except Exception as e:
            logger.error(f"Turtle解析失败: {e}")
            raise ValueError(f"Turtle格式错误: {str(e)}")
    
    async def _parse_turtle_manually(self, data: Union[str, bytes]) -> List[Dict[str, Any]]:
        """手动解析Turtle（当RDFLib不可用时）"""
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            lines = data.split('\n')
            triples = []
            namespaces = {}
            current_subject = None
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                
                # 处理前缀声明
                if line.startswith('@prefix'):
                    parts = line.split()
                    if len(parts) >= 3:
                        prefix = parts[1].rstrip(':')
                        uri = parts[2].strip('<>').rstrip('.')
                        namespaces[prefix] = uri
                    continue
                
                # 简化的三元组解析
                if line.endswith('.'):
                    # 完整的三元组语句
                    line = line.rstrip('.')
                    parts = self._split_turtle_line(line)
                    
                    if len(parts) >= 3:
                        subject = self._expand_uri(parts[0], namespaces)
                        predicate = self._expand_uri(parts[1], namespaces)
                        obj = self._expand_uri(parts[2], namespaces)
                        
                        triple = {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj,
                            "object_type": "uri" if obj.startswith("http") else "literal"
                        }
                        triples.append(triple)
                        current_subject = subject
                
                elif line.endswith(';'):
                    # 续行，同一主语
                    line = line.rstrip(';')
                    parts = self._split_turtle_line(line)
                    
                    if len(parts) >= 2 and current_subject:
                        predicate = self._expand_uri(parts[0], namespaces)
                        obj = self._expand_uri(parts[1], namespaces)
                        
                        triple = {
                            "subject": current_subject,
                            "predicate": predicate,
                            "object": obj,
                            "object_type": "uri" if obj.startswith("http") else "literal"
                        }
                        triples.append(triple)
            
            return triples
            
        except Exception as e:
            raise ValueError(f"Turtle格式解析错误: {str(e)}")
    
    def _split_turtle_line(self, line: str) -> List[str]:
        """分割Turtle行"""
        # 简化的分割，实际实现需要处理引号内的空格
        parts = []
        current_part = ""
        in_quotes = False
        
        for char in line:
            if char == '"' and (not current_part or current_part[-1] != '\\'):
                in_quotes = not in_quotes
                current_part += char
            elif char.isspace() and not in_quotes:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts
    
    def _expand_uri(self, term: str, namespaces: Dict[str, str]) -> str:
        """展开URI"""
        if term.startswith('<') and term.endswith('>'):
            return term[1:-1]
        elif ':' in term and not term.startswith('http'):
            prefix, local = term.split(':', 1)
            if prefix in namespaces:
                return namespaces[prefix] + local
        elif term.startswith('"') and term.endswith('"'):
            return term[1:-1]  # 字面量
        
        return term
    
    def _get_object_type(self, obj) -> str:
        """获取对象类型"""
        if hasattr(obj, 'toPython'):
            if isinstance(obj, rdflib.URIRef):
                return "uri"
            elif isinstance(obj, rdflib.BNode):
                return "bnode"
            else:
                return "literal"
        else:
            return "literal"

class JSONLDProcessor(BaseFormatProcessor):
    """JSON-LD格式处理器"""
    
    def __init__(self):
        super().__init__()
        self.format_type = ImportFormat.JSON_LD
    
    async def parse(
        self, 
        data: Union[str, bytes, Dict], 
        mapping_rules: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """解析JSON-LD数据"""
        try:
            if isinstance(data, (str, bytes)):
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                json_data = json.loads(data)
            else:
                json_data = data
            
            if not RDFLIB_AVAILABLE:
                return await self._parse_jsonld_manually(json_data)
            
            # 使用RDFLib解析JSON-LD
            g = rdflib.Graph()
            g.parse(data=json.dumps(json_data), format="json-ld")
            
            # 转换为标准格式
            triples = []
            for subject, predicate, obj in g:
                triple = {
                    "subject": str(subject),
                    "predicate": str(predicate),
                    "object": str(obj),
                    "object_type": self._get_object_type(obj)
                }
                triples.append(triple)
            
            return triples
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {str(e)}")
        except Exception as e:
            logger.error(f"JSON-LD解析失败: {e}")
            raise ValueError(f"JSON-LD格式错误: {str(e)}")
    
    async def _parse_jsonld_manually(self, json_data: Dict) -> List[Dict[str, Any]]:
        """手动解析JSON-LD"""
        try:
            triples = []
            context = json_data.get('@context', {})
            
            # 处理单个对象或对象数组
            if '@graph' in json_data:
                objects = json_data['@graph']
            elif isinstance(json_data, list):
                objects = json_data
            else:
                objects = [json_data]
            
            for obj in objects:
                if isinstance(obj, dict):
                    subject_id = obj.get('@id', f"_:bnode{uuid.uuid4().hex[:8]}")
                    
                    for key, value in obj.items():
                        if key.startswith('@'):
                            continue
                        
                        predicate = self._expand_jsonld_term(key, context)
                        
                        # 处理值
                        if isinstance(value, list):
                            for v in value:
                                object_value, object_type = self._process_jsonld_value(v, context)
                                triples.append({
                                    "subject": subject_id,
                                    "predicate": predicate,
                                    "object": object_value,
                                    "object_type": object_type
                                })
                        else:
                            object_value, object_type = self._process_jsonld_value(value, context)
                            triples.append({
                                "subject": subject_id,
                                "predicate": predicate,
                                "object": object_value,
                                "object_type": object_type
                            })
            
            return triples
            
        except Exception as e:
            raise ValueError(f"JSON-LD解析错误: {str(e)}")
    
    def _expand_jsonld_term(self, term: str, context: Dict) -> str:
        """展开JSON-LD术语"""
        if term in context:
            return context[term]
        elif ':' in term:
            prefix, local = term.split(':', 1)
            if prefix in context:
                return context[prefix] + local
        
        return term
    
    def _process_jsonld_value(self, value: Any, context: Dict) -> tuple:
        """处理JSON-LD值"""
        if isinstance(value, dict):
            if '@id' in value:
                return value['@id'], "uri"
            elif '@value' in value:
                return value['@value'], "literal"
        elif isinstance(value, str):
            if value.startswith('http'):
                return value, "uri"
            else:
                return value, "literal"
        else:
            return str(value), "literal"
    
    def _get_object_type(self, obj) -> str:
        """获取对象类型"""
        if hasattr(obj, 'toPython'):
            if isinstance(obj, rdflib.URIRef):
                return "uri"
            elif isinstance(obj, rdflib.BNode):
                return "bnode"
            else:
                return "literal"
        else:
            return "literal"

class CSVProcessor(BaseFormatProcessor):
    """CSV格式处理器"""
    
    def __init__(self):
        super().__init__()
        self.format_type = ImportFormat.CSV
    
    async def parse(
        self, 
        data: Union[str, bytes], 
        mapping_rules: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """解析CSV数据"""
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            # 检测分隔符
            dialect = csv.Sniffer().sniff(data[:1000])
            
            # 解析CSV
            csv_reader = csv.DictReader(StringIO(data), dialect=dialect)
            rows = list(csv_reader)
            
            # 应用映射规则
            if not mapping_rules:
                # 默认映射规则：假设有subject, predicate, object列
                mapping_rules = self._generate_default_mapping(csv_reader.fieldnames)
            
            triples = []
            for row_index, row in enumerate(rows):
                try:
                    triple = self._row_to_triple(row, mapping_rules, row_index)
                    if triple:
                        triples.append(triple)
                except Exception as e:
                    logger.warning(f"CSV行 {row_index + 1} 解析失败: {e}")
            
            return triples
            
        except Exception as e:
            logger.error(f"CSV解析失败: {e}")
            raise ValueError(f"CSV格式错误: {str(e)}")
    
    def _generate_default_mapping(self, fieldnames: List[str]) -> Dict[str, str]:
        """生成默认映射规则"""
        mapping = {}
        
        # 尝试自动识别主要列
        fieldnames_lower = [name.lower() for name in fieldnames]
        
        # 主语列
        for subject_col in ['subject', 'entity', 'id', 'uri', 's']:
            if subject_col in fieldnames_lower:
                mapping['subject'] = fieldnames[fieldnames_lower.index(subject_col)]
                break
        
        # 谓词列
        for predicate_col in ['predicate', 'property', 'relation', 'p']:
            if predicate_col in fieldnames_lower:
                mapping['predicate'] = fieldnames[fieldnames_lower.index(predicate_col)]
                break
        
        # 宾语列
        for object_col in ['object', 'value', 'o']:
            if object_col in fieldnames_lower:
                mapping['object'] = fieldnames[fieldnames_lower.index(object_col)]
                break
        
        # 如果没有找到标准列，使用前三列
        if len(mapping) == 0 and len(fieldnames) >= 3:
            mapping = {
                'subject': fieldnames[0],
                'predicate': fieldnames[1],
                'object': fieldnames[2]
            }
        
        return mapping
    
    def _row_to_triple(self, row: Dict[str, str], mapping: Dict[str, str], row_index: int) -> Optional[Dict[str, Any]]:
        """将CSV行转换为三元组"""
        subject = row.get(mapping.get('subject', ''), '').strip()
        predicate = row.get(mapping.get('predicate', ''), '').strip()
        obj = row.get(mapping.get('object', ''), '').strip()
        
        # 跳过空行
        if not subject or not predicate or not obj:
            return None
        
        # 确保主语和谓词是URI格式
        if not subject.startswith('http'):
            subject = f"http://example.org/entity/{subject}"
        
        if not predicate.startswith('http'):
            predicate = f"http://example.org/property/{predicate}"
        
        # 判断宾语类型
        object_type = "uri" if obj.startswith('http') else "literal"
        
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "object_type": object_type,
            "source_row": row_index + 1
        }

class ExcelProcessor(BaseFormatProcessor):
    """Excel格式处理器"""
    
    def __init__(self):
        super().__init__()
        self.format_type = ImportFormat.EXCEL
    
    async def parse(
        self, 
        data: Union[str, bytes], 
        mapping_rules: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """解析Excel数据"""
        try:
            if not PANDAS_AVAILABLE:
                raise ValueError("需要安装pandas库才能处理Excel文件")
            
            # 读取Excel文件
            if isinstance(data, str):
                # 假设是文件路径
                df = pd.read_excel(data)
            else:
                # 字节数据
                df = pd.read_excel(BytesIO(data))
            
            # 转换为CSV格式处理
            csv_data = df.to_csv(index=False)
            
            # 使用CSV处理器
            csv_processor = CSVProcessor()
            return await csv_processor.parse(csv_data, mapping_rules)
            
        except Exception as e:
            logger.error(f"Excel解析失败: {e}")
            raise ValueError(f"Excel格式错误: {str(e)}")

class NTriplesProcessor(BaseFormatProcessor):
    """N-Triples格式处理器"""
    
    def __init__(self):
        super().__init__()
        self.format_type = ImportFormat.N_TRIPLES
    
    async def parse(
        self, 
        data: Union[str, bytes], 
        mapping_rules: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """解析N-Triples数据"""
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            triples = []
            lines = data.split('\n')
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                
                # N-Triples格式：<subject> <predicate> <object> .
                if not line.endswith('.'):
                    logger.warning(f"N-Triples行 {line_num + 1} 格式错误：缺少结束点")
                    continue
                
                line = line.rstrip('.')
                parts = self._parse_ntriples_line(line)
                
                if len(parts) != 3:
                    logger.warning(f"N-Triples行 {line_num + 1} 格式错误：应该包含3个部分")
                    continue
                
                subject, predicate, obj = parts
                
                triple = {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "object_type": self._get_ntriples_object_type(obj),
                    "source_line": line_num + 1
                }
                triples.append(triple)
            
            return triples
            
        except Exception as e:
            logger.error(f"N-Triples解析失败: {e}")
            raise ValueError(f"N-Triples格式错误: {str(e)}")
    
    def _parse_ntriples_line(self, line: str) -> List[str]:
        """解析N-Triples行"""
        parts = []
        current_part = ""
        in_uri = False
        in_literal = False
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char == '<' and not in_literal:
                in_uri = True
                current_part += char
            elif char == '>' and in_uri and not in_literal:
                in_uri = False
                current_part += char
                if not in_literal:
                    parts.append(current_part.strip())
                    current_part = ""
            elif char == '"' and not in_uri:
                if not in_literal:
                    in_literal = True
                else:
                    # 检查是否是转义的引号
                    if i > 0 and line[i-1] != '\\':
                        in_literal = False
                current_part += char
            elif char.isspace() and not in_uri and not in_literal:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""
            else:
                current_part += char
            
            i += 1
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts
    
    def _get_ntriples_object_type(self, obj: str) -> str:
        """获取N-Triples对象类型"""
        if obj.startswith('<') and obj.endswith('>'):
            return "uri"
        elif obj.startswith('_:'):
            return "bnode"
        else:
            return "literal"

class DataImporter:
    """数据导入器主类"""
    
    def __init__(self, graph_store=None, version_manager=None):
        self.graph_store = graph_store
        self.version_manager = version_manager
        
        # 格式处理器
        self.format_processors = {
            ImportFormat.RDF_XML: RDFXMLProcessor(),
            ImportFormat.TURTLE: TurtleProcessor(),
            ImportFormat.JSON_LD: JSONLDProcessor(),
            ImportFormat.N_TRIPLES: NTriplesProcessor(),
            ImportFormat.CSV: CSVProcessor(),
            ImportFormat.EXCEL: ExcelProcessor(),
            ImportFormat.TSV: CSVProcessor(),  # TSV使用CSV处理器
        }
        
        # 活跃的导入任务
        self.active_jobs = {}
        
        # 格式检测器
        self.format_detector = FormatDetector()
    
    async def import_data(self, import_job: ImportJob) -> ImportResult:
        """执行数据导入"""
        start_time = time.time()
        job_id = import_job.job_id
        
        # 初始化结果
        result = ImportResult(
            job_id=job_id,
            status=ImportStatus.RUNNING
        )
        
        try:
            # 注册活跃任务
            self.active_jobs[job_id] = {
                'job': import_job,
                'result': result,
                'start_time': start_time
            }
            
            # 1. 格式检测
            if import_job.source_format == ImportFormat.AUTO:
                detected_format = self.format_detector.detect_format(
                    import_job.source_data,
                    import_job.source_file_name
                )
                import_job.source_format = detected_format
                logger.info(f"检测到格式: {detected_format}")
            
            # 2. 选择处理器
            processor = self.format_processors.get(import_job.source_format)
            if not processor:
                raise ValueError(f"不支持的格式: {import_job.source_format}")
            
            # 3. 验证模式
            if import_job.import_mode == ImportMode.VALIDATE_ONLY:
                return await self._validate_only(import_job, processor, result)
            
            # 4. 创建导入版本（如果有版本管理器）
            import_version = None
            if self.version_manager:
                try:
                    import_version = await self.version_manager.create_import_version(
                        f"Import job {job_id}",
                        import_job.metadata
                    )
                    result.metadata['version_id'] = import_version.version_id
                except Exception as e:
                    logger.warning(f"创建导入版本失败: {e}")
            
            # 5. 数据解析
            logger.info(f"开始解析数据，格式: {import_job.source_format}")
            parsed_data = await processor.parse(
                import_job.source_data,
                import_job.mapping_rules
            )
            
            result.total_records = len(parsed_data)
            logger.info(f"解析完成，共 {result.total_records} 条记录")
            
            # 6. 数据验证
            if import_job.validation_config:
                validation_errors = await self._validate_parsed_data(
                    parsed_data,
                    import_job.validation_config
                )
                result.validation_errors = validation_errors
                
                if validation_errors and not import_job.validation_config.get('allow_errors', False):
                    result.status = ImportStatus.FAILED
                    result.add_error(
                        "validation_failed",
                        f"数据验证失败，发现 {len(validation_errors)} 个错误"
                    )
                    return result
            
            # 7. 冲突检测和解决
            if import_job.import_mode != ImportMode.FULL:
                resolved_data = await self._resolve_conflicts(
                    parsed_data,
                    import_job
                )
                parsed_data = resolved_data
            
            # 8. 执行导入
            await self._execute_import(parsed_data, import_job, result)
            
            # 9. 完成导入版本
            if import_version and self.version_manager:
                try:
                    await self.version_manager.finalize_import_version(
                        import_version.version_id,
                        result
                    )
                except Exception as e:
                    logger.warning(f"完成导入版本失败: {e}")
            
            # 10. 设置最终状态
            if result.failed_records == 0:
                result.status = ImportStatus.COMPLETED
            elif result.successful_records > 0:
                result.status = ImportStatus.COMPLETED
                result.add_warning(
                    "partial_success",
                    f"部分导入成功：{result.successful_records}/{result.total_records}"
                )
            else:
                result.status = ImportStatus.FAILED
            
            result.execution_time = time.time() - start_time
            logger.info(f"导入完成，用时 {result.execution_time:.2f} 秒")
            
            return result
            
        except Exception as e:
            logger.error(f"数据导入失败: {e}")
            result.status = ImportStatus.FAILED
            result.add_error("import_failed", str(e))
            result.execution_time = time.time() - start_time
            
            # 回滚版本
            if import_version and self.version_manager:
                try:
                    await self.version_manager.rollback_version(
                        import_version.version_id
                    )
                except Exception as rollback_error:
                    logger.error(f"回滚版本失败: {rollback_error}")
            
            return result
        
        finally:
            # 清理活跃任务
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _validate_only(
        self, 
        import_job: ImportJob, 
        processor: BaseFormatProcessor, 
        result: ImportResult
    ) -> ImportResult:
        """仅验证模式"""
        try:
            # 格式验证
            format_errors = await processor.validate(
                import_job.source_data,
                import_job.validation_config.get('rules', [])
            )
            
            result.validation_errors.extend(format_errors)
            
            # 尝试解析以进行更深度的验证
            try:
                parsed_data = await processor.parse(
                    import_job.source_data,
                    import_job.mapping_rules
                )
                result.total_records = len(parsed_data)
                
                # 内容验证
                if import_job.validation_config:
                    content_errors = await self._validate_parsed_data(
                        parsed_data,
                        import_job.validation_config
                    )
                    result.validation_errors.extend(content_errors)
                
            except Exception as parse_error:
                result.validation_errors.append({
                    "type": "parse_error",
                    "message": str(parse_error),
                    "severity": "error"
                })
            
            # 设置状态
            error_count = sum(1 for err in result.validation_errors if err.get('severity') == 'error')
            
            if error_count == 0:
                result.status = ImportStatus.COMPLETED
                result.successful_records = result.total_records
            else:
                result.status = ImportStatus.FAILED
                result.failed_records = error_count
            
            return result
            
        except Exception as e:
            result.status = ImportStatus.FAILED
            result.add_error("validation_error", str(e))
            return result
    
    async def _validate_parsed_data(
        self, 
        parsed_data: List[Dict[str, Any]], 
        validation_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """验证解析后的数据"""
        errors = []
        
        validation_rules = validation_config.get('rules', [])
        
        for rule in validation_rules:
            if not isinstance(rule, ValidationRule):
                continue
            
            if not rule.enabled:
                continue
            
            try:
                # 应用验证规则
                rule_errors = await self._apply_validation_rule(rule, parsed_data)
                errors.extend(rule_errors)
                
            except Exception as e:
                errors.append({
                    "type": "rule_error",
                    "message": f"验证规则 {rule.rule_id} 执行失败: {str(e)}",
                    "severity": "error"
                })
        
        return errors
    
    async def _apply_validation_rule(
        self, 
        rule: ValidationRule, 
        parsed_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """应用验证规则"""
        errors = []
        
        for index, record in enumerate(parsed_data):
            try:
                valid = True
                error_message = rule.error_message or f"验证规则 {rule.rule_id} 失败"
                
                if rule.rule_type == "required":
                    # 必填字段检查
                    if rule.field_name and not record.get(rule.field_name):
                        valid = False
                
                elif rule.rule_type == "format":
                    # 格式检查
                    if rule.field_name and rule.rule_expression:
                        field_value = record.get(rule.field_name, "")
                        if field_value and not re.match(rule.rule_expression, str(field_value)):
                            valid = False
                
                elif rule.rule_type == "constraint":
                    # 约束检查
                    if rule.rule_expression:
                        # 简单的约束表达式评估
                        try:
                            # 这里可以实现更复杂的约束检查逻辑
                            if not safe_eval_bool(rule.rule_expression, record):
                                valid = False
                        except Exception:
                            valid = False
                
                if not valid:
                    severity = "warning" if rule.warning_only else "error"
                    errors.append({
                        "type": "validation_error",
                        "rule_id": rule.rule_id,
                        "message": error_message,
                        "severity": severity,
                        "record_index": index,
                        "record_data": record
                    })
                    
            except Exception as e:
                errors.append({
                    "type": "rule_application_error",
                    "rule_id": rule.rule_id,
                    "message": f"应用验证规则失败: {str(e)}",
                    "severity": "error",
                    "record_index": index
                })
        
        return errors
    
    async def _resolve_conflicts(
        self, 
        parsed_data: List[Dict[str, Any]], 
        import_job: ImportJob
    ) -> List[Dict[str, Any]]:
        """解决数据冲突"""
        if not self.graph_store:
            logger.warning("无图存储，跳过冲突检测")
            return parsed_data
        
        resolved_data = []
        
        for record in parsed_data:
            try:
                # 检查是否存在冲突
                conflict = await self._check_conflict(record)
                
                if conflict:
                    # 应用冲突解决策略
                    resolved_record = await self._apply_conflict_resolution(
                        record,
                        conflict,
                        import_job.conflict_resolution
                    )
                    
                    if resolved_record:
                        resolved_data.append(resolved_record)
                else:
                    resolved_data.append(record)
                    
            except Exception as e:
                logger.error(f"冲突解决失败: {e}")
                # 根据策略决定是否包含记录
                if import_job.conflict_resolution != ConflictResolution.ERROR:
                    resolved_data.append(record)
        
        return resolved_data
    
    async def _check_conflict(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检查数据冲突"""
        # 简化实现：检查是否存在相同的主语-谓词组合
        subject = record.get('subject')
        predicate = record.get('predicate')
        
        if not subject or not predicate:
            return None
        
        # 这里应该查询图存储检查是否存在
        # 暂时返回None表示无冲突
        return None
    
    async def _apply_conflict_resolution(
        self, 
        record: Dict[str, Any], 
        conflict: Dict[str, Any], 
        resolution: ConflictResolution
    ) -> Optional[Dict[str, Any]]:
        """应用冲突解决策略"""
        if resolution == ConflictResolution.SKIP:
            return None
        elif resolution == ConflictResolution.OVERWRITE:
            return record
        elif resolution == ConflictResolution.MERGE:
            # 合并逻辑
            merged_record = dict(record)
            # 这里可以实现具体的合并逻辑
            return merged_record
        elif resolution == ConflictResolution.ERROR:
            raise ValueError(f"数据冲突: {record}")
        else:
            return record
    
    async def _execute_import(
        self, 
        parsed_data: List[Dict[str, Any]], 
        import_job: ImportJob, 
        result: ImportResult
    ):
        """执行数据导入"""
        chunk_size = import_job.chunk_size
        
        for i in range(0, len(parsed_data), chunk_size):
            chunk = parsed_data[i:i + chunk_size]
            
            try:
                await self._import_chunk(chunk, import_job, result)
            except Exception as e:
                logger.error(f"导入数据块失败 (索引 {i}): {e}")
                result.add_error("chunk_import_failed", str(e), i)
    
    async def _import_chunk(
        self, 
        chunk: List[Dict[str, Any]], 
        import_job: ImportJob, 
        result: ImportResult
    ):
        """导入数据块"""
        for record in chunk:
            try:
                # 这里应该调用图存储的API来插入数据
                # 暂时模拟成功
                
                # 根据记录类型更新统计
                if 'subject' in record:
                    if record.get('source_row') or record.get('source_line'):
                        # 新记录
                        result.created_entities.append(record['subject'])
                    else:
                        # 更新记录
                        result.updated_entities.append(record['subject'])
                
                result.successful_records += 1
                result.processed_records += 1
                
            except Exception as e:
                result.add_error(
                    "record_import_failed",
                    str(e),
                    record.get('source_row') or record.get('source_line'),
                    record
                )
                result.processed_records += 1
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if job_id not in self.active_jobs:
            return None
        
        job_info = self.active_jobs[job_id]
        result = job_info['result']
        
        return {
            'job_id': job_id,
            'status': result.status,
            'progress': {
                'total_records': result.total_records,
                'processed_records': result.processed_records,
                'successful_records': result.successful_records,
                'failed_records': result.failed_records,
                'percentage': (
                    (result.processed_records / max(result.total_records, 1)) * 100
                ) if result.total_records > 0 else 0
            },
            'execution_time': time.time() - job_info['start_time'],
            'errors': len(result.errors),
            'warnings': len(result.warnings)
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """取消导入任务"""
        if job_id not in self.active_jobs:
            return False
        
        job_info = self.active_jobs[job_id]
        job_info['result'].status = ImportStatus.CANCELLED
        
        # 这里可以实现更复杂的取消逻辑
        # 例如停止正在进行的操作
        
        return True
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式列表"""
        return [fmt.value for fmt in ImportFormat if fmt != ImportFormat.AUTO]

# 创建默认导入器实例
default_data_importer = DataImporter()

async def import_knowledge_data(
    data: Union[str, bytes, Dict],
    source_format: ImportFormat = ImportFormat.AUTO,
    import_mode: ImportMode = ImportMode.INCREMENTAL,
    mapping_rules: Dict[str, str] = None,
    validation_config: Dict[str, Any] = None,
    conflict_resolution: ConflictResolution = ConflictResolution.SKIP,
    source_file_name: str = None
) -> ImportResult:
    """导入知识数据的便捷函数"""
    import_job = ImportJob(
        job_id=str(uuid.uuid4()),
        source_format=source_format,
        import_mode=import_mode,
        source_data=data,
        source_file_name=source_file_name,
        mapping_rules=mapping_rules,
        validation_config=validation_config or {},
        conflict_resolution=conflict_resolution
    )
    
    return await default_data_importer.import_data(import_job)
