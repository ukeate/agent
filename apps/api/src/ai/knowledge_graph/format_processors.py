"""
格式处理器 - 统一的数据格式转换和处理工具
"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union, TextIO
from dataclasses import dataclass
from enum import Enum
from io import StringIO
import logging

try:
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, OWL
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    Graph = URIRef = Literal = BNode = Namespace = None
    RDF = RDFS = OWL = None

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """数据格式枚举"""
    RDF_XML = "rdf/xml"
    TURTLE = "turtle"
    N_TRIPLES = "n-triples"
    JSON_LD = "json-ld"
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    EXCEL = "excel"


@dataclass
class Triple:
    """三元组数据结构"""
    subject: str
    predicate: str
    object: str
    object_type: str = "literal"  # literal, uri, blank_node
    language: Optional[str] = None
    datatype: Optional[str] = None


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    data: Optional[Any] = None
    format: Optional[DataFormat] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    statistics: Optional[Dict[str, Any]] = None


class FormatProcessor:
    """通用格式处理器基类"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def detect_format(self, data: Union[str, bytes], filename: str = None) -> Optional[DataFormat]:
        """自动检测数据格式"""
        if filename:
            # 基于文件扩展名检测
            ext = filename.lower().split('.')[-1]
            format_map = {
                'rdf': DataFormat.RDF_XML,
                'xml': DataFormat.RDF_XML,
                'ttl': DataFormat.TURTLE,
                'turtle': DataFormat.TURTLE,
                'nt': DataFormat.N_TRIPLES,
                'ntriples': DataFormat.N_TRIPLES,
                'jsonld': DataFormat.JSON_LD,
                'json': DataFormat.JSON,
                'csv': DataFormat.CSV,
                'tsv': DataFormat.TSV,
                'xlsx': DataFormat.EXCEL,
                'xls': DataFormat.EXCEL,
                'yaml': DataFormat.YAML,
                'yml': DataFormat.YAML
            }
            if ext in format_map:
                return format_map[ext]
        
        # 基于内容检测
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='ignore')
        
        data_stripped = data.strip()
        
        # XML/RDF检测
        if data_stripped.startswith('<?xml') or '<rdf:RDF' in data or '<RDF' in data:
            return DataFormat.RDF_XML
        
        # Turtle检测
        if '@prefix' in data or '@base' in data or data_stripped.startswith('@'):
            return DataFormat.TURTLE
        
        # N-Triples检测
        if ' .' in data and ('<' in data or '_:' in data):
            return DataFormat.N_TRIPLES
        
        # JSON-LD检测
        if data_stripped.startswith('{') and '@context' in data:
            return DataFormat.JSON_LD
        
        # JSON检测
        if data_stripped.startswith(('{', '[')):
            try:
                json.loads(data)
                return DataFormat.JSON
            except:
                pass
        
        # CSV/TSV检测
        lines = data.split('\n')
        if len(lines) > 1:
            first_line = lines[0]
            if '\t' in first_line and first_line.count('\t') >= 2:
                return DataFormat.TSV
            elif ',' in first_line and first_line.count(',') >= 2:
                return DataFormat.CSV
        
        return None
    
    def normalize_to_triples(self, data: Any, format: DataFormat) -> List[Triple]:
        """将各种格式的数据标准化为三元组"""
        if format == DataFormat.RDF_XML:
            return self._parse_rdf_xml(data)
        elif format == DataFormat.TURTLE:
            return self._parse_turtle(data)
        elif format == DataFormat.N_TRIPLES:
            return self._parse_ntriples(data)
        elif format == DataFormat.JSON_LD:
            return self._parse_jsonld(data)
        elif format == DataFormat.JSON:
            return self._parse_json(data)
        elif format == DataFormat.CSV:
            return self._parse_csv(data)
        elif format == DataFormat.TSV:
            return self._parse_tsv(data)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _parse_rdf_xml(self, data: str) -> List[Triple]:
        """解析RDF/XML格式"""
        if not HAS_RDFLIB:
            # 简化解析
            return self._simple_xml_parse(data)
        
        try:
            graph = Graph()
            graph.parse(data=data, format='xml')
            
            triples = []
            for s, p, o in graph:
                triple = Triple(
                    subject=str(s),
                    predicate=str(p),
                    object=str(o),
                    object_type="uri" if isinstance(o, URIRef) else "literal"
                )
                if isinstance(o, Literal):
                    if o.language:
                        triple.language = str(o.language)
                    if o.datatype:
                        triple.datatype = str(o.datatype)
                triples.append(triple)
            
            return triples
        except Exception as e:
            self.logger.error(f"RDF/XML解析失败: {e}")
            return []
    
    def _simple_xml_parse(self, data: str) -> List[Triple]:
        """简化的XML解析（当rdflib不可用时）"""
        try:
            root = ET.fromstring(data)
            triples = []
            
            # 简单的RDF/XML解析逻辑
            for elem in root.iter():
                if elem.tag.endswith('}Description') or 'Description' in elem.tag:
                    subject = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
                    
                    for child in elem:
                        predicate = child.tag
                        object_val = child.text or child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '')
                        
                        if subject and predicate and object_val:
                            triples.append(Triple(
                                subject=subject,
                                predicate=predicate,
                                object=object_val,
                                object_type="uri" if object_val.startswith("http") else "literal"
                            ))
            
            return triples
        except Exception as e:
            self.logger.error(f"简化XML解析失败: {e}")
            return []
    
    def _parse_turtle(self, data: str) -> List[Triple]:
        """解析Turtle格式"""
        if not HAS_RDFLIB:
            return self._simple_turtle_parse(data)
        
        try:
            graph = Graph()
            graph.parse(data=data, format='turtle')
            
            triples = []
            for s, p, o in graph:
                triple = Triple(
                    subject=str(s),
                    predicate=str(p),
                    object=str(o),
                    object_type="uri" if isinstance(o, URIRef) else "literal"
                )
                if isinstance(o, Literal):
                    if o.language:
                        triple.language = str(o.language)
                    if o.datatype:
                        triple.datatype = str(o.datatype)
                triples.append(triple)
            
            return triples
        except Exception as e:
            self.logger.error(f"Turtle解析失败: {e}")
            return []
    
    def _simple_turtle_parse(self, data: str) -> List[Triple]:
        """简化的Turtle解析"""
        triples = []
        lines = data.split('\n')
        
        current_subject = None
        prefixes = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 处理前缀定义
            if line.startswith('@prefix'):
                parts = line.split()
                if len(parts) >= 3:
                    prefix = parts[1].rstrip(':')
                    namespace = parts[2].strip('<>')
                    prefixes[prefix] = namespace
                continue
            
            # 处理三元组
            if ' .' in line:
                parts = line.replace(' .', '').split(None, 2)
                if len(parts) >= 3:
                    subject, predicate, obj = parts
                    
                    # 展开前缀
                    subject = self._expand_prefix(subject, prefixes)
                    predicate = self._expand_prefix(predicate, prefixes)
                    obj = self._expand_prefix(obj, prefixes)
                    
                    triples.append(Triple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        object_type="uri" if obj.startswith("<") or ":" in obj else "literal"
                    ))
        
        return triples
    
    def _expand_prefix(self, term: str, prefixes: Dict[str, str]) -> str:
        """展开前缀缩写"""
        if ':' in term and not term.startswith('<'):
            prefix, local = term.split(':', 1)
            if prefix in prefixes:
                return f"{prefixes[prefix]}{local}"
        return term.strip('<>')
    
    def _parse_ntriples(self, data: str) -> List[Triple]:
        """解析N-Triples格式"""
        triples = []
        lines = data.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.endswith(' .'):
                line = line[:-2].strip()
                parts = self._split_ntriple_line(line)
                
                if len(parts) >= 3:
                    subject = parts[0].strip('<>')
                    predicate = parts[1].strip('<>')
                    obj = parts[2]
                    
                    # 处理对象类型
                    object_type = "literal"
                    language = None
                    datatype = None
                    
                    if obj.startswith('<') and obj.endswith('>'):
                        obj = obj.strip('<>')
                        object_type = "uri"
                    elif obj.startswith('"'):
                        # 处理字面量
                        if obj.endswith('"'):
                            obj = obj[1:-1]
                        else:
                            # 可能有语言标签或数据类型
                            if '@' in obj:
                                obj, language = obj.rsplit('@', 1)
                                obj = obj[1:]  # 去掉开头的引号
                            elif '^^' in obj:
                                obj, datatype = obj.rsplit('^^', 1)
                                obj = obj[1:-1]  # 去掉引号
                                datatype = datatype.strip('<>')
                    elif obj.startswith('_:'):
                        object_type = "blank_node"
                    
                    triples.append(Triple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        object_type=object_type,
                        language=language,
                        datatype=datatype
                    ))
        
        return triples
    
    def _split_ntriple_line(self, line: str) -> List[str]:
        """分割N-Triples行"""
        parts = []
        current = ""
        in_quotes = False
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_quotes = not in_quotes
                current += char
            elif char == ' ' and not in_quotes:
                if current.strip():
                    parts.append(current.strip())
                    current = ""
            else:
                current += char
            
            i += 1
        
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def _parse_jsonld(self, data: str) -> List[Triple]:
        """解析JSON-LD格式"""
        try:
            json_data = json.loads(data)
            triples = []
            
            # 简化的JSON-LD解析
            if isinstance(json_data, dict):
                if '@graph' in json_data:
                    entities = json_data['@graph']
                else:
                    entities = [json_data]
                
                for entity in entities:
                    if isinstance(entity, dict) and '@id' in entity:
                        subject = entity['@id']
                        
                        for key, value in entity.items():
                            if key.startswith('@'):
                                continue
                            
                            predicate = key
                            
                            if isinstance(value, str):
                                triples.append(Triple(
                                    subject=subject,
                                    predicate=predicate,
                                    object=value,
                                    object_type="literal"
                                ))
                            elif isinstance(value, dict) and '@id' in value:
                                triples.append(Triple(
                                    subject=subject,
                                    predicate=predicate,
                                    object=value['@id'],
                                    object_type="uri"
                                ))
            
            return triples
        except Exception as e:
            self.logger.error(f"JSON-LD解析失败: {e}")
            return []
    
    def _parse_json(self, data: str) -> List[Triple]:
        """解析JSON格式"""
        try:
            json_data = json.loads(data)
            triples = []
            
            if isinstance(json_data, dict) and 'triples' in json_data:
                # 标准三元组JSON格式
                for triple_data in json_data['triples']:
                    if isinstance(triple_data, dict):
                        triples.append(Triple(
                            subject=triple_data.get('subject', ''),
                            predicate=triple_data.get('predicate', ''),
                            object=triple_data.get('object', ''),
                            object_type=triple_data.get('object_type', 'literal'),
                            language=triple_data.get('language'),
                            datatype=triple_data.get('datatype')
                        ))
            elif isinstance(json_data, list):
                # 三元组数组格式
                for item in json_data:
                    if isinstance(item, dict) and all(k in item for k in ['subject', 'predicate', 'object']):
                        triples.append(Triple(
                            subject=item['subject'],
                            predicate=item['predicate'],
                            object=item['object'],
                            object_type=item.get('object_type', 'literal')
                        ))
            
            return triples
        except Exception as e:
            self.logger.error(f"JSON解析失败: {e}")
            return []
    
    def _parse_csv(self, data: str) -> List[Triple]:
        """解析CSV格式"""
        try:
            reader = csv.DictReader(StringIO(data))
            triples = []
            
            for row in reader:
                if 'subject' in row and 'predicate' in row and 'object' in row:
                    triples.append(Triple(
                        subject=row['subject'],
                        predicate=row['predicate'],
                        object=row['object'],
                        object_type=row.get('object_type', 'literal'),
                        language=row.get('language'),
                        datatype=row.get('datatype')
                    ))
            
            return triples
        except Exception as e:
            self.logger.error(f"CSV解析失败: {e}")
            return []
    
    def _parse_tsv(self, data: str) -> List[Triple]:
        """解析TSV格式"""
        try:
            reader = csv.DictReader(StringIO(data), delimiter='\t')
            triples = []
            
            for row in reader:
                if 'subject' in row and 'predicate' in row and 'object' in row:
                    triples.append(Triple(
                        subject=row['subject'],
                        predicate=row['predicate'],
                        object=row['object'],
                        object_type=row.get('object_type', 'literal'),
                        language=row.get('language'),
                        datatype=row.get('datatype')
                    ))
            
            return triples
        except Exception as e:
            self.logger.error(f"TSV解析失败: {e}")
            return []


class FormatConverter:
    """格式转换器"""
    
    def __init__(self):
        self.processor = FormatProcessor()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def convert(self, data: str, source_format: DataFormat, target_format: DataFormat) -> ProcessingResult:
        """转换数据格式"""
        try:
            # 解析源数据
            triples = self.processor.normalize_to_triples(data, source_format)
            
            if not triples:
                return ProcessingResult(
                    success=False,
                    error_message="无法解析源数据或数据为空"
                )
            
            # 转换为目标格式
            if target_format == DataFormat.JSON:
                output_data = self._triples_to_json(triples)
            elif target_format == DataFormat.CSV:
                output_data = self._triples_to_csv(triples)
            elif target_format == DataFormat.TSV:
                output_data = self._triples_to_tsv(triples)
            elif target_format == DataFormat.N_TRIPLES:
                output_data = self._triples_to_ntriples(triples)
            elif target_format == DataFormat.TURTLE:
                output_data = self._triples_to_turtle(triples)
            elif target_format == DataFormat.JSON_LD:
                output_data = self._triples_to_jsonld(triples)
            else:
                return ProcessingResult(
                    success=False,
                    error_message=f"不支持的目标格式: {target_format}"
                )
            
            return ProcessingResult(
                success=True,
                data=output_data,
                format=target_format,
                statistics={
                    "total_triples": len(triples),
                    "source_format": source_format.value,
                    "target_format": target_format.value
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"格式转换失败: {str(e)}"
            )
    
    def _triples_to_json(self, triples: List[Triple]) -> str:
        """将三元组转换为JSON"""
        data = {
            "triples": [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "object_type": t.object_type,
                    "language": t.language,
                    "datatype": t.datatype
                } for t in triples
            ]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _triples_to_csv(self, triples: List[Triple]) -> str:
        """将三元组转换为CSV"""
        output = StringIO()
        writer = csv.writer(output)
        
        # 写入头部
        writer.writerow(['subject', 'predicate', 'object', 'object_type', 'language', 'datatype'])
        
        # 写入数据
        for triple in triples:
            writer.writerow([
                triple.subject,
                triple.predicate,
                triple.object,
                triple.object_type,
                triple.language or '',
                triple.datatype or ''
            ])
        
        return output.getvalue()
    
    def _triples_to_tsv(self, triples: List[Triple]) -> str:
        """将三元组转换为TSV"""
        output = StringIO()
        writer = csv.writer(output, delimiter='\t')
        
        # 写入头部
        writer.writerow(['subject', 'predicate', 'object', 'object_type', 'language', 'datatype'])
        
        # 写入数据
        for triple in triples:
            writer.writerow([
                triple.subject,
                triple.predicate,
                triple.object,
                triple.object_type,
                triple.language or '',
                triple.datatype or ''
            ])
        
        return output.getvalue()
    
    def _triples_to_ntriples(self, triples: List[Triple]) -> str:
        """将三元组转换为N-Triples"""
        lines = []
        
        for triple in triples:
            # 主语
            if triple.subject.startswith('_:'):
                s = triple.subject
            else:
                s = f'<{triple.subject}>'
            
            # 谓语
            p = f'<{triple.predicate}>'
            
            # 宾语
            if triple.object_type == "uri":
                o = f'<{triple.object}>'
            elif triple.object_type == "blank_node":
                o = triple.object
            else:
                # 字面量
                escaped = triple.object.replace('\\', '\\\\').replace('"', '\\"')
                o = f'"{escaped}"'
                
                if triple.language:
                    o += f'@{triple.language}'
                elif triple.datatype:
                    o += f'^^<{triple.datatype}>'
            
            lines.append(f'{s} {p} {o} .')
        
        return '\n'.join(lines)
    
    def _triples_to_turtle(self, triples: List[Triple]) -> str:
        """将三元组转换为Turtle"""
        lines = [
            '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
            '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
            '@prefix owl: <http://www.w3.org/2002/07/owl#> .',
            ''
        ]
        
        # 按主语分组
        subjects = {}
        for triple in triples:
            if triple.subject not in subjects:
                subjects[triple.subject] = []
            subjects[triple.subject].append(triple)
        
        # 生成Turtle格式
        for subject, subject_triples in subjects.items():
            if subject.startswith('_:'):
                s = subject
            else:
                s = f'<{subject}>'
            
            predicates = []
            for triple in subject_triples:
                p = f'<{triple.predicate}>'
                
                if triple.object_type == "uri":
                    o = f'<{triple.object}>'
                elif triple.object_type == "blank_node":
                    o = triple.object
                else:
                    escaped = triple.object.replace('\\', '\\\\').replace('"', '\\"')
                    o = f'"{escaped}"'
                    
                    if triple.language:
                        o += f'@{triple.language}'
                    elif triple.datatype:
                        o += f'^^<{triple.datatype}>'
                
                predicates.append(f'    {p} {o}')
            
            lines.append(f'{s}')
            lines.extend(predicates)
            lines.append('    .')
            lines.append('')
        
        return '\n'.join(lines)
    
    def _triples_to_jsonld(self, triples: List[Triple]) -> str:
        """将三元组转换为JSON-LD"""
        # 按主语分组
        entities = {}
        
        for triple in triples:
            subject = triple.subject
            if subject not in entities:
                entities[subject] = {"@id": subject}
            
            predicate = triple.predicate
            obj = triple.object
            
            if triple.object_type == "uri":
                obj = {"@id": obj}
            elif triple.language:
                obj = {"@value": obj, "@language": triple.language}
            elif triple.datatype:
                obj = {"@value": obj, "@type": triple.datatype}
            
            # 添加属性
            if predicate in entities[subject]:
                # 如果属性已存在，转换为数组
                existing = entities[subject][predicate]
                if not isinstance(existing, list):
                    entities[subject][predicate] = [existing]
                entities[subject][predicate].append(obj)
            else:
                entities[subject][predicate] = obj
        
        # 创建JSON-LD结构
        jsonld = {
            "@context": {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#"
            },
            "@graph": list(entities.values())
        }
        
        return json.dumps(jsonld, indent=2, ensure_ascii=False)


# 便捷函数
def detect_and_convert(data: str, target_format: DataFormat, filename: str = None) -> ProcessingResult:
    """
    自动检测数据格式并转换为目标格式
    
    Args:
        data: 输入数据
        target_format: 目标格式
        filename: 文件名（用于格式检测）
        
    Returns:
        ProcessingResult: 转换结果
    """
    processor = FormatProcessor()
    converter = FormatConverter()
    
    # 检测源格式
    source_format = processor.detect_format(data, filename)
    if not source_format:
        return ProcessingResult(
            success=False,
            error_message="无法检测数据格式"
        )
    
    # 执行转换
    return converter.convert(data, source_format, target_format)


if __name__ == "__main__":
    # 测试格式处理器
    print("测试格式处理器...")
    
    # 测试格式检测
    processor = FormatProcessor()
    
    # JSON数据测试
    json_data = '{"triples": [{"subject": "ex:John", "predicate": "rdf:type", "object": "ex:Person"}]}'
    format = processor.detect_format(json_data)
    print(f"JSON格式检测: {format}")
    
    # N-Triples数据测试
    nt_data = '<http://example.org/John> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .'
    format = processor.detect_format(nt_data)
    print(f"N-Triples格式检测: {format}")
    
    # 测试格式转换
    converter = FormatConverter()
    result = converter.convert(json_data, DataFormat.JSON, DataFormat.N_TRIPLES)
    
    if result.success:
        print("JSON到N-Triples转换成功:")
        print(result.data)
    else:
        print(f"转换失败: {result.error_message}")
    
    print("格式处理器测试完成")