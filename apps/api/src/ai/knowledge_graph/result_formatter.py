"""
SPARQL查询结果格式转换器

支持多种结果格式的转换：
- JSON格式（标准和紧凑）
- XML格式（SPARQL Results格式）
- CSV格式 
- Turtle/RDF格式
- 自定义格式
"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from io import StringIO

from src.core.logging import get_logger
logger = get_logger(__name__)

class ResultFormat(str, Enum):
    """结果格式类型"""
    JSON = "json"
    JSON_COMPACT = "json_compact"
    XML = "xml"
    CSV = "csv"
    TSV = "tsv"
    TURTLE = "turtle"
    RDF_XML = "rdf_xml"
    HTML = "html"
    PLAIN_TEXT = "plain_text"

class SPARQLResultFormatter:
    """SPARQL结果格式转换器"""
    
    def __init__(self):
        self.formatters = {
            ResultFormat.JSON: self._format_json,
            ResultFormat.JSON_COMPACT: self._format_json_compact,
            ResultFormat.XML: self._format_xml,
            ResultFormat.CSV: self._format_csv,
            ResultFormat.TSV: self._format_tsv,
            ResultFormat.TURTLE: self._format_turtle,
            ResultFormat.RDF_XML: self._format_rdf_xml,
            ResultFormat.HTML: self._format_html,
            ResultFormat.PLAIN_TEXT: self._format_plain_text
        }
    
    def format_results(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        format_type: ResultFormat,
        query_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """格式化查询结果"""
        try:
            if format_type not in self.formatters:
                raise ValueError(f"不支持的格式类型: {format_type}")
            
            formatter = self.formatters[format_type]
            formatted_data = formatter(results, result_type, query_metadata or {})
            
            return {
                "format": format_type,
                "content_type": self._get_content_type(format_type),
                "data": formatted_data,
                "metadata": {
                    "result_count": len(results),
                    "result_type": result_type,
                    "format_type": format_type
                }
            }
            
        except Exception as e:
            logger.error(f"结果格式化失败: {e}")
            return {
                "format": format_type,
                "content_type": "text/plain",
                "data": f"格式化错误: {str(e)}",
                "error": str(e)
            }
    
    def _format_json(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为标准JSON"""
        if result_type == "boolean":
            # ASK查询结果
            result_data = {
                "head": {},
                "boolean": results[0].get("result", False) if results else False
            }
        elif result_type == "bindings":
            # SELECT查询结果
            variables = []
            if results:
                variables = list(results[0].keys())
            
            bindings = []
            for result in results:
                binding = {}
                for var, value in result.items():
                    binding[var] = self._format_binding_value(value)
                bindings.append(binding)
            
            result_data = {
                "head": {"vars": variables},
                "results": {"bindings": bindings}
            }
        elif result_type == "graph":
            # CONSTRUCT/DESCRIBE查询结果
            result_data = {
                "head": {},
                "results": {
                    "triples": results
                }
            }
        else:
            result_data = {
                "head": {},
                "results": results
            }
        
        # 添加元数据
        if metadata:
            result_data["metadata"] = metadata
        
        return json.dumps(result_data, indent=2, ensure_ascii=False)
    
    def _format_json_compact(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为紧凑JSON"""
        json_data = self._format_json(results, result_type, metadata)
        # 重新解析并去除缩进
        data = json.loads(json_data)
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    
    def _format_xml(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为SPARQL Results XML格式"""
        # 创建根元素
        root = ET.Element("sparql")
        root.set("xmlns", "http://www.w3.org/2005/sparql-results#")
        
        # 头部
        head = ET.SubElement(root, "head")
        
        if result_type == "boolean":
            # ASK查询
            boolean_elem = ET.SubElement(root, "boolean")
            boolean_elem.text = str(results[0].get("result", False) if results else False).lower()
        
        elif result_type == "bindings":
            # SELECT查询
            if results:
                variables = list(results[0].keys())
                for var in variables:
                    var_elem = ET.SubElement(head, "variable")
                    var_elem.set("name", var)
            
            results_elem = ET.SubElement(root, "results")
            
            for result in results:
                result_elem = ET.SubElement(results_elem, "result")
                
                for var, value in result.items():
                    binding_elem = ET.SubElement(result_elem, "binding")
                    binding_elem.set("name", var)
                    
                    value_elem = self._create_xml_value_element(value)
                    binding_elem.append(value_elem)
        
        # 转换为字符串
        return ET.tostring(root, encoding="unicode", xml_declaration=True)
    
    def _format_csv(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为CSV"""
        if not results:
            return ""
        
        if result_type == "boolean":
            return f"result\n{results[0].get('result', False)}"
        
        if result_type == "bindings":
            output = StringIO()
            writer = csv.DictWriter(
                output, 
                fieldnames=list(results[0].keys()),
                quoting=csv.QUOTE_MINIMAL
            )
            
            writer.writeheader()
            for result in results:
                # 确保所有值都是字符串
                string_result = {}
                for key, value in result.items():
                    string_result[key] = self._value_to_string(value)
                writer.writerow(string_result)
            
            return output.getvalue()
        
        # 对于图结果，转换为三元组格式
        if result_type == "graph":
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(["subject", "predicate", "object"])
            
            for triple in results:
                writer.writerow([
                    self._value_to_string(triple.get("subject", "")),
                    self._value_to_string(triple.get("predicate", "")),
                    self._value_to_string(triple.get("object", ""))
                ])
            
            return output.getvalue()
        
        return ""
    
    def _format_tsv(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为TSV（制表符分隔）"""
        csv_data = self._format_csv(results, result_type, metadata)
        # 将逗号替换为制表符
        lines = csv_data.split('\n')
        tsv_lines = []
        
        for line in lines:
            if line.strip():
                # 简单替换，实际实现应该考虑引号内的逗号
                tsv_line = line.replace(',', '\t')
                tsv_lines.append(tsv_line)
        
        return '\n'.join(tsv_lines)
    
    def _format_turtle(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为Turtle格式"""
        if result_type != "graph":
            return "# Turtle format only available for graph results\n"
        
        turtle_lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            ""
        ]
        
        for triple in results:
            subject = self._format_turtle_term(triple.get("subject", ""))
            predicate = self._format_turtle_term(triple.get("predicate", ""))
            obj = self._format_turtle_term(triple.get("object", ""))
            
            turtle_lines.append(f"{subject} {predicate} {obj} .")
        
        return '\n'.join(turtle_lines)
    
    def _format_rdf_xml(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为RDF/XML格式"""
        if result_type != "graph":
            return "<!-- RDF/XML format only available for graph results -->\n"
        
        # 创建RDF根元素
        root = ET.Element("rdf:RDF")
        root.set("xmlns:rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        root.set("xmlns:rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        
        # 按主语分组三元组
        subjects = {}
        for triple in results:
            subject = triple.get("subject", "")
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(triple)
        
        # 为每个主语创建Description元素
        for subject, triples in subjects.items():
            desc = ET.SubElement(root, "rdf:Description")
            if subject.startswith("http"):
                desc.set("rdf:about", subject)
            else:
                desc.set("rdf:nodeID", subject)
            
            for triple in triples:
                predicate = triple.get("predicate", "")
                obj = triple.get("object", "")
                
                # 简化谓词名
                pred_name = predicate.split("/")[-1] if "/" in predicate else predicate
                prop_elem = ET.SubElement(desc, pred_name)
                
                if obj.startswith("http"):
                    prop_elem.set("rdf:resource", obj)
                else:
                    prop_elem.text = obj
        
        return ET.tostring(root, encoding="unicode", xml_declaration=True)
    
    def _format_html(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为HTML表格"""
        if not results:
            return "<p>无结果</p>"
        
        if result_type == "boolean":
            result_value = results[0].get("result", False) if results else False
            return f"<p>查询结果: <strong>{result_value}</strong></p>"
        
        html_lines = [
            "<table border='1' cellpadding='5' cellspacing='0'>",
            "<thead><tr>"
        ]
        
        # 表头
        if result_type == "bindings" and results:
            for var in results[0].keys():
                html_lines.append(f"<th>{var}</th>")
        elif result_type == "graph":
            html_lines.extend(["<th>Subject</th>", "<th>Predicate</th>", "<th>Object</th>"])
        
        html_lines.extend(["</tr></thead>", "<tbody>"])
        
        # 数据行
        for result in results:
            html_lines.append("<tr>")
            
            if result_type == "bindings":
                for value in result.values():
                    formatted_value = self._html_escape(self._value_to_string(value))
                    html_lines.append(f"<td>{formatted_value}</td>")
            elif result_type == "graph":
                for key in ["subject", "predicate", "object"]:
                    value = self._html_escape(self._value_to_string(result.get(key, "")))
                    html_lines.append(f"<td>{value}</td>")
            
            html_lines.append("</tr>")
        
        html_lines.extend(["</tbody>", "</table>"])
        
        # 添加元数据
        if metadata:
            html_lines.append(f"<p><small>结果数: {len(results)}</small></p>")
        
        return '\n'.join(html_lines)
    
    def _format_plain_text(
        self, 
        results: List[Dict[str, Any]], 
        result_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """格式化为纯文本"""
        if not results:
            return "无结果\n"
        
        if result_type == "boolean":
            result_value = results[0].get("result", False) if results else False
            return f"查询结果: {result_value}\n"
        
        lines = []
        
        if result_type == "bindings":
            if results:
                # 计算列宽
                headers = list(results[0].keys())
                col_widths = {}
                
                for header in headers:
                    col_widths[header] = len(header)
                
                for result in results:
                    for key, value in result.items():
                        col_widths[key] = max(col_widths[key], len(self._value_to_string(value)))
                
                # 表头
                header_line = " | ".join(
                    header.ljust(col_widths[header]) for header in headers
                )
                lines.append(header_line)
                lines.append("-" * len(header_line))
                
                # 数据行
                for result in results:
                    data_line = " | ".join(
                        self._value_to_string(result[header]).ljust(col_widths[header])
                        for header in headers
                    )
                    lines.append(data_line)
        
        elif result_type == "graph":
            for triple in results:
                subject = self._value_to_string(triple.get("subject", ""))
                predicate = self._value_to_string(triple.get("predicate", ""))
                obj = self._value_to_string(triple.get("object", ""))
                lines.append(f"{subject} {predicate} {obj}")
        
        if metadata:
            lines.append(f"\n结果数: {len(results)}")
        
        return '\n'.join(lines) + '\n'
    
    def _format_binding_value(self, value: Any) -> Dict[str, str]:
        """格式化绑定值为SPARQL结果格式"""
        if value is None:
            return {"type": "literal", "value": ""}
        
        value_str = str(value)
        
        # 检测URI
        if value_str.startswith("http://") or value_str.startswith("https://"):
            return {"type": "uri", "value": value_str}
        
        # 检测空白节点
        if value_str.startswith("_:"):
            return {"type": "bnode", "value": value_str}
        
        # 默认为字面量
        return {"type": "literal", "value": value_str}
    
    def _create_xml_value_element(self, value: Any) -> ET.Element:
        """创建XML值元素"""
        value_str = str(value) if value is not None else ""
        
        if value_str.startswith("http://") or value_str.startswith("https://"):
            elem = ET.Element("uri")
        elif value_str.startswith("_:"):
            elem = ET.Element("bnode")
        else:
            elem = ET.Element("literal")
        
        elem.text = value_str
        return elem
    
    def _format_turtle_term(self, term: str) -> str:
        """格式化Turtle术语"""
        if not term:
            return '""'
        
        if term.startswith("http://") or term.startswith("https://"):
            return f"<{term}>"
        elif term.startswith("_:"):
            return term
        else:
            # 字面量需要引号
            escaped = term.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
    
    def _value_to_string(self, value: Any) -> str:
        """将值转换为字符串"""
        if value is None:
            return ""
        return str(value)
    
    def _html_escape(self, text: str) -> str:
        """HTML转义"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    def _get_content_type(self, format_type: ResultFormat) -> str:
        """获取内容类型"""
        content_types = {
            ResultFormat.JSON: "application/json",
            ResultFormat.JSON_COMPACT: "application/json",
            ResultFormat.XML: "application/sparql-results+xml",
            ResultFormat.CSV: "text/csv",
            ResultFormat.TSV: "text/tab-separated-values",
            ResultFormat.TURTLE: "text/turtle",
            ResultFormat.RDF_XML: "application/rdf+xml",
            ResultFormat.HTML: "text/html",
            ResultFormat.PLAIN_TEXT: "text/plain"
        }
        
        return content_types.get(format_type, "text/plain")
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式列表"""
        return [format_type.value for format_type in ResultFormat]

# 创建默认格式化器实例
default_formatter = SPARQLResultFormatter()

def format_sparql_results(
    results: List[Dict[str, Any]], 
    result_type: str,
    format_type: ResultFormat = ResultFormat.JSON,
    query_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """格式化SPARQL结果的便捷函数"""
    return default_formatter.format_results(
        results, 
        result_type, 
        format_type, 
        query_metadata
    )
