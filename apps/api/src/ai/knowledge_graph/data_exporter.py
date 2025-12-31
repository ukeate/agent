"""
数据导出器 - 支持多种格式的知识图谱数据导出
"""

import asyncio
import json
import csv
import io
import zipfile
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, TextIO
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
from src.core.logging import get_logger, setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

try:
    from rdflib import Graph, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    Graph = URIRef = Literal = BNode = None
    RDF = RDFS = OWL = None
    SPARQLStore = None

class ExportFormat(Enum):
    """支持的导出格式"""
    RDF_XML = "rdf/xml"
    TURTLE = "turtle" 
    JSON_LD = "json-ld"
    N_TRIPLES = "n-triples"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    GRAPHML = "graphml"

class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    ZIP = "zip"
    GZIP = "gzip"

@dataclass
class ExportOptions:
    """导出选项配置"""
    format: ExportFormat
    compression: CompressionType = CompressionType.NONE
    include_metadata: bool = True
    include_statistics: bool = False
    chunk_size: int = 10000
    max_file_size_mb: int = 100
    filter_predicates: Optional[List[str]] = None
    filter_subjects: Optional[List[str]] = None
    custom_namespace_prefixes: Optional[Dict[str, str]] = None
    export_directory: Optional[str] = None

@dataclass
class ExportJob:
    """导出任务定义"""
    job_id: str
    graph_uri: Optional[str] = None
    sparql_query: Optional[str] = None
    options: Optional[ExportOptions] = None
    output_filename: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class ExportResult:
    """导出结果"""
    job_id: str
    success: bool
    output_files: List[str] = None
    total_triples: int = 0
    total_entities: int = 0
    total_relations: int = 0
    export_time_ms: int = 0
    file_size_bytes: int = 0
    error_message: Optional[str] = None
    warnings: List[str] = None
    statistics: Optional[Dict[str, Any]] = None

class DataExporter:
    """知识图谱数据导出器"""
    
    def __init__(self, store_endpoint: str = None):
        self.store_endpoint = store_endpoint
        self.active_jobs: Dict[str, ExportJob] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    async def export_data(self, export_job: ExportJob) -> ExportResult:
        """
        执行数据导出
        
        Args:
            export_job: 导出任务配置
            
        Returns:
            ExportResult: 导出结果
        """
        start_time = utc_now()
        self.active_jobs[export_job.job_id] = export_job
        
        try:
            self.logger.info(f"开始导出任务: {export_job.job_id}")
            
            # 获取数据
            if export_job.sparql_query:
                data = await self._export_by_query(export_job)
            else:
                data = await self._export_by_graph(export_job)
            
            # 格式化输出
            result = await self._format_and_write(data, export_job)
            
            # 计算执行时间
            execution_time = (utc_now() - start_time).total_seconds() * 1000
            result.export_time_ms = int(execution_time)
            
            # 添加统计信息
            if export_job.options and export_job.options.include_statistics:
                result.statistics = await self._generate_statistics(data)
            
            self.logger.info(f"导出任务完成: {export_job.job_id}, 耗时: {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"导出任务失败 {export_job.job_id}: {str(e)}")
            return ExportResult(
                job_id=export_job.job_id,
                success=False,
                error_message=str(e),
                export_time_ms=int((utc_now() - start_time).total_seconds() * 1000)
            )
        finally:
            if export_job.job_id in self.active_jobs:
                del self.active_jobs[export_job.job_id]
    
    async def _export_by_query(self, export_job: ExportJob) -> Dict[str, Any]:
        """通过SPARQL查询导出数据"""
        if not HAS_RDFLIB:
            # 简化实现 - 实际项目中需要连接到三元组存储
            return {
                "triples": [
                    {"subject": "ex:Entity1", "predicate": "rdf:type", "object": "ex:Person"},
                    {"subject": "ex:Entity1", "predicate": "ex:name", "object": "John Doe"}
                ],
                "metadata": {"query": export_job.sparql_query}
            }
        
        # RDFLib实现
        store = SPARQLStore(self.store_endpoint) if self.store_endpoint else None
        graph = Graph(store=store) if store else Graph()
        
        # 执行查询
        results = graph.query(export_job.sparql_query)
        
        triples = []
        for row in results:
            triple_data = {}
            for i, var in enumerate(results.vars):
                triple_data[str(var)] = str(row[i])
            triples.append(triple_data)
        
        return {
            "triples": triples,
            "metadata": {"query": export_job.sparql_query, "total_results": len(triples)}
        }
    
    async def _export_by_graph(self, export_job: ExportJob) -> Dict[str, Any]:
        """通过图URI导出数据"""
        if not HAS_RDFLIB:
            # 简化实现
            return {
                "triples": [
                    {"subject": "ex:Entity1", "predicate": "rdf:type", "object": "ex:Person"},
                    {"subject": "ex:Entity2", "predicate": "rdf:type", "object": "ex:Organization"}
                ],
                "metadata": {"graph_uri": export_job.graph_uri}
            }
        
        # RDFLib实现
        store = SPARQLStore(self.store_endpoint) if self.store_endpoint else None  
        graph = Graph(store=store) if store else Graph()
        
        if export_job.graph_uri:
            # 从特定命名图导出
            graph_ref = URIRef(export_job.graph_uri)
            triples = list(graph.triples((None, None, None), context=graph_ref))
        else:
            # 导出所有三元组
            triples = list(graph.triples((None, None, None)))
        
        triple_data = []
        for s, p, o in triples:
            triple_data.append({
                "subject": str(s),
                "predicate": str(p), 
                "object": str(o)
            })
        
        return {
            "triples": triple_data,
            "metadata": {"graph_uri": export_job.graph_uri, "total_triples": len(triple_data)}
        }
    
    async def _format_and_write(self, data: Dict[str, Any], export_job: ExportJob) -> ExportResult:
        """格式化数据并写入文件"""
        options = export_job.options or ExportOptions(format=ExportFormat.JSON)
        output_files = []
        
        try:
            if options.format == ExportFormat.RDF_XML:
                file_path = await self._write_rdf_xml(data, export_job)
            elif options.format == ExportFormat.TURTLE:
                file_path = await self._write_turtle(data, export_job)
            elif options.format == ExportFormat.JSON_LD:
                file_path = await self._write_jsonld(data, export_job)
            elif options.format == ExportFormat.N_TRIPLES:
                file_path = await self._write_ntriples(data, export_job)
            elif options.format == ExportFormat.CSV:
                file_path = await self._write_csv(data, export_job)
            elif options.format == ExportFormat.JSON:
                file_path = await self._write_json(data, export_job)
            elif options.format == ExportFormat.EXCEL:
                file_path = await self._write_excel(data, export_job)
            elif options.format == ExportFormat.GRAPHML:
                file_path = await self._write_graphml(data, export_job)
            else:
                raise ValueError(f"不支持的导出格式: {options.format}")
            
            output_files.append(file_path)
            
            # 应用压缩
            if options.compression != CompressionType.NONE:
                compressed_file = await self._apply_compression(file_path, options.compression)
                output_files = [compressed_file]
            
            # 获取文件大小
            file_size = sum(Path(f).stat().st_size for f in output_files if Path(f).exists())
            
            return ExportResult(
                job_id=export_job.job_id,
                success=True,
                output_files=output_files,
                total_triples=len(data.get("triples", [])),
                file_size_bytes=file_size
            )
            
        except Exception as e:
            self.logger.error(f"格式化写入失败: {str(e)}")
            raise
    
    async def _write_json(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入JSON格式"""
        output_dir = export_job.options.export_directory or "/tmp"
        filename = export_job.output_filename or f"{export_job.job_id}.json"
        file_path = Path(output_dir) / filename
        
        # 格式化数据
        output_data = {
            "metadata": data.get("metadata", {}),
            "triples": data.get("triples", []),
            "exported_at": utc_now().isoformat()
        }
        
        if export_job.options and export_job.options.include_metadata:
            output_data["export_info"] = asdict(export_job.options)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    async def _write_csv(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入CSV格式"""
        output_dir = export_job.options.export_directory or "/tmp"
        filename = export_job.output_filename or f"{export_job.job_id}.csv"
        file_path = Path(output_dir) / filename
        
        triples = data.get("triples", [])
        if not triples:
            raise ValueError("没有数据可导出")
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            if isinstance(triples[0], dict):
                # 字典格式
                fieldnames = list(triples[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(triples)
            else:
                # 三元组格式  
                writer = csv.writer(f)
                writer.writerow(["Subject", "Predicate", "Object"])
                for triple in triples:
                    if isinstance(triple, dict):
                        writer.writerow([
                            triple.get("subject", ""),
                            triple.get("predicate", ""), 
                            triple.get("object", "")
                        ])
        
        return str(file_path)
    
    async def _write_rdf_xml(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入RDF/XML格式"""
        if not HAS_RDFLIB:
            raise ImportError("需要安装rdflib库来支持RDF/XML导出")
        
        output_dir = export_job.options.export_directory or "/tmp"
        filename = export_job.output_filename or f"{export_job.job_id}.rdf"
        file_path = Path(output_dir) / filename
        
        # 创建RDF图
        graph = Graph()
        
        # 添加命名空间前缀
        if export_job.options and export_job.options.custom_namespace_prefixes:
            for prefix, namespace in export_job.options.custom_namespace_prefixes.items():
                graph.bind(prefix, namespace)
        
        # 添加三元组
        for triple in data.get("triples", []):
            if isinstance(triple, dict):
                s = URIRef(triple["subject"]) if triple["subject"].startswith("http") else BNode(triple["subject"])
                p = URIRef(triple["predicate"]) 
                o = URIRef(triple["object"]) if triple["object"].startswith("http") else Literal(triple["object"])
                graph.add((s, p, o))
        
        # 序列化为RDF/XML
        rdf_data = graph.serialize(format='xml')
        
        with open(file_path, 'wb') as f:
            f.write(rdf_data)
        
        return str(file_path)
    
    async def _write_turtle(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入Turtle格式"""
        if not HAS_RDFLIB:
            raise ImportError("需要安装rdflib库来支持Turtle导出")
        
        output_dir = export_job.options.export_directory or "/tmp"
        filename = export_job.output_filename or f"{export_job.job_id}.ttl"
        file_path = Path(output_dir) / filename
        
        # 创建RDF图
        graph = Graph()
        
        # 添加三元组
        for triple in data.get("triples", []):
            if isinstance(triple, dict):
                s = URIRef(triple["subject"]) if triple["subject"].startswith("http") else BNode(triple["subject"])
                p = URIRef(triple["predicate"])
                o = URIRef(triple["object"]) if triple["object"].startswith("http") else Literal(triple["object"])
                graph.add((s, p, o))
        
        # 序列化为Turtle
        turtle_data = graph.serialize(format='turtle')
        
        with open(file_path, 'wb') as f:
            f.write(turtle_data)
        
        return str(file_path)
    
    async def _write_jsonld(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入JSON-LD格式"""
        output_dir = export_job.options.export_directory or "/tmp" 
        filename = export_job.output_filename or f"{export_job.job_id}.jsonld"
        file_path = Path(output_dir) / filename
        
        # 简化的JSON-LD结构
        jsonld_data = {
            "@context": {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "@graph": []
        }
        
        # 将三元组转换为JSON-LD
        entities = {}
        for triple in data.get("triples", []):
            if isinstance(triple, dict):
                subj = triple["subject"]
                pred = triple["predicate"] 
                obj = triple["object"]
                
                if subj not in entities:
                    entities[subj] = {"@id": subj}
                
                entities[subj][pred] = obj
        
        jsonld_data["@graph"] = list(entities.values())
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(jsonld_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    async def _write_ntriples(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入N-Triples格式"""
        output_dir = export_job.options.export_directory or "/tmp"
        filename = export_job.output_filename or f"{export_job.job_id}.nt"
        file_path = Path(output_dir) / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for triple in data.get("triples", []):
                if isinstance(triple, dict):
                    s = f'<{triple["subject"]}>' if triple["subject"].startswith("http") else f'_:{triple["subject"]}'
                    p = f'<{triple["predicate"]}>'
                    o = f'<{triple["object"]}>' if triple["object"].startswith("http") else f'"{triple["object"]}"'
                    f.write(f'{s} {p} {o} .\n')
        
        return str(file_path)
    
    async def _write_excel(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入Excel格式"""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("需要安装openpyxl库来支持Excel导出")
        
        output_dir = export_job.options.export_directory or "/tmp"
        filename = export_job.output_filename or f"{export_job.job_id}.xlsx"
        file_path = Path(output_dir) / filename
        
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Knowledge Graph Data"
        
        # 写入头部
        triples = data.get("triples", [])
        if triples and isinstance(triples[0], dict):
            headers = list(triples[0].keys())
            sheet.append(headers)
            
            # 写入数据
            for triple in triples:
                row_data = [triple.get(h, "") for h in headers]
                sheet.append(row_data)
        
        workbook.save(file_path)
        return str(file_path)
    
    async def _write_graphml(self, data: Dict[str, Any], export_job: ExportJob) -> str:
        """写入GraphML格式"""
        output_dir = export_job.options.export_directory or "/tmp"
        filename = export_job.output_filename or f"{export_job.job_id}.graphml"
        file_path = Path(output_dir) / filename
        
        # 简化的GraphML结构
        graphml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <graph id="knowledge_graph" edgedefault="directed">
'''
        
        graphml_footer = '''  </graph>
</graphml>'''
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(graphml_header)
            
            # 收集节点和边
            nodes = set()
            edges = []
            
            for triple in data.get("triples", []):
                if isinstance(triple, dict):
                    subj = triple["subject"]
                    pred = triple["predicate"]
                    obj = triple["object"]
                    
                    nodes.add(subj)
                    nodes.add(obj)
                    edges.append((subj, obj, pred))
            
            # 写入节点
            for node in nodes:
                f.write(f'    <node id="{node}"/>\n')
            
            # 写入边
            for i, (source, target, label) in enumerate(edges):
                f.write(f'    <edge id="e{i}" source="{source}" target="{target}"/>\n')
            
            f.write(graphml_footer)
        
        return str(file_path)
    
    async def _apply_compression(self, file_path: str, compression: CompressionType) -> str:
        """应用压缩"""
        if compression == CompressionType.ZIP:
            zip_path = f"{file_path}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, Path(file_path).name)
            
            # 删除原文件
            Path(file_path).unlink()
            return zip_path
        
        elif compression == CompressionType.GZIP:
            import gzip
            gz_path = f"{file_path}.gz"
            with open(file_path, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # 删除原文件
            Path(file_path).unlink()
            return gz_path
        
        return file_path
    
    async def _generate_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成导出统计信息"""
        triples = data.get("triples", [])
        
        statistics = {
            "total_triples": len(triples),
            "unique_subjects": 0,
            "unique_predicates": 0,
            "unique_objects": 0,
            "generated_at": utc_now().isoformat()
        }
        
        if triples:
            subjects = set()
            predicates = set()
            objects = set()
            
            for triple in triples:
                if isinstance(triple, dict):
                    subjects.add(triple.get("subject", ""))
                    predicates.add(triple.get("predicate", ""))
                    objects.add(triple.get("object", ""))
            
            statistics.update({
                "unique_subjects": len(subjects),
                "unique_predicates": len(predicates),
                "unique_objects": len(objects)
            })
        
        return statistics
    
    async def get_export_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取导出任务状态"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": "running",
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "options": asdict(job.options) if job.options else None
            }
        return None
    
    async def cancel_export(self, job_id: str) -> bool:
        """取消导出任务"""
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
            self.logger.info(f"已取消导出任务: {job_id}")
            return True
        return False
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的导出格式"""
        return [format.value for format in ExportFormat]
    
    def get_supported_compressions(self) -> List[str]:
        """获取支持的压缩格式"""
        return [compression.value for compression in CompressionType]

# 便捷函数
async def export_knowledge_graph(
    job_id: str,
    format: ExportFormat = ExportFormat.JSON,
    graph_uri: Optional[str] = None,
    sparql_query: Optional[str] = None,
    output_filename: Optional[str] = None,
    compression: CompressionType = CompressionType.NONE,
    export_directory: Optional[str] = None
) -> ExportResult:
    """
    导出知识图谱数据的便捷函数
    
    Args:
        job_id: 导出任务ID
        format: 导出格式
        graph_uri: 图URI (可选)
        sparql_query: SPARQL查询 (可选) 
        output_filename: 输出文件名
        compression: 压缩类型
        export_directory: 导出目录
        
    Returns:
        ExportResult: 导出结果
    """
    exporter = DataExporter()
    
    options = ExportOptions(
        format=format,
        compression=compression,
        export_directory=export_directory
    )
    
    job = ExportJob(
        job_id=job_id,
        graph_uri=graph_uri,
        sparql_query=sparql_query,
        options=options,
        output_filename=output_filename,
        created_at=utc_now()
    )
    
    return await exporter.export_data(job)

if __name__ == "__main__":
    # 测试导出功能
    async def test_export():
        setup_logging()
        logger.info("测试数据导出")
        
        # JSON导出测试
        result = await export_knowledge_graph(
            job_id="test_json_export",
            format=ExportFormat.JSON,
            output_filename="test_export.json"
        )
        
        logger.info("JSON导出结果", success=result.success)
        if result.success:
            logger.info("输出文件", files=result.output_files)
            logger.info("三元组数量", total=result.total_triples)
        
        # CSV导出测试
        result = await export_knowledge_graph(
            job_id="test_csv_export", 
            format=ExportFormat.CSV,
            output_filename="test_export.csv",
            compression=CompressionType.ZIP
        )
        
        logger.info("CSV压缩导出结果", success=result.success)
        if result.success:
            logger.info("压缩文件", files=result.output_files)
        logger.info("数据导出测试完成")
    
    asyncio.run(test_export())
