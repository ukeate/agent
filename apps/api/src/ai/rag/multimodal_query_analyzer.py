"""多模态查询类型分析器"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from .multimodal_config import QueryContext, QueryType

class MultimodalQueryAnalyzer:
    """查询类型智能识别"""
    
    # 视觉相关关键词
    VISUAL_KEYWORDS = [
        "图片", "图像", "照片", "截图", "图表", "图形",
        "image", "picture", "photo", "screenshot", "chart", 
        "diagram", "visual", "看", "显示", "展示"
    ]
    
    # 表格相关关键词
    TABLE_KEYWORDS = [
        "表格", "表", "数据", "统计", "列表", "Excel",
        "table", "spreadsheet", "data", "statistics", "list",
        "CSV", "行", "列", "单元格"
    ]
    
    # 文档相关关键词
    DOCUMENT_KEYWORDS = [
        "文档", "文件", "报告", "论文", "PDF", "Word",
        "document", "file", "report", "paper", "article",
        "页", "章节", "段落", "section"
    ]
    
    # 多模态组合关键词
    MIXED_KEYWORDS = [
        "包含", "结合", "综合", "所有", "全部",
        "contains", "includes", "combined", "all", "both",
        "以及", "还有", "和"
    ]
    
    def __init__(self):
        """初始化查询分析器"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        # 文件扩展名模式
        self.file_extension_pattern = re.compile(
            r'\.(pdf|docx?|xlsx?|csv|png|jpe?g|gif|txt|md|html?)(?:\s|$)',
            re.IGNORECASE
        )
        
        # 数字和范围模式
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        self.range_pattern = re.compile(r'\b\d+\s*[-到至]\s*\d+\b')
        
        # 问题类型模式
        self.question_pattern = re.compile(
            r'^(什么|何|哪|怎么|如何|为什么|是否|有没有|多少|'
            r'what|which|how|why|when|where|who|is|are|does|do)',
            re.IGNORECASE
        )
    
    def analyze_query(
        self,
        query: str,
        files: Optional[List[str]] = None
    ) -> QueryContext:
        """分析查询类型
        
        Args:
            query: 查询文本
            files: 上传的文件列表
            
        Returns:
            查询上下文
        """
        context = QueryContext(query=query)
        
        # 分析查询文本
        query_lower = query.lower()
        
        # 检查是否有上传文件
        if files:
            context.input_files = files
            file_types = self._analyze_file_types(files)
            context = self._update_context_from_files(context, file_types)
        
        # 分析查询类型
        query_type = self._determine_query_type(query_lower)
        context.query_type = query_type
        
        # 设置搜索需求
        context.requires_image_search = self._requires_image_search(query_lower, context)
        context.requires_table_search = self._requires_table_search(query_lower, context)
        
        # 提取过滤条件
        context.filters = self._extract_filters(query)
        
        # 提取检索参数
        context = self._extract_retrieval_params(query, context)
        
        return context
    
    def _analyze_file_types(self, files: List[str]) -> Dict[str, List[str]]:
        """分析文件类型
        
        Args:
            files: 文件路径列表
            
        Returns:
            按类型分组的文件字典
        """
        file_types = {
            "images": [],
            "documents": [],
            "tables": [],
            "others": []
        }
        
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            
            if ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
                file_types["images"].append(file_path)
            elif ext in [".pdf", ".docx", ".doc", ".txt", ".md", ".html"]:
                file_types["documents"].append(file_path)
            elif ext in [".xlsx", ".xls", ".csv", ".tsv"]:
                file_types["tables"].append(file_path)
            else:
                file_types["others"].append(file_path)
        
        return file_types
    
    def _update_context_from_files(
        self,
        context: QueryContext,
        file_types: Dict[str, List[str]]
    ) -> QueryContext:
        """根据文件类型更新上下文
        
        Args:
            context: 查询上下文
            file_types: 文件类型字典
            
        Returns:
            更新后的上下文
        """
        # 如果有图像文件，可能需要视觉搜索
        if file_types["images"]:
            context.requires_image_search = True
        
        # 如果有表格文件，可能需要表格搜索
        if file_types["tables"]:
            context.requires_table_search = True
        
        # 根据文件组合确定查询类型
        has_images = bool(file_types["images"])
        has_docs = bool(file_types["documents"])
        has_tables = bool(file_types["tables"])
        
        if has_images and (has_docs or has_tables):
            context.query_type = QueryType.MIXED
        elif has_images:
            context.query_type = QueryType.VISUAL
        elif has_docs or has_tables:
            context.query_type = QueryType.DOCUMENT
        
        return context
    
    def _determine_query_type(self, query_lower: str) -> QueryType:
        """确定查询类型
        
        Args:
            query_lower: 小写的查询文本
            
        Returns:
            查询类型
        """
        # 计算各类关键词的匹配分数
        visual_score = sum(1 for kw in self.VISUAL_KEYWORDS if kw in query_lower)
        table_score = sum(1 for kw in self.TABLE_KEYWORDS if kw in query_lower)
        doc_score = sum(1 for kw in self.DOCUMENT_KEYWORDS if kw in query_lower)
        mixed_score = sum(1 for kw in self.MIXED_KEYWORDS if kw in query_lower)
        
        # 检查文件扩展名
        if self.file_extension_pattern.search(query_lower):
            extensions = self.file_extension_pattern.findall(query_lower)
            for ext in extensions:
                ext = ext.lower()
                if ext in ["png", "jpg", "jpeg", "gif"]:
                    visual_score += 2
                elif ext in ["xlsx", "xls", "csv"]:
                    table_score += 2
                elif ext in ["pdf", "docx", "doc"]:
                    doc_score += 2
        
        # 如果有多种类型的关键词或混合关键词
        type_counts = sum([visual_score > 0, table_score > 0, doc_score > 0])
        if type_counts > 1 or mixed_score > 0:
            return QueryType.MIXED
        
        # 根据最高分数确定类型
        scores = {
            QueryType.VISUAL: visual_score,
            QueryType.DOCUMENT: doc_score + table_score,
            QueryType.TEXT: 1  # 默认基础分
        }
        
        return max(scores, key=scores.get)
    
    def _requires_image_search(
        self,
        query_lower: str,
        context: QueryContext
    ) -> bool:
        """判断是否需要图像搜索
        
        Args:
            query_lower: 小写的查询文本
            context: 查询上下文
            
        Returns:
            是否需要图像搜索
        """
        # 如果查询类型是视觉或混合，需要图像搜索
        if context.query_type in [QueryType.VISUAL, QueryType.MIXED]:
            return True
        
        # 检查视觉关键词
        return any(kw in query_lower for kw in self.VISUAL_KEYWORDS)
    
    def _requires_table_search(
        self,
        query_lower: str,
        context: QueryContext
    ) -> bool:
        """判断是否需要表格搜索
        
        Args:
            query_lower: 小写的查询文本
            context: 查询上下文
            
        Returns:
            是否需要表格搜索
        """
        # 检查表格关键词
        if any(kw in query_lower for kw in self.TABLE_KEYWORDS):
            return True
        
        # 检查数字和范围（可能在查询表格数据）
        if self.range_pattern.search(query_lower):
            return True
        
        # 如果有多个数字，可能在查询数据
        numbers = self.number_pattern.findall(query_lower)
        if len(numbers) >= 3:
            return True
        
        return False
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """提取过滤条件
        
        Args:
            query: 查询文本
            
        Returns:
            过滤条件字典
        """
        filters = {}
        
        # 提取日期范围
        date_pattern = re.compile(
            r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
            re.IGNORECASE
        )
        dates = date_pattern.findall(query)
        if dates:
            filters["dates"] = dates
        
        # 提取文件类型
        extensions = self.file_extension_pattern.findall(query)
        if extensions:
            filters["file_types"] = [ext.lower() for ext in extensions]
        
        # 提取数字范围
        ranges = self.range_pattern.findall(query)
        if ranges:
            filters["ranges"] = ranges
        
        # 提取引号中的精确匹配词
        quote_pattern = re.compile(r'["\']([^"\']+)["\']')
        exact_matches = quote_pattern.findall(query)
        if exact_matches:
            filters["exact_match"] = exact_matches
        
        return filters
    
    def _extract_retrieval_params(
        self,
        query: str,
        context: QueryContext
    ) -> QueryContext:
        """提取检索参数
        
        Args:
            query: 查询文本
            context: 查询上下文
            
        Returns:
            更新后的上下文
        """
        # 提取top_k参数
        topk_pattern = re.compile(r'(?:top|前|最[多好])\s*(\d+)\s*[个条项]?', re.IGNORECASE)
        topk_match = topk_pattern.search(query)
        if topk_match:
            try:
                context.top_k = int(topk_match.group(1))
            except ValueError:
                logger.warning("捕获到ValueError，已继续执行", exc_info=True)
        
        # 提取相似度阈值
        similarity_pattern = re.compile(r'相似度?\s*[>≥>=]\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
        sim_match = similarity_pattern.search(query)
        if sim_match:
            try:
                context.similarity_threshold = float(sim_match.group(1))
                # 如果是百分比，转换为小数
                if context.similarity_threshold > 1:
                    context.similarity_threshold /= 100
            except ValueError:
                logger.warning("捕获到ValueError，已继续执行", exc_info=True)
        
        return context
    
    def get_query_complexity(self, context: QueryContext) -> float:
        """评估查询复杂度
        
        Args:
            context: 查询上下文
            
        Returns:
            复杂度分数(0-1)
        """
        complexity = 0.0
        
        # 基础复杂度：查询长度
        query_length = len(context.query)
        complexity += min(query_length / 200, 0.3)  # 最多贡献0.3
        
        # 查询类型复杂度
        type_complexity = {
            QueryType.TEXT: 0.1,
            QueryType.DOCUMENT: 0.2,
            QueryType.VISUAL: 0.25,
            QueryType.MIXED: 0.3
        }
        complexity += type_complexity.get(context.query_type, 0.1)
        
        # 多模态需求复杂度
        if context.requires_image_search:
            complexity += 0.15
        if context.requires_table_search:
            complexity += 0.15
        
        # 过滤条件复杂度
        if context.filters:
            complexity += min(len(context.filters) * 0.05, 0.15)
        
        # 输入文件复杂度
        if context.input_files:
            complexity += min(len(context.input_files) * 0.03, 0.15)
        
        return min(complexity, 1.0)
logger = get_logger(__name__)

from src.core.logging import get_logger
