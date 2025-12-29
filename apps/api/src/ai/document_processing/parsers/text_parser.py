"""文本文档解析器"""

import re
from pathlib import Path
from typing import List, Dict, Any
import chardet
import markdown
from bs4 import BeautifulSoup
from .base_parser import BaseParser, ParsedDocument, ParsedElement

class TextParser(BaseParser):
    """文本文档解析器
    
    支持纯文本、Markdown、HTML等文本格式
    """
    
    SUPPORTED_EXTENSIONS = [".txt", ".md", ".markdown", ".rst", ".log", ".html", ".htm"]
    
    def __init__(
        self,
        parse_markdown: bool = True,
        extract_links: bool = True,
        extract_headers: bool = True
    ):
        """初始化文本解析器
        
        Args:
            parse_markdown: 是否解析Markdown语法
            extract_links: 是否提取链接
            extract_headers: 是否提取标题结构
        """
        super().__init__()
        self.parse_markdown = parse_markdown
        self.extract_links = extract_links
        self.extract_headers = extract_headers
    
    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析文本文档
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            解析后的文档
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc_id = self.generate_doc_id(file_path)
        metadata = self.extract_metadata(file_path)
        
        # 检测编码
        encoding = self._detect_encoding(file_path)
        metadata["encoding"] = encoding
        
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # 降级到二进制模式
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
        
        # 根据文件类型选择解析方法
        ext = file_path.suffix.lower()
        
        if ext in [".md", ".markdown"]:
            elements = await self._parse_markdown(content)
            file_type = "markdown"
        elif ext in [".html", ".htm"]:
            elements = await self._parse_html(content)
            file_type = "html"
        else:
            elements = await self._parse_plain_text(content)
            file_type = "text"
        
        # 计算文本统计信息
        stats = self._calculate_text_stats(content)
        metadata.update(stats)
        
        return ParsedDocument(
            doc_id=doc_id,
            file_path=str(file_path),
            file_type=file_type,
            elements=elements,
            metadata=metadata
        )
    
    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码
        
        Args:
            file_path: 文件路径
            
        Returns:
            编码类型
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    async def _parse_markdown(self, content: str) -> List[ParsedElement]:
        """解析Markdown文档
        
        Args:
            content: Markdown内容
            
        Returns:
            解析的元素列表
        """
        elements = []
        
        if self.parse_markdown:
            # 转换Markdown为HTML
            html = markdown.markdown(
                content,
                extensions=['extra', 'codehilite', 'tables', 'toc']
            )
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # 提取标题结构
            if self.extract_headers:
                headers = []
                for level in range(1, 7):
                    for header in soup.find_all(f'h{level}'):
                        headers.append({
                            "level": level,
                            "text": header.get_text().strip(),
                        })
                
                if headers:
                    # 创建目录结构
                    toc_lines = []
                    for header in headers:
                        indent = "  " * (header["level"] - 1)
                        toc_lines.append(f"{indent}- {header['text']}")
                    
                    elements.append(ParsedElement(
                        content="\n".join(toc_lines),
                        element_type="toc",
                        metadata={
                            "headers": headers,
                            "header_count": len(headers),
                        }
                    ))
            
            # 提取代码块
            code_blocks = soup.find_all('code')
            if code_blocks:
                for idx, code in enumerate(code_blocks[:10]):  # 限制前10个代码块
                    # 检查是否为代码块（而不是内联代码）
                    parent = code.parent
                    if parent and parent.name == 'pre':
                        elements.append(ParsedElement(
                            content=code.get_text(),
                            element_type="code_block",
                            metadata={
                                "block_index": idx,
                                "language": code.get('class', [''])[0].replace('language-', '') if code.get('class') else 'text',
                            }
                        ))
            
            # 提取表格
            tables = soup.find_all('table')
            for idx, table in enumerate(tables):
                table_data = self._parse_html_table(table)
                if table_data:
                    elements.append(ParsedElement(
                        content=table_data["content"],
                        element_type="table",
                        metadata={
                            "table_index": idx,
                            **table_data["metadata"]
                        }
                    ))
            
            # 提取链接
            if self.extract_links:
                links = []
                for link in soup.find_all('a', href=True):
                    links.append({
                        "text": link.get_text().strip(),
                        "url": link['href']
                    })
                
                if links:
                    link_lines = [f"- [{link['text']}]({link['url']})" for link in links[:20]]
                    elements.append(ParsedElement(
                        content="\n".join(link_lines),
                        element_type="links",
                        metadata={
                            "total_links": len(links),
                            "links": links[:20],  # 存储前20个链接
                        }
                    ))
            
            # 提取纯文本内容
            text_content = soup.get_text(separator='\n', strip=True)
        else:
            text_content = content
        
        # 添加主要文本内容
        elements.append(ParsedElement(
            content=text_content,
            element_type="text",
            metadata={
                "format": "markdown",
                "char_count": len(text_content),
                "line_count": len(text_content.splitlines()),
            }
        ))
        
        return elements
    
    async def _parse_html(self, content: str) -> List[ParsedElement]:
        """解析HTML文档
        
        Args:
            content: HTML内容
            
        Returns:
            解析的元素列表
        """
        elements = []
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # 移除脚本和样式标签
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 提取元数据
        metadata_elem = ParsedElement(
            content="HTML Metadata",
            element_type="metadata",
            metadata={}
        )
        
        # 提取标题
        title = soup.find('title')
        if title:
            metadata_elem.metadata["title"] = title.get_text().strip()
        
        # 提取meta标签
        meta_tags = {}
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                meta_tags[meta['name']] = meta.get('content', '')
            elif meta.get('property'):
                meta_tags[meta['property']] = meta.get('content', '')
        
        if meta_tags:
            metadata_elem.metadata["meta_tags"] = meta_tags
        
        if metadata_elem.metadata:
            elements.append(metadata_elem)
        
        # 提取标题结构
        if self.extract_headers:
            headers = []
            for level in range(1, 7):
                for header in soup.find_all(f'h{level}'):
                    headers.append({
                        "level": level,
                        "text": header.get_text().strip(),
                    })
            
            if headers:
                toc_lines = []
                for header in headers:
                    indent = "  " * (header["level"] - 1)
                    toc_lines.append(f"{indent}- {header['text']}")
                
                elements.append(ParsedElement(
                    content="\n".join(toc_lines),
                    element_type="toc",
                    metadata={
                        "headers": headers,
                        "header_count": len(headers),
                    }
                ))
        
        # 提取表格
        tables = soup.find_all('table')
        for idx, table in enumerate(tables):
            table_data = self._parse_html_table(table)
            if table_data:
                elements.append(ParsedElement(
                    content=table_data["content"],
                    element_type="table",
                    metadata={
                        "table_index": idx,
                        **table_data["metadata"]
                    }
                ))
        
        # 提取链接
        if self.extract_links:
            links = []
            for link in soup.find_all('a', href=True):
                links.append({
                    "text": link.get_text().strip() or link['href'],
                    "url": link['href']
                })
            
            if links:
                unique_links = []
                seen = set()
                for link in links:
                    if link['url'] not in seen:
                        unique_links.append(link)
                        seen.add(link['url'])
                
                link_lines = [f"- {link['text']}: {link['url']}" for link in unique_links[:20]]
                elements.append(ParsedElement(
                    content="\n".join(link_lines),
                    element_type="links",
                    metadata={
                        "total_links": len(links),
                        "unique_links": len(unique_links),
                        "links": unique_links[:20],
                    }
                ))
        
        # 提取纯文本内容
        text_content = soup.get_text(separator='\n', strip=True)
        elements.append(ParsedElement(
            content=text_content,
            element_type="text",
            metadata={
                "format": "html",
                "char_count": len(text_content),
                "line_count": len(text_content.splitlines()),
            }
        ))
        
        return elements
    
    async def _parse_plain_text(self, content: str) -> List[ParsedElement]:
        """解析纯文本文档
        
        Args:
            content: 文本内容
            
        Returns:
            解析的元素列表
        """
        elements = []
        
        # 检测是否为日志文件
        if self._is_log_file(content):
            elements.extend(await self._parse_log_file(content))
        
        # 检测段落
        paragraphs = self._extract_paragraphs(content)
        if len(paragraphs) > 1:
            # 创建段落摘要
            summary_lines = []
            for idx, para in enumerate(paragraphs[:5], 1):
                preview = para[:100] + "..." if len(para) > 100 else para
                summary_lines.append(f"Paragraph {idx}: {preview}")
            
            if len(paragraphs) > 5:
                summary_lines.append(f"... and {len(paragraphs) - 5} more paragraphs")
            
            elements.append(ParsedElement(
                content="\n".join(summary_lines),
                element_type="summary",
                metadata={
                    "paragraph_count": len(paragraphs),
                }
            ))
        
        # 添加完整文本
        elements.append(ParsedElement(
            content=content,
            element_type="text",
            metadata={
                "format": "plain",
                "char_count": len(content),
                "line_count": len(content.splitlines()),
                "word_count": len(content.split()),
            }
        ))
        
        return elements
    
    def _parse_html_table(self, table) -> Dict[str, Any]:
        """解析HTML表格
        
        Args:
            table: BeautifulSoup表格对象
            
        Returns:
            表格数据字典
        """
        headers = []
        rows = []
        
        # 提取表头
        thead = table.find('thead')
        if thead:
            for th in thead.find_all('th'):
                headers.append(th.get_text().strip())
        else:
            # 尝试从第一行提取
            first_row = table.find('tr')
            if first_row:
                for th in first_row.find_all('th'):
                    headers.append(th.get_text().strip())
        
        # 提取数据行
        tbody = table.find('tbody') or table
        for tr in tbody.find_all('tr'):
            row = []
            for td in tr.find_all(['td', 'th']):
                row.append(td.get_text().strip())
            if row and (not headers or len(row) > 0):
                rows.append(row)
        
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]
        
        # 格式化为文本
        content_lines = []
        if headers:
            content_lines.append(" | ".join(headers))
            content_lines.append("-" * (len(" | ".join(headers))))
        
        for row in rows[:20]:  # 限制显示前20行
            content_lines.append(" | ".join(row))
        
        if len(rows) > 20:
            content_lines.append(f"... and {len(rows) - 20} more rows")
        
        return {
            "content": "\n".join(content_lines),
            "metadata": {
                "headers": headers,
                "row_count": len(rows),
                "column_count": len(headers) if headers else (len(rows[0]) if rows else 0),
            }
        }
    
    def _is_log_file(self, content: str) -> bool:
        """检测是否为日志文件
        
        Args:
            content: 文件内容
            
        Returns:
            是否为日志文件
        """
        # 简单的日志模式检测
        log_patterns = [
            r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}',  # ISO日期时间
            r'\[(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\]',  # 日志级别
            r'^\d+\s+\w+\s+\d+\s+\d+:\d+:\d+',  # Unix日志格式
        ]
        
        lines = content.splitlines()[:10]  # 检查前10行
        for line in lines:
            for pattern in log_patterns:
                if re.search(pattern, line):
                    return True
        
        return False
    
    async def _parse_log_file(self, content: str) -> List[ParsedElement]:
        """解析日志文件
        
        Args:
            content: 日志内容
            
        Returns:
            解析的元素列表
        """
        elements = []
        lines = content.splitlines()
        
        # 统计日志级别
        log_levels = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }
        
        for line in lines:
            for level in log_levels:
                if level in line.upper():
                    log_levels[level] += 1
        
        # 提取错误和警告
        errors = []
        warnings = []
        
        for idx, line in enumerate(lines):
            if "ERROR" in line.upper():
                errors.append({"line": idx + 1, "content": line[:200]})
            elif "WARNING" in line.upper():
                warnings.append({"line": idx + 1, "content": line[:200]})
        
        # 创建日志摘要
        summary_lines = ["## Log Summary"]
        summary_lines.append(f"Total lines: {len(lines)}")
        summary_lines.append("\n### Log Levels:")
        for level, count in log_levels.items():
            if count > 0:
                summary_lines.append(f"- {level}: {count}")
        
        if errors:
            summary_lines.append(f"\n### Errors ({len(errors)} found):")
            for err in errors[:5]:
                summary_lines.append(f"- Line {err['line']}: {err['content']}")
            if len(errors) > 5:
                summary_lines.append(f"... and {len(errors) - 5} more errors")
        
        if warnings:
            summary_lines.append(f"\n### Warnings ({len(warnings)} found):")
            for warn in warnings[:5]:
                summary_lines.append(f"- Line {warn['line']}: {warn['content']}")
            if len(warnings) > 5:
                summary_lines.append(f"... and {len(warnings) - 5} more warnings")
        
        elements.append(ParsedElement(
            content="\n".join(summary_lines),
            element_type="log_summary",
            metadata={
                "log_levels": log_levels,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "total_lines": len(lines),
            }
        ))
        
        return elements
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """提取段落
        
        Args:
            content: 文本内容
            
        Returns:
            段落列表
        """
        # 使用双换行符分割段落
        paragraphs = re.split(r'\n\s*\n', content)
        
        # 过滤空段落和过短的段落
        meaningful_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # 至少50个字符
                meaningful_paragraphs.append(para)
        
        return meaningful_paragraphs
    
    def _calculate_text_stats(self, content: str) -> Dict[str, Any]:
        """计算文本统计信息
        
        Args:
            content: 文本内容
            
        Returns:
            统计信息字典
        """
        lines = content.splitlines()
        words = content.split()
        
        # 计算句子数（简单估计）
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        stats = {
            "total_lines": len(lines),
            "blank_lines": sum(1 for line in lines if not line.strip()),
            "total_words": len(words),
            "total_chars": len(content),
            "sentence_count": sentence_count,
        }
        
        # 计算平均值
        if stats["total_lines"] > stats["blank_lines"]:
            stats["avg_line_length"] = round(
                stats["total_chars"] / (stats["total_lines"] - stats["blank_lines"]), 2
            )
        
        if sentence_count > 0:
            stats["avg_sentence_length"] = round(
                stats["total_words"] / sentence_count, 2
            )
        
        return stats
