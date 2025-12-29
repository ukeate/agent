"""
训练数据收集器实现

这个模块包含各种数据收集器的实现：
- API数据收集器
- 文件数据收集器  
- 网页数据收集器
"""

import asyncio
import aiohttp
import aiofiles
import json
import re
import chardet
from pathlib import Path
from typing import Dict, Any, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import trafilatura
from .core import DataCollector, DataSource, DataRecord

from src.core.logging import get_logger
logger = get_logger(__name__)

class APIDataCollector(DataCollector):
    """API数据收集器"""
    
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """从API收集数据"""
        
        config = self.source.config
        url = config.get('url')
        headers = config.get('headers', {})
        params = config.get('params', {})
        batch_size = config.get('batch_size', 100)
        max_retries = config.get('max_retries', 3)
        retry_delay = config.get('retry_delay', 1)
        
        if not url:
            raise ValueError("API URL is required")
        
        logger.info(f"Starting API data collection from {url}")
        
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers=headers
        ) as session:
            offset = 0
            consecutive_failures = 0
            
            while True:
                current_params = {**params, 'offset': offset, 'limit': batch_size}
                
                for attempt in range(max_retries):
                    try:
                        async with session.get(url, params=current_params) as response:
                            if response.status == 429:  # Rate limited
                                wait_time = int(response.headers.get('Retry-After', retry_delay * (2 ** attempt)))
                                logger.warning(f"Rate limited, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            
                            if response.status != 200:
                                logger.error(f"API request failed with status {response.status}")
                                consecutive_failures += 1
                                if consecutive_failures >= 3:
                                    return
                                break
                            
                            data = await response.json()
                            consecutive_failures = 0
                            
                            # 处理不同的API响应格式
                            items = self._extract_items_from_response(data)
                            
                            if not items:
                                logger.info("No more items found, stopping collection")
                                return
                            
                            for item in items:
                                record_id = self.generate_record_id(item)
                                record = DataRecord(
                                    record_id=record_id,
                                    source_id=self.source.source_id,
                                    raw_data=item,
                                    metadata={
                                        'api_url': url,
                                        'api_params': current_params,
                                        'response_status': response.status,
                                        'content_type': response.content_type,
                                        'collected_at': record.created_at.isoformat() if record.created_at else None
                                    }
                                )
                                yield record
                            
                            offset += batch_size
                            break  # Success, exit retry loop
                    
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout on attempt {attempt + 1}")
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    
                    except Exception as e:
                        logger.error(f"Error collecting data from API (attempt {attempt + 1}): {e}")
                        if attempt == max_retries - 1:
                            raise
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                
                # 避免API限流
                await asyncio.sleep(config.get('delay', 1))
    
    def _extract_items_from_response(self, data: Any) -> list:
        """从API响应中提取数据项"""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 常见的分页响应格式
            for key in ['data', 'items', 'results', 'records', 'entries']:
                if key in data:
                    items = data[key]
                    if isinstance(items, list):
                        return items
            # 如果没有找到标准字段，返回整个对象作为单个项
            return [data]
        else:
            return []

class FileDataCollector(DataCollector):
    """文件数据收集器"""
    
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """从文件收集数据"""
        
        config = self.source.config
        file_path = config.get('file_path')
        file_format = config.get('format', 'auto')
        encoding = config.get('encoding', 'auto')
        
        if not file_path:
            raise ValueError("File path is required")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Starting file data collection from {file_path}")
        
        # 自动检测编码
        if encoding == 'auto':
            encoding = await self._detect_encoding(path)
        
        # 自动检测格式
        if file_format == 'auto':
            file_format = self._detect_format(path)
        
        # 根据格式处理文件
        if file_format == 'json':
            async for record in self._process_json_file(path, encoding):
                yield record
        elif file_format == 'jsonl':
            async for record in self._process_jsonl_file(path, encoding):
                yield record
        elif file_format == 'csv':
            async for record in self._process_csv_file(path, encoding):
                yield record
        elif file_format == 'excel':
            async for record in self._process_excel_file(path):
                yield record
        elif file_format == 'text':
            async for record in self._process_text_file(path, encoding, config):
                yield record
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    async def _detect_encoding(self, path: Path) -> str:
        """检测文件编码"""
        try:
            with open(path, 'rb') as f:
                raw_data = f.read(10000)  # 读取前10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _detect_format(self, path: Path) -> str:
        """检测文件格式"""
        extension = path.suffix.lower()
        format_mapping = {
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.ndjson': 'jsonl',
            '.csv': 'csv',
            '.tsv': 'csv',
            '.txt': 'text',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.parquet': 'parquet'
        }
        return format_mapping.get(extension, 'text')
    
    async def _process_json_file(self, path: Path, encoding: str) -> AsyncIterator[DataRecord]:
        """处理JSON文件"""
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            content = await f.read()
            data = json.loads(content)
            
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]
            
            for i, item in enumerate(items):
                record_id = self.generate_record_id(item)
                record = DataRecord(
                    record_id=record_id,
                    source_id=self.source.source_id,
                    raw_data=item,
                    metadata={
                        'file_path': str(path),
                        'file_format': 'json',
                        'encoding': encoding,
                        'item_index': i,
                        'file_size': path.stat().st_size
                    }
                )
                yield record
    
    async def _process_jsonl_file(self, path: Path, encoding: str) -> AsyncIterator[DataRecord]:
        """处理JSONL文件"""
        line_number = 0
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            async for line in f:
                line = line.strip()
                line_number += 1
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    record_id = self.generate_record_id(item)
                    record = DataRecord(
                        record_id=record_id,
                        source_id=self.source.source_id,
                        raw_data=item,
                        metadata={
                            'file_path': str(path),
                            'file_format': 'jsonl',
                            'encoding': encoding,
                            'line_number': line_number,
                            'file_size': path.stat().st_size
                        }
                    )
                    yield record
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_number}: {line[:100]}... Error: {e}")
                    continue
    
    async def _process_csv_file(self, path: Path, encoding: str) -> AsyncIterator[DataRecord]:
        """处理CSV文件"""
        loop = asyncio.get_running_loop()
        
        # 使用线程池处理CSV文件
        with ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(
                executor, 
                lambda: pd.read_csv(path, encoding=encoding)
            )
            
            for index, row in df.iterrows():
                # 将NaN值转换为None
                item = row.where(pd.notnull(row), None).to_dict()
                record_id = self.generate_record_id(item)
                record = DataRecord(
                    record_id=record_id,
                    source_id=self.source.source_id,
                    raw_data=item,
                    metadata={
                        'file_path': str(path),
                        'file_format': 'csv',
                        'encoding': encoding,
                        'row_index': int(index),
                        'file_size': path.stat().st_size
                    }
                )
                yield record
    
    async def _process_excel_file(self, path: Path) -> AsyncIterator[DataRecord]:
        """处理Excel文件"""
        loop = asyncio.get_running_loop()
        
        with ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(
                executor,
                lambda: pd.read_excel(path)
            )
            
            for index, row in df.iterrows():
                item = row.where(pd.notnull(row), None).to_dict()
                record_id = self.generate_record_id(item)
                record = DataRecord(
                    record_id=record_id,
                    source_id=self.source.source_id,
                    raw_data=item,
                    metadata={
                        'file_path': str(path),
                        'file_format': 'excel',
                        'row_index': int(index),
                        'file_size': path.stat().st_size
                    }
                )
                yield record
    
    async def _process_text_file(
        self, 
        path: Path, 
        encoding: str, 
        config: Dict[str, Any]
    ) -> AsyncIterator[DataRecord]:
        """处理纯文本文件"""
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            content = await f.read()
            
            split_by = config.get('split_by', 'line')
            
            if split_by == 'line':
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        item = {
                            'text': line.strip(),
                            'line_number': i + 1
                        }
                        record_id = self.generate_record_id(item)
                        record = DataRecord(
                            record_id=record_id,
                            source_id=self.source.source_id,
                            raw_data=item,
                            metadata={
                                'file_path': str(path),
                                'file_format': 'text',
                                'encoding': encoding,
                                'split_by': split_by
                            }
                        )
                        yield record
            elif split_by == 'paragraph':
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for i, paragraph in enumerate(paragraphs):
                    item = {
                        'text': paragraph,
                        'paragraph_number': i + 1
                    }
                    record_id = self.generate_record_id(item)
                    record = DataRecord(
                        record_id=record_id,
                        source_id=self.source.source_id,
                        raw_data=item,
                        metadata={
                            'file_path': str(path),
                            'file_format': 'text',
                            'encoding': encoding,
                            'split_by': split_by
                        }
                    )
                    yield record

class WebDataCollector(DataCollector):
    """网页数据收集器"""
    
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """从网页收集数据"""
        
        config = self.source.config
        urls = config.get('urls', [])
        selectors = config.get('selectors', {})
        max_pages = config.get('max_pages', 100)
        use_trafilatura = config.get('use_trafilatura', True)
        delay = config.get('delay', 2)
        
        if not urls:
            raise ValueError("URLs are required for web data collection")
        
        logger.info(f"Starting web data collection from {len(urls)} URLs")
        
        connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=config.get('timeout', 30))
        
        headers = {
            'User-Agent': config.get(
                'user_agent', 
                'Mozilla/5.0 (compatible; DataCollector/1.0)'
            )
        }
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        ) as session:
            
            for i, url in enumerate(urls[:max_pages]):
                try:
                    async with session.get(url) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to fetch {url}: {response.status}")
                            continue
                        
                        content_type = response.headers.get('Content-Type', '')
                        if 'text/html' not in content_type:
                            logger.warning(f"Non-HTML content type for {url}: {content_type}")
                            continue
                        
                        html_content = await response.text()
                        
                        extracted_data = {
                            'url': url,
                            'status_code': response.status,
                            'content_type': content_type,
                            'content_length': len(html_content)
                        }
                        
                        # 使用BeautifulSoup解析HTML
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # 提取基本信息
                        extracted_data['title'] = soup.title.string if soup.title else ''
                        
                        # 使用trafilatura提取主要内容
                        if use_trafilatura:
                            main_text = trafilatura.extract(html_content)
                            extracted_data['main_text'] = main_text or ''
                        
                        # 使用自定义选择器提取数据
                        for field, selector in selectors.items():
                            try:
                                elements = soup.select(selector)
                                if elements:
                                    extracted_data[field] = [
                                        elem.get_text(strip=True) for elem in elements
                                    ]
                            except Exception as e:
                                logger.warning(f"Error extracting {field} from {url}: {e}")
                        
                        # 提取元数据
                        meta_tags = soup.find_all('meta')
                        meta_data = {}
                        for meta in meta_tags:
                            name = meta.get('name') or meta.get('property')
                            content = meta.get('content')
                            if name and content:
                                meta_data[name] = content
                        
                        extracted_data['meta_data'] = meta_data
                        
                        # 提取所有链接
                        links = [a.get('href') for a in soup.find_all('a', href=True)]
                        extracted_data['links'] = links[:50]  # 限制链接数量
                        
                        record_id = self.generate_record_id(extracted_data)
                        record = DataRecord(
                            record_id=record_id,
                            source_id=self.source.source_id,
                            raw_data=extracted_data,
                            metadata={
                                'url': url,
                                'collected_at': record.created_at.isoformat() if record.created_at else None,
                                'content_length': len(html_content),
                                'extraction_method': 'beautifulsoup+trafilatura' if use_trafilatura else 'beautifulsoup',
                                'page_index': i
                            }
                        )
                        yield record
                
                except Exception as e:
                    logger.error(f"Error collecting data from {url}: {e}")
                    continue
                
                # 避免过于频繁的请求
                await asyncio.sleep(delay)

class DatabaseDataCollector(DataCollector):
    """数据库数据收集器"""
    
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """从数据库收集数据"""
        
        config = self.source.config
        connection_string = config.get('connection_string')
        query = config.get('query')
        batch_size = config.get('batch_size', 1000)
        
        if not connection_string or not query:
            raise ValueError("Database connection string and query are required")
        
        logger.info("Starting database data collection")
        
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine

            engine = create_async_engine(connection_string)

            try:
                async with engine.connect() as conn:
                    result = await conn.stream(text(query))

                    batch = []
                    async for row in result:
                        item = dict(row._mapping)
                        batch.append(item)

                        if len(batch) >= batch_size:
                            for record in self._process_batch(batch):
                                yield record
                            batch = []

                    if batch:
                        for record in self._process_batch(batch):
                            yield record
            finally:
                await engine.dispose()

        except ImportError:
            raise ValueError("SQLAlchemy is required for database data collection")
        except Exception as e:
            logger.error(f"Database collection error: {e}")
            raise
    
    def _process_batch(self, batch: list) -> list:
        """处理数据批次"""
        records = []
        for i, item in enumerate(batch):
            record_id = self.generate_record_id(item)
            record = DataRecord(
                record_id=record_id,
                source_id=self.source.source_id,
                raw_data=item,
                metadata={
                    'collection_method': 'database',
                    'batch_index': i,
                    'connection_string': self.source.config.get('connection_string', '').split('@')[0] + '@***'  # 隐藏敏感信息
                }
            )
            records.append(record)
        return records

# 收集器工厂
class CollectorFactory:
    """数据收集器工厂"""
    
    _collectors = {
        'api': APIDataCollector,
        'file': FileDataCollector,
        'web': WebDataCollector,
        'database': DatabaseDataCollector
    }
    
    @classmethod
    def create_collector(cls, source: DataSource) -> DataCollector:
        """创建数据收集器"""
        source_type = source.source_type.value if hasattr(source.source_type, 'value') else source.source_type
        
        if source_type not in cls._collectors:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        collector_class = cls._collectors[source_type]
        return collector_class(source)
    
    @classmethod
    def register_collector(cls, source_type: str, collector_class: type):
        """注册自定义收集器"""
        cls._collectors[source_type] = collector_class
    
    @classmethod
    def get_supported_types(cls) -> list:
        """获取支持的数据源类型"""
        return list(cls._collectors.keys())
