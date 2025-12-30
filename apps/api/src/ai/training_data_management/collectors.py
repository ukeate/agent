"""
数据收集器模块

支持从多种数据源收集数据：API、文件、网页等
"""

import asyncio
import aiohttp
import aiofiles
import pandas as pd
import json
import re
import hashlib
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import chardet
from .models import DataSource, DataRecord

from src.core.logging import get_logger
# 条件导入可选依赖
try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

class DataCollector(ABC):
    """数据收集器基类"""
    
    def __init__(self, source: DataSource):
        self.source = source
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """收集数据的抽象方法"""
        ...
    
    def generate_record_id(self, data: Dict[str, Any]) -> str:
        """生成唯一的记录ID"""
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

class APIDataCollector(DataCollector):
    """API数据收集器"""
    
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """从API收集数据"""
        
        config = self.source.config
        url = config.get('url')
        headers = config.get('headers', {})
        params = config.get('params', {})
        batch_size = config.get('batch_size', 100)
        
        if not url:
            raise ValueError("API URL is required")
        
        self.logger.info(f"Starting API data collection from {url}")
        
        # 使用连接池优化性能
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers=headers
        ) as session:
            offset = 0
            
            while True:
                current_params = {**params, 'offset': offset, 'limit': batch_size}
                
                try:
                    async with session.get(url, headers=headers, params=current_params) as response:
                        if response.status != 200:
                            self.logger.error(f"API request failed with status {response.status}")
                            break
                        
                        data = await response.json()
                        
                        # 处理不同的API响应格式
                        if isinstance(data, list):
                            items = data
                        elif isinstance(data, dict):
                            items = data.get('data', data.get('items', data.get('results', [])))
                        else:
                            items = []
                        
                        if not items:
                            break
                        
                        for item in items:
                            record_id = self.generate_record_id(item)
                            record = DataRecord(
                                record_id=record_id,
                                source_id=self.source.source_id,
                                raw_data=item,
                                metadata={
                                    'api_url': url,
                                    'api_params': current_params,
                                    'collected_at': utc_now().isoformat()
                                }
                            )
                            yield record
                        
                        offset += batch_size
                        
                        # 避免API限流
                        await asyncio.sleep(config.get('delay', 1))
                
                except Exception as e:
                    self.logger.error(f"Error collecting data from API: {e}")
                    break

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
        
        self.logger.info(f"Starting file data collection from {file_path}")
        
        # 自动检测编码
        if encoding == 'auto':
            with open(path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
        
        # 自动检测格式
        if file_format == 'auto':
            extension = path.suffix.lower()
            format_mapping = {
                '.json': 'json',
                '.jsonl': 'jsonl',
                '.csv': 'csv',
                '.txt': 'text',
                '.parquet': 'parquet'
            }
            file_format = format_mapping.get(extension, 'text')
        
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
        
        elif file_format == 'text':
            async for record in self._process_text_file(path, encoding, config):
                yield record
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    async def _process_json_file(self, path: Path, encoding: str) -> AsyncIterator[DataRecord]:
        """处理JSON文件"""
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            content = await f.read()
            data = json.loads(content)
            
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = [data]
            else:
                items = []
            
            for item in items:
                record_id = self.generate_record_id(item)
                record = DataRecord(
                    record_id=record_id,
                    source_id=self.source.source_id,
                    raw_data=item,
                    metadata={
                        'file_path': str(path),
                        'file_format': 'json',
                        'encoding': encoding
                    }
                )
                yield record
    
    async def _process_jsonl_file(self, path: Path, encoding: str) -> AsyncIterator[DataRecord]:
        """处理JSONL文件"""
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            async for line in f:
                line = line.strip()
                if line:
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
                                'encoding': encoding
                            }
                        )
                        yield record
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON line: {line[:100]}... Error: {e}")
    
    async def _process_csv_file(self, path: Path, encoding: str) -> AsyncIterator[DataRecord]:
        """处理CSV文件"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(
                executor, 
                lambda: pd.read_csv(path, encoding=encoding)
            )
            
            for _, row in df.iterrows():
                item = row.to_dict()
                record_id = self.generate_record_id(item)
                record = DataRecord(
                    record_id=record_id,
                    source_id=self.source.source_id,
                    raw_data=item,
                    metadata={
                        'file_path': str(path),
                        'file_format': 'csv',
                        'encoding': encoding
                    }
                )
                yield record
    
    async def _process_text_file(self, path: Path, encoding: str, config: Dict) -> AsyncIterator[DataRecord]:
        """处理文本文件"""
        async with aiofiles.open(path, 'r', encoding=encoding) as f:
            content = await f.read()
            
            # 按行或段落分割
            if config.get('split_by', 'line') == 'line':
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        item = {'text': line.strip(), 'line_number': i + 1}
                        record_id = self.generate_record_id(item)
                        record = DataRecord(
                            record_id=record_id,
                            source_id=self.source.source_id,
                            raw_data=item,
                            metadata={
                                'file_path': str(path),
                                'file_format': 'text',
                                'encoding': encoding
                            }
                        )
                        yield record

class WebDataCollector(DataCollector):
    """网页数据收集器"""
    
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """从网页收集数据"""
        
        if not HAS_BEAUTIFULSOUP:
            self.logger.error("BeautifulSoup is not installed. Install with: pip install beautifulsoup4")
            return
        
        config = self.source.config
        urls = config.get('urls', [])
        selectors = config.get('selectors', {})
        max_pages = config.get('max_pages', 100)
        
        if not urls:
            raise ValueError("URLs are required for web data collection")
        
        self.logger.info(f"Starting web data collection from {len(urls)} URLs")
        
        async with aiohttp.ClientSession() as session:
            for url in urls[:max_pages]:
                try:
                    async with session.get(url) as response:
                        if response.status != 200:
                            self.logger.warning(f"Failed to fetch {url}: {response.status}")
                            continue
                        
                        html_content = await response.text()
                        
                        # 使用trafilatura提取主要内容
                        main_text = ""
                        if HAS_TRAFILATURA:
                            main_text = trafilatura.extract(html_content) or ""
                        
                        # 使用BeautifulSoup进行自定义提取
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        extracted_data = {
                            'url': url,
                            'title': soup.title.string if soup.title else '',
                            'main_text': main_text,
                            'html_content': html_content[:1000] + '...' if len(html_content) > 1000 else html_content
                        }
                        
                        # 使用自定义选择器提取数据
                        for field, selector in selectors.items():
                            try:
                                elements = soup.select(selector)
                                if elements:
                                    extracted_data[field] = [elem.get_text(strip=True) for elem in elements]
                            except Exception as e:
                                self.logger.warning(f"Error extracting {field} from {url}: {e}")
                        
                        record_id = self.generate_record_id(extracted_data)
                        record = DataRecord(
                            record_id=record_id,
                            source_id=self.source.source_id,
                            raw_data=extracted_data,
                            metadata={
                                'url': url,
                                'collected_at': utc_now().isoformat(),
                                'content_length': len(html_content)
                            }
                        )
                        yield record
                
                except Exception as e:
                    self.logger.error(f"Error collecting data from {url}: {e}")
                
                # 避免过于频繁的请求
                await asyncio.sleep(config.get('delay', 2))

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
        
        # 这里简化实现，实际项目中需要支持不同数据库类型
        self.logger.info(f"Starting database data collection with query: {query}")
        
        return

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
        """创建数据收集器实例"""
        
        collector_class = cls._collectors.get(source.source_type)
        if not collector_class:
            raise ValueError(f"Unsupported source type: {source.source_type}")
        
        return collector_class(source)
    
    @classmethod
    def register_collector(cls, source_type: str, collector_class: type):
        """注册新的收集器类型"""
        cls._collectors[source_type] = collector_class
