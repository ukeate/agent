import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import asyncio
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
全面API使用情况分析工具
分析后端API端点与前端使用情况的对应关系
"""

class APIUsageAnalyzer:
    def __init__(self):
        self.api_endpoints = {}  # API模块 -> 端点列表
        self.frontend_services = {}  # 服务文件 -> API调用
        self.frontend_pages = {}  # 页面文件 -> 服务使用
        self.usage_mapping = defaultdict(list)  # API端点 -> 使用位置
        
    def extract_fastapi_routes(self, file_path: str) -> List[Dict]:
        """从FastAPI文件中提取路由信息"""
        routes = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找路由定义模式
            route_patterns = [
                r'@router\.(?P<method>get|post|put|delete|patch)\([\'"](?P<path>[^\'"]+)[\'"]',
                r'@app\.(?P<method>get|post|put|delete|patch)\([\'"](?P<path>[^\'"]+)[\'"]',
                r'router\.(?P<method>add_api_route)\([\'"](?P<path>[^\'"]+)[\'"]'
            ]
            
            # 查找函数定义
            func_pattern = r'(?:async\s+)?def\s+(\w+)\s*\([^)]*\):'
            functions = re.findall(func_pattern, content)
            
            for pattern in route_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    method = match.group('method')
                    path = match.group('path')
                    
                    # 寻找对应的函数名
                    func_name = None
                    start_pos = match.end()
                    next_func = re.search(func_pattern, content[start_pos:])
                    if next_func:
                        func_name = next_func.group(1)
                    
                    routes.append({
                        'method': method.upper(),
                        'path': path,
                        'function': func_name,
                        'file': os.path.basename(file_path)
                    })
                    
        except Exception as e:
            logger.error(f"分析API文件错误 {file_path}: {e}")
            
        return routes
    
    def extract_frontend_api_calls(self, file_path: str) -> List[Dict]:
        """从前端服务文件中提取API调用"""
        api_calls = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 匹配API调用模式
            patterns = [
                r'(?:get|post|put|delete|patch)\s*\(\s*[\'"`]([^\'"`]+)[\'"`]',
                r'fetch\s*\(\s*[\'"`]([^\'"`]+)[\'"`]',
                r'axios\.(?:get|post|put|delete|patch)\s*\(\s*[\'"`]([^\'"`]+)[\'"`]',
                r'apiClient\.(?:get|post|put|delete|patch)\s*\(\s*[\'"`]([^\'"`]+)[\'"`]',
                r'const\s+\w+\s*=\s*[\'"`]([^\'"`]*(?:/api/|/v1/)[^\'"`]+)[\'"`]'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if '/api/' in match or '/v1/' in match:
                        api_calls.append({
                            'endpoint': match,
                            'file': os.path.basename(file_path)
                        })
                        
        except Exception as e:
            logger.error(f"分析前端服务文件错误 {file_path}: {e}")
            
        return api_calls
    
    def extract_service_usage(self, file_path: str) -> List[Dict]:
        """从前端页面文件中提取服务使用"""
        service_usage = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 匹配服务导入和使用模式
            import_pattern = r'import\s+(?:\{[^}]+\}|\w+)\s+from\s+[\'"`]\.\.?/services/(\w+)[\'"`]'
            usage_pattern = r'(\w+Service)\.(\w+)\s*\('
            
            # 查找服务导入
            imports = re.findall(import_pattern, content)
            for service_import in imports:
                service_usage.append({
                    'service': service_import,
                    'type': 'import',
                    'file': os.path.basename(file_path)
                })
            
            # 查找服务使用
            usages = re.findall(usage_pattern, content)
            for service_name, method in usages:
                service_usage.append({
                    'service': service_name,
                    'method': method,
                    'type': 'usage',
                    'file': os.path.basename(file_path)
                })
                
        except Exception as e:
            logger.error(f"分析前端页面文件错误 {file_path}: {e}")
            
        return service_usage
    
    def normalize_endpoint(self, endpoint: str) -> str:
        """标准化端点路径"""
        # 移除查询参数
        endpoint = endpoint.split('?')[0]
        # 移除基础URL
        endpoint = re.sub(r'^https?://[^/]+', '', endpoint)
        # 确保以/开头
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        return endpoint
    
    def match_endpoints(self) -> Dict[str, Dict]:
        """匹配API端点与前端使用"""
        matching_results = {}
        
        for api_module, endpoints in self.api_endpoints.items():
            for endpoint_info in endpoints:
                endpoint_key = f"{endpoint_info['method']} {endpoint_info['path']}"
                matching_results[endpoint_key] = {
                    'api_info': endpoint_info,
                    'frontend_usage': [],
                    'is_used': False
                }
        
        # 检查前端API调用
        for service_file, api_calls in self.frontend_services.items():
            for call in api_calls:
                normalized_call = self.normalize_endpoint(call['endpoint'])
                
                # 尝试匹配端点
                for endpoint_key, endpoint_data in matching_results.items():
                    api_path = endpoint_data['api_info']['path']
                    
                    # 简单路径匹配
                    if api_path in normalized_call or normalized_call in api_path:
                        endpoint_data['frontend_usage'].append({
                            'service': service_file,
                            'call': call['endpoint']
                        })
                        endpoint_data['is_used'] = True
        
        return matching_results
    
    def analyze_directories(self):
        """分析所有相关目录"""
        logger.info("开始分析API和前端文件...")
        
        # 分析后端API文件
        api_dir = Path("./api/v1")
        if api_dir.exists():
            for api_file in api_dir.glob("*.py"):
                if api_file.name != "__init__.py":
                    routes = self.extract_fastapi_routes(str(api_file))
                    if routes:
                        self.api_endpoints[api_file.name] = routes
                        logger.info(f"✓ 分析API文件: {api_file.name} (发现 {len(routes)} 个端点)")
        
        # 分析前端服务文件
        services_dir = Path("/Users/runout/awork/code/my_git/agent/apps/web/src/services")
        if services_dir.exists():
            for service_file in services_dir.glob("*.ts"):
                api_calls = self.extract_frontend_api_calls(str(service_file))
                if api_calls:
                    self.frontend_services[service_file.name] = api_calls
                    logger.info(f"✓ 分析服务文件: {service_file.name} (发现 {len(api_calls)} 个API调用)")
        
        # 分析前端页面文件
        pages_dir = Path("/Users/runout/awork/code/my_git/agent/apps/web/src/pages")
        if pages_dir.exists():
            for page_file in pages_dir.glob("*.tsx"):
                service_usage = self.extract_service_usage(str(page_file))
                if service_usage:
                    self.frontend_pages[page_file.name] = service_usage
                    logger.info(f"✓ 分析页面文件: {page_file.name} (发现 {len(service_usage)} 个服务使用)")
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """生成详细报告"""
        matching_results = self.match_endpoints()
        
        # 统计数据
        total_endpoints = len(matching_results)
        used_endpoints = sum(1 for data in matching_results.values() if data['is_used'])
        unused_endpoints = total_endpoints - used_endpoints
        usage_rate = (used_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0
        
        # 按模块统计
        module_stats = defaultdict(lambda: {'total': 0, 'used': 0, 'unused': 0})
        for endpoint_key, data in matching_results.items():
            module = data['api_info']['file']
            module_stats[module]['total'] += 1
            if data['is_used']:
                module_stats[module]['used'] += 1
            else:
                module_stats[module]['unused'] += 1
        
        # 未使用的API端点
        unused_endpoints_list = []
        for endpoint_key, data in matching_results.items():
            if not data['is_used']:
                unused_endpoints_list.append({
                    'endpoint': endpoint_key,
                    'module': data['api_info']['file'],
                    'function': data['api_info']['function']
                })
        
        report = {
            'summary': {
                'total_api_modules': len(self.api_endpoints),
                'total_endpoints': total_endpoints,
                'used_endpoints': used_endpoints,
                'unused_endpoints': unused_endpoints,
                'usage_rate': round(usage_rate, 2),
                'total_frontend_services': len(self.frontend_services),
                'total_frontend_pages': len(self.frontend_pages)
            },
            'module_statistics': dict(module_stats),
            'endpoint_details': matching_results,
            'unused_endpoints': unused_endpoints_list,
            'api_endpoints_by_module': self.api_endpoints,
            'frontend_services': self.frontend_services,
            'frontend_pages_summary': {
                k: len(v) for k, v in self.frontend_pages.items()
            }
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """打印报告到控制台"""
        logger.info("\n" + "="*80)
        logger.info("API使用情况分析报告")
        logger.info("="*80)
        
        summary = report['summary']
        logger.info(f"\n📊 总体统计:")
        logger.info(f"   API模块数量: {summary['total_api_modules']}")
        logger.info(f"   API端点总数: {summary['total_endpoints']}")
        logger.info(f"   已使用端点: {summary['used_endpoints']}")
        logger.info(f"   未使用端点: {summary['unused_endpoints']}")
        logger.info(f"   使用率: {summary['usage_rate']}%")
        logger.info(f"   前端服务数量: {summary['total_frontend_services']}")
        logger.info(f"   前端页面数量: {summary['total_frontend_pages']}")
        
        logger.info(f"\n📈 各模块使用率统计:")
        for module, stats in report['module_statistics'].items():
            rate = (stats['used'] / stats['total'] * 100) if stats['total'] > 0 else 0
            logger.info(f"   {module}: {stats['used']}/{stats['total']} ({rate:.1f}%)")
        
        logger.error(f"\n❌ 未使用的API端点 ({len(report['unused_endpoints'])} 个):")
        for endpoint in report['unused_endpoints'][:20]:  # 只显示前20个
            logger.info(f"   {endpoint['endpoint']} - {endpoint['module']}")
        if len(report['unused_endpoints']) > 20:
            logger.info(f"   ... 还有 {len(report['unused_endpoints']) - 20} 个")
    
    def save_report(self, report: Dict[str, Any], filename: str = "api_usage_comprehensive_report.json"):
        """保存报告到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"\n💾 详细报告已保存到: {filename}")

async def main():
    analyzer = APIUsageAnalyzer()
    
    # 分析所有目录
    analyzer.analyze_directories()
    
    # 生成报告
    report = analyzer.generate_detailed_report()
    
    # 显示报告
    analyzer.print_report(report)
    
    # 保存报告
    analyzer.save_report(report)

if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
