import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
API端点与前端调用关系分析工具
分析后端API端点与前端页面/服务的对应关系
"""

class APIAnalyzer:
    def __init__(self, api_dir: str, frontend_dir: str):
        self.api_dir = Path(api_dir)
        self.frontend_dir = Path(frontend_dir)
        self.api_endpoints = {}
        self.frontend_calls = defaultdict(list)
        
    def extract_api_endpoints(self) -> Dict[str, List[Dict]]:
        """提取所有API端点"""
        endpoints = {}
        
        # 扫描api/v1目录下的所有Python文件
        for py_file in self.api_dir.glob("**/*.py"):
            if py_file.name.startswith("__") or "test" in py_file.name.lower():
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取FastAPI路由装饰器
                patterns = [
                    r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                    r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                ]
                
                file_endpoints = []
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for method, path in matches:
                        file_endpoints.append({
                            'method': method.upper(),
                            'path': path,
                            'file': str(py_file.relative_to(self.api_dir))
                        })
                
                if file_endpoints:
                    module_name = py_file.stem
                    endpoints[module_name] = file_endpoints
                    
            except Exception as e:
                logger.info(f"Error reading {py_file}: {e}")
                
        return endpoints
    
    def extract_frontend_api_calls(self) -> Dict[str, List[str]]:
        """提取前端API调用"""
        api_calls = defaultdict(list)
        
        # 扫描前端目录下的所有TypeScript/JavaScript文件
        for file_ext in ['*.ts', '*.tsx', '*.js', '*.jsx']:
            for ts_file in self.frontend_dir.glob(f"**/{file_ext}"):
                if "node_modules" in str(ts_file) or "test" in str(ts_file):
                    continue
                    
                try:
                    with open(ts_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 查找API调用模式
                    patterns = [
                        r'["\']([^"\']*\/api\/[^"\']+)["\']',  # API路径
                        r'fetch\s*\(\s*["`\']([^"`\']+)["`\']',  # fetch调用
                        r'axios\.(get|post|put|delete|patch)\s*\(\s*["`\']([^"`\']+)["`\']',  # axios调用
                        r'apiClient\.(get|post|put|delete|patch)\s*\(\s*["`\']([^"`\']+)["`\']',  # 自定义API客户端
                    ]
                    
                    found_calls = set()
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if pattern.count('(') == 2:  # 带方法的模式
                            for method, path in matches:
                                if '/api/' in path:
                                    found_calls.add(f"{method.upper()} {path}")
                        else:  # 只有路径的模式
                            for path in matches:
                                if isinstance(path, tuple):
                                    path = path[0] if path else ""
                                if '/api/' in path:
                                    found_calls.add(f"GET {path}")  # 默认GET
                    
                    if found_calls:
                        relative_path = str(ts_file.relative_to(self.frontend_dir))
                        api_calls[relative_path] = list(found_calls)
                        
                except Exception as e:
                    logger.info(f"Error reading {ts_file}: {e}")
                    
        return dict(api_calls)
    
    def analyze_usage(self) -> Dict[str, Dict]:
        """分析API使用情况"""
        api_endpoints = self.extract_api_endpoints()
        frontend_calls = self.extract_frontend_api_calls()
        
        # 创建端点到调用的映射
        endpoint_usage = {}
        
        # 扁平化所有API端点
        all_endpoints = []
        for module, endpoints in api_endpoints.items():
            for endpoint in endpoints:
                endpoint['module'] = module
                all_endpoints.append(endpoint)
        
        # 扁平化所有前端调用
        all_calls = []
        for file_path, calls in frontend_calls.items():
            for call in calls:
                all_calls.append({
                    'call': call,
                    'file': file_path
                })
        
        # 分析使用情况
        for endpoint in all_endpoints:
            endpoint_key = f"{endpoint['method']} {endpoint['path']}"
            used_by = []
            
            # 查找匹配的前端调用
            for call_info in all_calls:
                call = call_info['call']
                # 精确匹配或模糊匹配
                if (endpoint_key == call or 
                    endpoint['path'] in call or
                    call.split()[-1].endswith(endpoint['path'])):
                    used_by.append(call_info['file'])
            
            endpoint_usage[endpoint_key] = {
                'method': endpoint['method'],
                'path': endpoint['path'],
                'module': endpoint['module'],
                'file': endpoint['file'],
                'used_by': list(set(used_by)),  # 去重
                'is_used': len(used_by) > 0,
                'usage_count': len(set(used_by))
            }
        
        return endpoint_usage, api_endpoints, frontend_calls
    
    def generate_report(self) -> str:
        """生成分析报告"""
        usage_data, api_endpoints, frontend_calls = self.analyze_usage()
        
        # 统计数据
        total_endpoints = len(usage_data)
        used_endpoints = sum(1 for data in usage_data.values() if data['is_used'])
        unused_endpoints = total_endpoints - used_endpoints
        usage_rate = (used_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0
        
        report = []
        report.append("# API端点与前端页面对应关系分析报告")
        report.append("")
        report.append("## 总览统计")
        report.append(f"- 总API端点数: {total_endpoints}")
        report.append(f"- 已使用端点数: {used_endpoints}")
        report.append(f"- 未使用端点数: {unused_endpoints}")
        report.append(f"- 使用率: {usage_rate:.1f}%")
        report.append("")
        
        # 详细表格
        report.append("## 详细分析表格")
        report.append("")
        report.append("| API端点 | HTTP方法 | 路径 | 所属模块 | 是否使用 | 使用次数 | 被使用的文件 |")
        report.append("|---------|----------|------|----------|----------|----------|--------------|")
        
        # 按使用情况排序
        sorted_endpoints = sorted(usage_data.items(), key=lambda x: (not x[1]['is_used'], x[0]))
        
        for endpoint_key, data in sorted_endpoints:
            used_files = ', '.join(data['used_by'][:3])  # 最多显示3个文件
            if len(data['used_by']) > 3:
                used_files += f" (+{len(data['used_by'])-3}个)"
            
            status = "✅" if data['is_used'] else "❌"
            report.append(
                f"| {endpoint_key} | {data['method']} | {data['path']} | {data['module']} | "
                f"{status} | {data['usage_count']} | {used_files} |"
            )
        
        report.append("")
        
        # 未使用的端点列表
        unused = [k for k, v in usage_data.items() if not v['is_used']]
        if unused:
            report.append("## 未使用的API端点")
            report.append("")
            for endpoint in unused:
                data = usage_data[endpoint]
                report.append(f"- `{endpoint}` (模块: {data['module']}, 文件: {data['file']})")
            report.append("")
        
        # 按模块统计
        module_stats = defaultdict(lambda: {'total': 0, 'used': 0})
        for data in usage_data.values():
            module = data['module']
            module_stats[module]['total'] += 1
            if data['is_used']:
                module_stats[module]['used'] += 1
        
        report.append("## 按模块统计")
        report.append("")
        report.append("| 模块 | 总端点数 | 已使用 | 使用率 |")
        report.append("|------|----------|--------|--------|")
        
        for module, stats in sorted(module_stats.items()):
            rate = (stats['used'] / stats['total'] * 100) if stats['total'] > 0 else 0
            report.append(f"| {module} | {stats['total']} | {stats['used']} | {rate:.1f}% |")
        
        return "\n".join(report)

def main():
    # 设置路径
    current_dir = Path(__file__).parent
    api_dir = current_dir / "api" / "v1"
    frontend_dir = current_dir / ".." / ".." / "web" / "src"
    
    logger.info(f"API目录: {api_dir}")
    logger.info(f"前端目录: {frontend_dir}")
    
    if not api_dir.exists():
        logger.error(f"错误: API目录不存在 - {api_dir}")
        return
        
    if not frontend_dir.exists():
        logger.error(f"错误: 前端目录不存在 - {frontend_dir}")
        return
    
    # 创建分析器
    analyzer = APIAnalyzer(str(api_dir), str(frontend_dir))
    
    # 生成报告
    report = analyzer.generate_report()
    
    # 保存报告
    report_file = current_dir / "api_frontend_mapping_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"分析报告已保存到: {report_file}")
    logger.info("\n" + "="*50)
    logger.info(report)

if __name__ == "__main__":
    setup_logging()
    main()
