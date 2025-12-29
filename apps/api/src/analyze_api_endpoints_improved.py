import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
改进版前端服务文件API端点分析器
"""

def extract_api_endpoints_improved(file_path: str) -> List[Dict[str, str]]:
    """从文件中提取API端点调用 - 改进版本"""
    endpoints = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return []
    
    # 清理注释和字符串字面量，避免误匹配
    lines = content.split('\n')
    clean_lines = []
    in_multiline_comment = False
    
    for line in lines:
        # 跳过单行注释
        if '//' in line:
            line = line[:line.index('//')]
        
        # 处理多行注释
        if '/*' in line and '*/' in line:
            # 单行内的多行注释
            start = line.index('/*')
            end = line.index('*/') + 2
            line = line[:start] + line[end:]
        elif '/*' in line:
            in_multiline_comment = True
            line = line[:line.index('/*')]
        elif '*/' in line and in_multiline_comment:
            in_multiline_comment = False
            line = line[line.index('*/') + 2:]
        elif in_multiline_comment:
            continue
            
        clean_lines.append(line)
    
    clean_content = '\n'.join(clean_lines)
    
    # 更精确的API调用模式
    patterns = [
        # apiClient.method('path', ...)
        r"apiClient\.(get|post|put|delete|patch)\s*\(\s*[`'\"]([^`'\"]+)[`'\"]\s*[,\)]",
        # axios.method('path', ...)
        r"axios\.(get|post|put|delete|patch)\s*\(\s*[`'\"]([^`'\"]+)[`'\"]\s*[,\)]",
        # this.client.method('path', ...)
        r"this\.client\.(get|post|put|delete|patch)\s*\(\s*[`'\"]([^`'\"]+)[`'\"]\s*[,\)]",
        # await fetch('path', {method: 'POST'})
        r"fetch\s*\(\s*[`'\"]([^`'\"]+)[`'\"]\s*,\s*\{[^}]*method\s*:\s*[`'\"]([^`'\"]+)[`'\"]*[^}]*\}",
        # await fetch('path') - 默认GET
        r"fetch\s*\(\s*[`'\"]([^`'\"]+)[`'\"]\s*[,\)]",
    ]
    
    for pattern_idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, clean_content, re.MULTILINE | re.IGNORECASE)
        
        for match in matches:
            if pattern_idx <= 2:  # apiClient, axios, this.client
                method, path = match
                if is_api_path(path):
                    endpoints.append({
                        'method': method.upper(),
                        'path': normalize_path(path),
                        'pattern_type': 'explicit_call',
                        'confidence': 'high'
                    })
            elif pattern_idx == 3:  # fetch with explicit method
                path, method = match
                if is_api_path(path):
                    endpoints.append({
                        'method': method.upper(),
                        'path': normalize_path(path),
                        'pattern_type': 'fetch_explicit',
                        'confidence': 'high'
                    })
            elif pattern_idx == 4:  # fetch without method (GET)
                path = match
                if is_api_path(path):
                    endpoints.append({
                        'method': 'GET',
                        'path': normalize_path(path),
                        'pattern_type': 'fetch_implicit',
                        'confidence': 'medium'
                    })
    
    # 去重
    seen = set()
    unique_endpoints = []
    for endpoint in endpoints:
        key = f"{endpoint['method']}:{endpoint['path']}"
        if key not in seen:
            seen.add(key)
            unique_endpoints.append(endpoint)
    
    return unique_endpoints

def is_api_path(path: str) -> bool:
    """判断是否为API路径"""
    api_indicators = [
        '/api/v1/',
        '/api/',
        '/mcp/',
        '/health',
        '/metrics',
        '/status',
        '/workflows',
        '/agent',
        '/monitoring',
        '/documents',
        '/rag/',
        '/supervisor',
        '/security',
        '/events',
        '/fine-tuning',
        '/memories',
        '/entities',
        '/platform',
        '/reasoning'
    ]
    
    return any(indicator in path.lower() for indicator in api_indicators)

def normalize_path(path: str) -> str:
    """规范化路径"""
    # 移除baseURL变量拼接
    path = path.replace('${this.baseUrl}', '')
    path = path.replace('${API_BASE_URL}', '')
    
    # 确保以/开头
    if not path.startswith('/'):
        path = '/' + path
        
    return path

def find_service_usage(service_name: str, pages_dir: str) -> Dict[str, List[str]]:
    """查找服务的使用情况"""
    usage_info = {
        'pages': [],
        'components': [],
        'hooks': []
    }
    
    pages_path = Path(pages_dir)
    if not pages_path.exists():
        return usage_info
    
    base_name = service_name.replace('.ts', '').replace('Service', '').replace('Api', '')
    
    # 可能的导入模式
    import_patterns = [
        rf"import.*{base_name}Service",
        rf"import.*{base_name}Api", 
        rf"import.*{base_name}",
        rf"from.*{service_name.replace('.ts', '')}",
    ]
    
    try:
        # 搜索页面文件
        for file_path in pages_path.rglob("*.tsx"):
            if 'pages' in str(file_path):
                file_type = 'pages'
            elif 'components' in str(file_path):
                file_type = 'components'  
            elif 'hooks' in str(file_path):
                file_type = 'hooks'
            else:
                file_type = 'pages'
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in import_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        relative_path = str(file_path.relative_to(pages_path))
                        usage_info[file_type].append(relative_path)
                        break
                        
            except Exception:
                continue
    except Exception:
        logger.exception("扫描页面文件失败", exc_info=True)
    
    return usage_info

def generate_summary_report(results: Dict, pages_dir: str) -> str:
    """生成汇总报告"""
    report = []
    report.append("# 前端服务API端点调用分析汇总\n")
    
    # 统计信息
    services_with_apis = len([s for s in results if results[s]])
    total_endpoints = sum(len(endpoints) for endpoints in results.values())
    
    report.append(f"## 总览")
    report.append(f"- 分析服务文件数量: {len(results)}")
    report.append(f"- 包含API调用的服务数量: {services_with_apis}")
    report.append(f"- 总API端点数量: {total_endpoints}\n")
    
    # 按API数量排序的服务列表
    sorted_services = sorted(
        [(name, endpoints) for name, endpoints in results.items() if endpoints],
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    report.append("## 服务API调用概览\n")
    report.append("| 服务文件 | API端点数量 | 主要API类型 | 相关页面 |")
    report.append("|---------|------------|------------|---------|")
    
    for service_name, endpoints in sorted_services:
        # 获取API类型统计
        api_types = {}
        for ep in endpoints:
            path_parts = ep['path'].strip('/').split('/')
            if len(path_parts) >= 2:
                api_type = f"/{path_parts[0]}/{path_parts[1]}"
            else:
                api_type = f"/{path_parts[0]}" if path_parts else "/"
            api_types[api_type] = api_types.get(api_type, 0) + 1
        
        main_api_type = max(api_types.items(), key=lambda x: x[1])[0] if api_types else "N/A"
        
        # 获取相关页面
        usage_info = find_service_usage(service_name, pages_dir)
        page_count = len(usage_info['pages'])
        page_info = f"{page_count} 个页面" if page_count > 0 else "无"
        
        report.append(f"| {service_name} | {len(endpoints)} | {main_api_type} | {page_info} |")
    
    report.append("\n## API端点分类统计\n")
    
    # 统计API路径前缀
    path_stats = {}
    method_stats = {}
    confidence_stats = {}
    
    for endpoints in results.values():
        for ep in endpoints:
            # 路径前缀统计
            path_parts = ep['path'].strip('/').split('/')
            if len(path_parts) >= 2:
                prefix = f"/{path_parts[0]}/{path_parts[1]}"
            else:
                prefix = f"/{path_parts[0]}" if path_parts else "/"
            path_stats[prefix] = path_stats.get(prefix, 0) + 1
            
            # HTTP方法统计
            method_stats[ep['method']] = method_stats.get(ep['method'], 0) + 1
            
            # 置信度统计
            confidence = ep.get('confidence', 'unknown')
            confidence_stats[confidence] = confidence_stats.get(confidence, 0) + 1
    
    report.append("### API路径前缀分布:")
    for prefix, count in sorted(path_stats.items(), key=lambda x: x[1], reverse=True):
        report.append(f"- `{prefix}`: {count} 个端点")
    
    report.append("\n### HTTP方法分布:")
    for method, count in sorted(method_stats.items()):
        report.append(f"- `{method}`: {count} 个端点")
    
    report.append("\n### 检测置信度分布:")
    for confidence, count in sorted(confidence_stats.items()):
        report.append(f"- {confidence}: {count} 个端点")
    
    # 详细服务映射
    report.append("\n## 详细服务API映射\n")
    
    for service_name, endpoints in sorted_services:
        report.append(f"### {service_name}")
        
        usage_info = find_service_usage(service_name, pages_dir)
        if usage_info['pages']:
            report.append("**相关页面:**")
            for page in usage_info['pages'][:3]:
                report.append(f"- {page}")
            if len(usage_info['pages']) > 3:
                report.append(f"- ... 还有 {len(usage_info['pages']) - 3} 个")
        
        report.append("**API端点:**")
        # 按HTTP方法和路径排序
        sorted_endpoints = sorted(endpoints, key=lambda x: (x['method'], x['path']))
        for ep in sorted_endpoints:
            confidence_badge = "🟢" if ep.get('confidence') == 'high' else "🟡" if ep.get('confidence') == 'medium' else "🔴"
            report.append(f"- {confidence_badge} `{ep['method']} {ep['path']}` ({ep['pattern_type']})")
        
        report.append("")
    
    return "\n".join(report)

def main():
    """主函数"""
    base_path = "/Users/runout/awork/code/my_git/agent/apps/web/src"
    services_dir = os.path.join(base_path, "services")
    pages_dir = os.path.join(base_path, "pages")
    
    logger.info("开始改进版API端点分析...")
    
    results = {}
    services_path = Path(services_dir)
    
    if not services_path.exists():
        logger.info(f"目录不存在: {services_dir}")
        return
    
    for file_path in services_path.glob("*.ts"):
        file_name = file_path.name
        logger.info(f"分析文件: {file_name}")
        
        endpoints = extract_api_endpoints_improved(str(file_path))
        results[file_name] = endpoints
        
        if endpoints:
            logger.info(f"  找到 {len(endpoints)} 个API端点")
    
    # 生成改进的报告
    report = generate_summary_report(results, pages_dir)
    
    # 保存报告
    report_file = "/Users/runout/awork/code/my_git/agent/frontend_api_analysis_improved.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n改进版分析完成! 报告已保存到: {report_file}")
    
    # 保存详细数据
    json_file = "/Users/runout/awork/code/my_git/agent/frontend_api_analysis_improved.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细数据已保存到: {json_file}")

if __name__ == "__main__":
    setup_logging()
    main()
