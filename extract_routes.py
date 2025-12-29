#!/usr/bin/env python3
"""
提取API路由信息的脚本
分析 apps/api/src/api/v1/ 目录下所有Python文件中定义的API路由端点
"""

import os
import re
import json
import ast
from pathlib import Path

def extract_router_prefix(file_content):
    """提取router的prefix"""
    pattern = r'router\s*=\s*APIRouter\s*\(\s*prefix\s*=\s*["\']([^"\']+)["\']'
    match = re.search(pattern, file_content)
    if match:
        return match.group(1)
    return ""

def extract_routes_from_file(file_path):
    """从单个文件中提取路由信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取router前缀
        router_prefix = extract_router_prefix(content)
        
        # 查找所有路由装饰器
        route_pattern = r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']*)["\']'
        matches = re.findall(route_pattern, content, re.MULTILINE)
        
        routes = []
        for method, path in matches:
            # 获取函数名 - 查找装饰器后的函数定义
            func_pattern = rf'@router\.{method}\s*\([^)]*\).*?\n(?:\s*.*?\n)*?\s*(?:async\s+)?def\s+(\w+)'
            func_match = re.search(func_pattern, content, re.DOTALL)
            if func_match:
                function_name = func_match.group(1)
            else:
                # 简单匹配，查找装饰器后紧接着的函数定义
                simple_pattern = rf'@router\.{method}.*?\n.*?def\s+(\w+)'
                simple_match = re.search(simple_pattern, content, re.DOTALL)
                function_name = simple_match.group(1) if simple_match else "unknown"
            
            # 构建完整路径
            full_path = f"/api/v1{router_prefix}{path}" if router_prefix else f"/api/v1{path}"
            
            routes.append({
                "path": full_path,
                "method": method.upper(),
                "function": function_name
            })
        
        return routes
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    """主函数"""
    api_dir = Path("apps/api/src/api/v1")
    
    if not api_dir.exists():
        print(f"Directory {api_dir} does not exist")
        return
    
    result = {}
    
    # 遍历所有Python文件
    for py_file in api_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        file_name = py_file.name
        print(f"Processing {file_name}...")
        
        routes = extract_routes_from_file(py_file)
        
        if routes:
            result[file_name] = {
                "routes": routes
            }
    
    # 输出结果到文件
    output_file = "api_routes_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    total_routes = sum(len(data["routes"]) for data in result.values())
    print(f"路由提取完成!")
    print(f"总计: {len(result)} 个文件, {total_routes} 个路由端点")
    print(f"结果已保存到: {output_file}")
    
    # 输出概览信息
    print("\n文件路由数量统计:")
    for file_name, data in sorted(result.items(), key=lambda x: len(x[1]["routes"]), reverse=True):
        routes_count = len(data["routes"])
        print(f"  {file_name}: {routes_count} 个路由")
    
    # 按HTTP方法统计
    method_stats = {}
    for data in result.values():
        for route in data["routes"]:
            method = route["method"]
            method_stats[method] = method_stats.get(method, 0) + 1
    
    print(f"\nHTTP方法统计:")
    for method, count in sorted(method_stats.items()):
        print(f"  {method}: {count} 个端点")

if __name__ == "__main__":
    main()