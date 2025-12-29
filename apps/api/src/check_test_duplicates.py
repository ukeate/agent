import re
import os
from collections import defaultdict
from typing import Dict, List, Set
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
检查测试重复性脚本
确保没有重复的API测试，每个端点只有一个对应的测试用例
"""

def extract_endpoints_from_test_file(file_path: str) -> List[str]:
    """从测试文件中提取API端点"""
    if not os.path.exists(file_path):
        return []
    
    endpoints = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有的API端点URL模式
    url_patterns = [
        r'f"{BASE_URL}([^"]+)"',  # f"{BASE_URL}/api/endpoint"
        r'"[^"]*(/[^"]*)"',       # "/api/endpoint"
        r"'[^']*(/[^']*)'",       # '/api/endpoint'
    ]
    
    for pattern in url_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if match.startswith('/') and len(match) > 1:
                # 清理端点路径，移除参数
                endpoint = re.sub(r'\{[^}]+\}', '{id}', match)  # 统一路径参数
                endpoint = re.sub(r'\?.*$', '', endpoint)  # 移除查询参数
                endpoints.append(endpoint)
    
    return endpoints

def analyze_test_coverage():
    """分析测试覆盖情况"""
    test_files = {
        'test_detailed_api_logic.py': '核心API模块测试',
        'test_remaining_apis_logic.py': '剩余API模块测试', 
        'test_advanced_api_modules.py': '高级API模块测试',
        'test_complete_api_no_duplicates.py': '完整API测试套件-无重复版'
    }
    
    all_endpoints = defaultdict(list)  # endpoint -> [file1, file2, ...]
    file_endpoints = {}  # file -> [endpoints]
    
    logger.info("🔍 检查API测试重复性")
    logger.info("=" * 60)
    
    # 分析每个测试文件
    for test_file, description in test_files.items():
        endpoints = extract_endpoints_from_test_file(test_file)
        file_endpoints[test_file] = endpoints
        
        logger.info(f"\n📋 {test_file}")
        logger.info(f"描述: {description}")
        logger.info(f"端点数量: {len(endpoints)}")
        
        for endpoint in endpoints:
            all_endpoints[endpoint].append(test_file)
    
    # 查找重复的端点
    logger.info("\n" + "=" * 60)
    logger.info("🔍 重复端点检查")
    logger.info("=" * 60)
    
    duplicates = {}
    unique_endpoints = set()
    
    for endpoint, files in all_endpoints.items():
        if len(files) > 1:
            duplicates[endpoint] = files
        unique_endpoints.add(endpoint)
    
    if duplicates:
        logger.warning("⚠️  发现重复测试的端点:")
        for endpoint, files in duplicates.items():
            logger.info(f"  • {endpoint}")
            logger.info(f"    重复在: {', '.join(files)}")
    else:
        logger.info("✅ 没有发现重复测试的端点")
    
    # 统计信息
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试覆盖统计")
    logger.info("=" * 60)
    
    total_tests = sum(len(endpoints) for endpoints in file_endpoints.values())
    unique_count = len(unique_endpoints)
    duplicate_count = total_tests - unique_count
    
    logger.info(f"总测试数量: {total_tests}")
    logger.info(f"唯一端点数: {unique_count}")
    logger.info(f"重复测试数: {duplicate_count}")
    logger.info(f"重复率: {duplicate_count/total_tests*100:.1f}%")
    
    # 详细分析每个文件
    logger.info("\n" + "=" * 60)
    logger.info("📋 文件详细分析")
    logger.info("=" * 60)
    
    for file_name, endpoints in file_endpoints.items():
        unique_in_file = len(set(endpoints))
        duplicates_in_file = len(endpoints) - unique_in_file
        
        logger.info(f"\n{file_name}:")
        logger.info(f"  总端点: {len(endpoints)}")
        logger.info(f"  唯一端点: {unique_in_file}")
        logger.info(f"  文件内重复: {duplicates_in_file}")
    
    # 生成清理建议
    if duplicates:
        logger.info("\n" + "=" * 60)
        logger.info("🔧 清理建议")
        logger.info("=" * 60)
        
        logger.info("建议保留策略:")
        logger.info("1. test_detailed_api_logic.py - 保留核心模块测试")
        logger.info("2. test_remaining_apis_logic.py - 保留补充模块测试")  
        logger.info("3. test_advanced_api_modules.py - 保留高级模块测试")
        logger.info("\n具体重复端点处理:")
        for endpoint, files in duplicates.items():
            # 建议保留哪个文件中的测试
            if 'test_detailed_api_logic.py' in files:
                keep_file = 'test_detailed_api_logic.py'
            elif 'test_remaining_apis_logic.py' in files:
                keep_file = 'test_remaining_apis_logic.py'
            else:
                keep_file = files[0]
            
            remove_files = [f for f in files if f != keep_file]
            logger.info(f"  • {endpoint}")
            logger.info(f"    保留: {keep_file}")
            logger.info(f"    移除: {', '.join(remove_files)}")
    
    return {
        'total_tests': total_tests,
        'unique_endpoints': unique_count,
        'duplicates': duplicates,
        'file_endpoints': file_endpoints
    }

def create_unified_test_file():
    """创建统一的、无重复的测试文件"""
    analysis = analyze_test_coverage()
    
    if not analysis['duplicates']:
        logger.info("\n✅ 无需创建统一测试文件，当前测试已无重复")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("🔧 创建统一测试文件")
    logger.info("=" * 60)
    
    # 这里可以实现创建统一测试文件的逻辑
    # 但由于涉及复杂的代码合并，暂时只提供分析结果
    logger.info("统一测试文件创建功能开发中...")
    logger.info("当前建议: 手动移除重复测试，保留最完整的版本")

if __name__ == "__main__":
    setup_logging()
    try:
        analysis = analyze_test_coverage()
        
        # 如果有重复，询问是否创建统一版本
        if analysis['duplicates']:
            logger.info(f"\n发现 {len(analysis['duplicates'])} 个重复测试的端点")
            # create_unified_test_file()
        else:
            logger.info(f"\n🎉 测试覆盖良好！{analysis['unique_endpoints']} 个唯一端点，无重复测试")
            
    except Exception as e:
        logger.error(f"❌ 分析过程中出现错误: {e}")
