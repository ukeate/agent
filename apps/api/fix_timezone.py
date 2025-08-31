#!/usr/bin/env python3
"""
批量修复时区处理不一致问题的脚本
"""

import os
import re
from pathlib import Path


def fix_timezone_issues():
    """修复时区问题"""
    
    # 需要修复的模式
    patterns = [
        (r'from datetime import datetime', r'from datetime import datetime\nfrom src.core.utils.timezone_utils import utc_now, utc_factory'),
        (r'datetime\.utcnow\(\)', r'utc_now()'),
        (r'datetime\.now\(\)', r'utc_now()'),
        (r'field\(default_factory=datetime\.now\)', r'field(default_factory=utc_factory)'),
        (r'datetime\.now\(timezone\.utc\)', r'utc_now()'),
    ]
    
    # 获取所有Python文件
    src_dir = Path('/Users/runout/awork/code/my_git/agent/apps/api/src')
    python_files = list(src_dir.rglob('*.py'))
    
    fixed_files = []
    
    for file_path in python_files:
        if file_path.name in ['fix_timezone.py', 'timezone_utils.py']:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 检查是否需要添加导入
            needs_import = False
            if 'utc_now()' in content or 'utc_now()' in content:
                if 'from src.core.utils.timezone_utils import' not in content:
                    needs_import = True
            
            # 应用修复模式
            for pattern, replacement in patterns:
                if needs_import and pattern == r'from datetime import datetime':
                    # 只在第一次匹配时添加导入
                    content = re.sub(pattern, replacement, content, count=1)
                    needs_import = False
                else:
                    content = re.sub(pattern, replacement, content)
            
            # 如果内容有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(str(file_path))
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    return fixed_files


if __name__ == "__main__":
    print("开始修复时区处理不一致问题...")
    fixed_files = fix_timezone_issues()
    
    if fixed_files:
        print(f"已修复 {len(fixed_files)} 个文件:")
        for file in fixed_files[:10]:  # 只显示前10个
            print(f"  - {file}")
        if len(fixed_files) > 10:
            print(f"  ... 和其他 {len(fixed_files) - 10} 个文件")
    else:
        print("没有发现需要修复的文件")
    
    print("修复完成!")