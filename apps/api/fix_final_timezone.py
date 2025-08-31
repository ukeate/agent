#!/usr/bin/env python3
"""
最终修复剩余的timezone问题
"""
import os
import re
from pathlib import Path

def fix_file(file_path: Path) -> bool:
    """修复单个文件的timezone问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 添加import（如果还没有）
        if 'from src.core.utils.timezone_utils import' not in content:
            if 'from datetime import datetime' in content:
                content = re.sub(
                    r'from datetime import datetime',
                    'from datetime import datetime\nfrom src.core.utils.timezone_utils import utc_now, utc_factory',
                    content,
                    count=1
                )
            elif 'import datetime' in content:
                content = re.sub(
                    r'import datetime',
                    'import datetime\nfrom src.core.utils.timezone_utils import utc_now, utc_factory',
                    content,
                    count=1
                )
            else:
                # 在第一个from/import语句后添加
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith(('from ', 'import ')) and not line.strip().startswith('"""') and not line.strip().startswith('#'):
                        lines.insert(i + 1, 'from src.core.utils.timezone_utils import utc_now, utc_factory')
                        content = '\n'.join(lines)
                        break
        
        # 替换utc_now()
        content = re.sub(r'datetime\.utcnow\(\)', 'utc_now()', content)
        
        # 替换Column中的default=utc_now
        content = re.sub(r'default=datetime\.utcnow', 'default=utc_now', content)
        
        # 替换onupdate=utc_now  
        content = re.sub(r'onupdate=datetime\.utcnow', 'onupdate=utc_now', content)
        
        # 替换field中的default_factory=utc_factory
        content = re.sub(r'default_factory=datetime\.now', 'default_factory=utc_factory', content)
        
        # 替换其他utc_now()用法
        content = re.sub(r'datetime\.now\(\)', 'utc_now()', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """主函数"""
    src_dir = Path(".")
    fixed_count = 0
    
    # 找到所有包含datetime.utcnow的Python文件
    py_files = list(src_dir.rglob("*.py"))
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'datetime.utcnow' in content:
                if fix_file(py_file):
                    fixed_count += 1
                    
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    print(f"\n修复完成！共修复了 {fixed_count} 个文件")

if __name__ == "__main__":
    main()