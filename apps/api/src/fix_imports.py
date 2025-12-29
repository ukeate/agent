import re
from pathlib import Path
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
批量修复API模块中的相对导入问题
将 from ... 格式的相对导入转换为 from src. 格式的绝对导入
"""

def fix_relative_imports(file_path):
    """修复单个文件中的相对导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复 from ...services 导入
    content = re.sub(r'from \.\.\.services\.', 'from src.services.', content)
    
    # 修复 from ..core 导入 
    content = re.sub(r'from \.\.\.core\.', 'from src.core.', content)
    content = re.sub(r'from \.\.core\.', 'from src.core.', content)
    
    # 修复 from ...ai 导入
    content = re.sub(r'from \.\.\.ai\.', 'from src.ai.', content)
    content = re.sub(r'from \.\.ai\.', 'from src.ai.', content)
    
    # 修复 from . 导入（同级模块）
    content = re.sub(r'from \.([a-zA-Z_][a-zA-Z0-9_]*)', r'from src.api.v1.\1', content)
    
    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("修复文件", path=str(file_path))
        return True
    return False

def main():
    # 获取所有API v1目录下的Python文件
    api_dir = Path(__file__).resolve().parent / "api" / "v1"
    api_files = list(api_dir.glob("*.py"))
    
    fixed_count = 0
    for file_path in api_files:
        if file_path.name == "__init__.py":
            continue
            
        if fix_relative_imports(file_path):
            fixed_count += 1
    
    logger.info("修复完成", fixed_count=fixed_count)

if __name__ == "__main__":
    setup_logging()
    main()
