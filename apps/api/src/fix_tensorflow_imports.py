import re
from pathlib import Path
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
修复TensorFlow导入的脚本
将所有直接的tensorflow导入替换为安全的延迟导入方式
"""

def fix_tensorflow_imports(file_path: Path):
    """修复单个文件中的TensorFlow导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 记录是否进行了修改
        modified = False
        original_content = content
        
        # 1. 替换直接导入 import tensorflow as tf
        if 'import tensorflow as tf' in content:
            content = content.replace(
                'import tensorflow as tf',
                'from src.core.tensorflow_config import tensorflow_lazy'
            )
            modified = True
            logger.info("已替换导入", detail="import tensorflow as tf")
        
        # 2. 替换所有 tf. 调用为 tensorflow_lazy.tf.
        # 但要避免替换字符串和注释中的内容
        if modified or 'tf.' in content:
            # 使用正则表达式替换tf.调用，但避免在字符串中替换
            pattern = r'\btf\.'
            def replace_tf_calls(match):
                return 'tensorflow_lazy.tf.'
            
            # 简单替换，可能需要更精确的处理
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                # 跳过注释和字符串
                if line.strip().startswith('#') or '"""' in line or "'''" in line:
                    new_lines.append(line)
                    continue
                
                # 在代码行中替换tf.调用
                if 'tf.' in line and not line.strip().startswith('#'):
                    line = re.sub(r'\btf\.', 'tensorflow_lazy.tf.', line)
                    modified = True
                
                new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        # 3. 添加函数开头的安全检查
        if modified and 'def ' in content and 'tensorflow_lazy.available' not in content:
            # 寻找使用tf的函数，添加检查
            lines = content.split('\n')
            new_lines = []
            in_function = False
            function_indent = 0
            added_check = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                
                # 检测函数定义
                if line.strip().startswith('def ') and 'tensorflow_lazy.tf.' in content[content.find(line):]:
                    in_function = True
                    function_indent = len(line) - len(line.lstrip())
                    added_check = False
                    
                # 在函数体开始添加检查
                elif in_function and not added_check and line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent > function_indent:
                        # 添加TensorFlow可用性检查
                        check_line = ' ' * current_indent + 'if not tensorflow_lazy.available:'
                        return_line = ' ' * current_indent + '    return'
                        new_lines.insert(-1, check_line)
                        new_lines.insert(-1, return_line)
                        new_lines.insert(-1, '')
                        added_check = True
                        in_function = False
        
        # 如果有修改，写回文件
        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("已修复文件", path=str(file_path))
            return True
        else:
            return False
            
    except Exception:
        logger.exception("修复失败", path=str(file_path))
        return False

def main():
    """主函数"""
    # 需要修复的文件列表（从之前的搜索结果）
    files_to_fix = [
        "ai/reinforcement_learning/performance/integration_test.py",
        "ai/reinforcement_learning/performance/gpu_accelerator.py", 
        "ai/reinforcement_learning/performance/optimized_replay_buffer.py",
        "ai/reinforcement_learning/performance/distributed_training.py",
        "ai/reinforcement_learning/performance/benchmark_optimizer.py",
        "ai/reinforcement_learning/qlearning/double_dqn.py",
        "ai/reinforcement_learning/qlearning/dqn.py", 
        "ai/reinforcement_learning/qlearning/dueling_dqn.py"
    ]
    
    base_path = Path(__file__).parent
    fixed_count = 0
    
    logger.info("开始修复TensorFlow导入")
    
    for file_path in files_to_fix:
        full_path = base_path / file_path
        if full_path.exists():
            logger.info("修复文件", path=file_path)
            if fix_tensorflow_imports(full_path):
                fixed_count += 1
        else:
            logger.warning("文件不存在", path=file_path)
    
    logger.info("修复完成", fixed_count=fixed_count)

if __name__ == "__main__":
    setup_logging()
    main()
