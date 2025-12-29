import os
import sys
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
环境配置脚本
在应用启动前设置必要的环境变量，解决TensorFlow在Apple Silicon上的兼容性问题
"""

def setup_tensorflow_environment():
    """
    设置TensorFlow环境变量
    """
    tensorflow_env_vars = {
        # 减少TensorFlow日志输出
        'TF_CPP_MIN_LOG_LEVEL': '2',
        # 使用传统Keras API，提高兼容性
        'TF_USE_LEGACY_KERAS': '1',
        # 禁用oneDNN优化，避免Apple Silicon兼容性问题
        'TF_ENABLE_ONEDNN_OPTS': '0',
        # 禁用segment reduction操作的确定性异常
        'TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS': '1',
        # 禁用GPU内存预分配
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        # 禁用NUMA节点的GPU设备枚举
        'TF_GPU_THREAD_MODE': 'gpu_private',
        # 使用单线程执行器
        'TF_NUM_INTEROP_THREADS': '1',
        'TF_NUM_INTRAOP_THREADS': '1',
    }
    
    logger.info("设置TensorFlow环境变量")
    for key, value in tensorflow_env_vars.items():
        os.environ[key] = value
        logger.info("环境变量", key=key, value=value)

def setup_general_environment():
    """设置通用Python环境变量 - 完全禁用可能导致mutex lock的所有库"""
    general_env_vars = {
        # 设置Python路径
        'PYTHONPATH': os.getcwd(),
        # 禁用Python字节码生成，减少启动开销
        'PYTHONDONTWRITEBYTECODE': '1',
        # 强制Python使用UTF-8编码
        'PYTHONIOENCODING': 'utf-8',
        # 启用Python开发模式
        'PYTHONDEVMODE': '1' if os.getenv('DEVELOPMENT', '').lower() == 'true' else '0',
        # 禁用tokenizers并行
        'TOKENIZERS_PARALLELISM': 'false',
        # 禁用HuggingFace离线模式
        'HF_DATASETS_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1',
        # 解决KMP重复库问题
        'KMP_DUPLICATE_LIB_OK': 'TRUE',
        # 完全禁用所有数学库的多线程
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'NUMEXPR_MAX_THREADS': '1',
        'OMP_NUM_THREADS': '1',
        # 禁用Intel MKL
        'MKL_THREADING_LAYER': 'sequential',
        'MKL_SERVICE_FORCE_INTEL': '1',
        # 禁用OpenBLAS多线程
        'OPENBLAS_NUM_THREADS': '1',
        'GOTO_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        # 禁用TensorFlow
        'DISABLE_TENSORFLOW': '1',
        'NO_TENSORFLOW': '1',
        # PyTorch设置
        'TORCH_SHOW_CPP_STACKTRACES': '1',
    }
    
    logger.info("设置通用环境变量")
    for key, value in general_env_vars.items():
        if key not in os.environ or key == 'PYTHONPATH':
            os.environ[key] = value
            logger.info("环境变量", key=key, value=value)

def check_system_compatibility():
    """
    检查系统兼容性
    """
    logger.info("检查系统兼容性")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 11):
        logger.warning(
            "Python版本可能不兼容，建议使用Python 3.11+",
            major=python_version.major,
            minor=python_version.minor,
        )
    else:
        logger.info(
            "Python版本",
            major=python_version.major,
            minor=python_version.minor,
            micro=python_version.micro,
        )
    
    # 检查平台
    import platform
    system_info = {
        "系统": platform.system(),
        "架构": platform.machine(),
        "处理器": platform.processor(),
        "Python实现": platform.python_implementation()
    }
    
    for key, value in system_info.items():
        logger.info("系统信息", key=key, value=value)
    
    # Apple Silicon特殊检查
    if platform.machine() in ('arm64', 'aarch64') and platform.system() == 'Darwin':
        logger.info("检测到Apple Silicon，已应用兼容性配置")
        return True
    
    return False

def main():
    """
    主函数
    """
    logger.info("AI Agent API 环境配置")
    logger.info("分隔线", line="=" * 60)
    
    # 检查系统兼容性
    is_apple_silicon = check_system_compatibility()
    
    # 设置环境变量
    setup_general_environment()
    
    # 如果是Apple Silicon，设置TensorFlow特殊配置
    if is_apple_silicon:
        setup_tensorflow_environment()
    
    logger.info("环境配置完成")
    logger.info("分隔线", line="=" * 60)

if __name__ == "__main__":
    setup_logging()
    main()
