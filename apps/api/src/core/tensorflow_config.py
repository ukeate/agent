"""
TensorFlow配置模块
统一处理TensorFlow初始化和配置，解决在Apple Silicon (M4 Pro) 上的mutex lock问题
"""

import os
import warnings
from typing import Optional

from src.core.logging import get_logger
logger = get_logger(__name__)

def configure_tensorflow_environment():
    """
    配置TensorFlow运行环境，解决Apple Silicon上的mutex lock问题
    """
    # 设置TensorFlow环境变量，必须在import tensorflow之前设置
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
    
    for key, value in tensorflow_env_vars.items():
        os.environ[key] = value

def safe_import_tensorflow():
    """
    安全导入TensorFlow，处理可能的错误
    
    Returns:
        tensorflow模块或None（如果导入失败）
    """
    try:
        # 显式检查禁用标记
        if os.environ.get('DISABLE_TENSORFLOW') == '1' or os.environ.get('NO_TENSORFLOW') == '1':
            return None
        # 配置环境变量
        configure_tensorflow_environment()
        
        # 导入TensorFlow
        import tensorflow as tf
        
        # 进行基本配置以避免mutex lock问题
        try:
            # 禁用GPU内存增长
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    try:
                        tf.config.experimental.set_memory_growth(device, True)
                    except RuntimeError:
                        logger.debug("GPU已初始化，跳过内存增长设置", exc_info=True)
            
            # 设置线程配置
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
        except Exception as config_error:
            warnings.warn(f"TensorFlow配置警告: {config_error}")
        
        return tf
        
    except ImportError as e:
        warnings.warn(f"TensorFlow导入失败: {e}")
        return None
    except Exception as e:
        warnings.warn(f"TensorFlow初始化失败: {e}")
        return None

def is_tensorflow_available() -> bool:
    """
    检查TensorFlow是否可用
    
    Returns:
        bool: TensorFlow是否可用
    """
    tf_module = safe_import_tensorflow()
    return tf_module is not None

def get_tensorflow_info() -> dict:
    """
    获取TensorFlow信息
    
    Returns:
        包含TensorFlow信息的字典
    """
    tf_module = safe_import_tensorflow()
    
    if tf_module is None:
        return {
            "available": False,
            "version": None,
            "gpu_available": False,
            "gpu_count": 0,
            "error": "TensorFlow not available"
        }
    
    try:
        gpu_devices = tf_module.config.list_physical_devices('GPU')
        return {
            "available": True,
            "version": tf_module.__version__,
            "gpu_available": len(gpu_devices) > 0,
            "gpu_count": len(gpu_devices),
            "gpu_devices": [device.name for device in gpu_devices] if gpu_devices else []
        }
    except Exception as e:
        return {
            "available": True,
            "version": tf_module.__version__,
            "gpu_available": False,
            "gpu_count": 0,
            "error": str(e)
        }

class TensorFlowLazyImport:
    """
    TensorFlow延迟导入类
    用于在需要时才导入TensorFlow，避免启动时的mutex lock问题
    """
    
    def __init__(self):
        self._tensorflow = None
        self._import_attempted = False
    
    @property
    def tf(self):
        """获取TensorFlow模块"""
        if not self._import_attempted:
            self._tensorflow = safe_import_tensorflow()
            self._import_attempted = True
        return self._tensorflow
    
    @property
    def available(self) -> bool:
        """检查TensorFlow是否可用"""
        return self.tf is not None
    
    def __getattr__(self, name):
        """代理对TensorFlow模块的属性访问"""
        if self.tf is None:
            raise RuntimeError("TensorFlow不可用，无法访问属性")
        return getattr(self.tf, name)

# 全局延迟导入实例
tensorflow_lazy = TensorFlowLazyImport()
