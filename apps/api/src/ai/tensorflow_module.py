import os
import warnings
from typing import Optional, Dict, Any, List

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
TensorFlow独立模块
将所有TensorFlow相关功能隔离，避免与主应用mutex lock冲突
"""

os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',  # 抑制TF日志
    'TF_ENABLE_ONEDNN_OPTS': '0',  # 禁用OneDNN优化
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
    'CUDA_VISIBLE_DEVICES': '',  # 禁用GPU
})

# 抑制所有警告
warnings.filterwarnings('ignore')

class TensorFlowService:
    """TensorFlow服务类 - 完全隔离的实现"""
    
    def __init__(self):
        self._tf = None
        self._initialized = False
        self._models = {}
        
    def _lazy_import_tensorflow(self):
        """延迟导入TensorFlow"""
        if self._tf is None:
            try:
                import tensorflow as tf
                # 配置TensorFlow
                tf.config.set_visible_devices([], 'GPU')  # 禁用GPU
                tf.get_logger().setLevel('ERROR')
                self._tf = tf
                logger.info("TensorFlow导入成功")
            except Exception as e:
                logger.error("TensorFlow导入失败", error=str(e), exc_info=True)
                raise ImportError(f"无法导入TensorFlow: {e}")
        return self._tf
    
    def initialize(self) -> bool:
        """初始化TensorFlow环境"""
        try:
            tf = self._lazy_import_tensorflow()
            
            # 设置内存增长
            physical_devices = tf.config.list_physical_devices('GPU')
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # 创建简单测试模型验证环境
            test_model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
                tf.keras.layers.Dense(1)
            ])
            test_model.compile(optimizer='adam', loss='mse')
            
            self._initialized = True
            logger.info("TensorFlow环境初始化成功")
            return True
            
        except Exception as e:
            logger.error("TensorFlow初始化失败", error=str(e), exc_info=True)
            return False
    
    def create_model(self, name: str, model_config: Dict[str, Any]) -> bool:
        """创建TensorFlow模型"""
        if not self._initialized:
            self.initialize()
        
        try:
            tf = self._lazy_import_tensorflow()
            
            # 根据配置创建模型
            if model_config.get('type') == 'sequential':
                model = tf.keras.Sequential()
                for layer_config in model_config.get('layers', []):
                    layer = self._create_layer(layer_config)
                    model.add(layer)
            else:
                raise ValueError(f"不支持的模型类型: {model_config.get('type')}")
            
            # 编译模型
            compile_config = model_config.get('compile', {})
            model.compile(
                optimizer=compile_config.get('optimizer', 'adam'),
                loss=compile_config.get('loss', 'mse'),
                metrics=compile_config.get('metrics', [])
            )
            
            self._models[name] = model
            logger.info("模型创建成功", model_name=name)
            return True
            
        except Exception as e:
            logger.error("创建模型失败", model_name=name, error=str(e), exc_info=True)
            return False
    
    def _create_layer(self, layer_config: Dict[str, Any]):
        """创建神经网络层"""
        tf = self._lazy_import_tensorflow()
        
        layer_type = layer_config.get('type')
        params = layer_config.get('params', {})
        
        if layer_type == 'dense':
            return tf.keras.layers.Dense(**params)
        elif layer_type == 'conv2d':
            return tf.keras.layers.Conv2D(**params)
        elif layer_type == 'lstm':
            return tf.keras.layers.LSTM(**params)
        elif layer_type == 'dropout':
            return tf.keras.layers.Dropout(**params)
        else:
            raise ValueError(f"不支持的层类型: {layer_type}")
    
    def train_model(self, model_name: str, train_data: Any, train_labels: Any, 
                   epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        """训练模型"""
        if model_name not in self._models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        try:
            model = self._models[model_name]
            tf = self._lazy_import_tensorflow()
            
            # 转换数据
            if not isinstance(train_data, tf.Tensor):
                train_data = tf.constant(train_data)
            if not isinstance(train_labels, tf.Tensor):
                train_labels = tf.constant(train_labels)
            
            # 训练模型
            history = model.fit(
                train_data, train_labels,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            result = {
                'success': True,
                'model_name': model_name,
                'epochs': epochs,
                'final_loss': float(history.history['loss'][-1]),
                'history': {k: [float(x) for x in v] for k, v in history.history.items()}
            }
            
            logger.info("模型训练完成", model_name=model_name)
            return result
            
        except Exception as e:
            logger.error("训练模型失败", model_name=model_name, error=str(e), exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def predict(self, model_name: str, input_data: Any) -> Dict[str, Any]:
        """模型预测"""
        if model_name not in self._models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        try:
            model = self._models[model_name]
            tf = self._lazy_import_tensorflow()
            
            # 转换输入数据
            if not isinstance(input_data, tf.Tensor):
                input_data = tf.constant(input_data)
            
            # 执行预测
            predictions = model.predict(input_data, verbose=0)
            
            return {
                'success': True,
                'model_name': model_name,
                'predictions': predictions.tolist(),
                'shape': predictions.shape
            }
            
        except Exception as e:
            logger.error("模型预测失败", model_name=model_name, error=str(e), exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def save_model(self, model_name: str, filepath: str) -> bool:
        """保存模型"""
        if model_name not in self._models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        try:
            model = self._models[model_name]
            model.save(filepath)
            logger.info("模型已保存", model_name=model_name, file_path=str(filepath))
            return True
            
        except Exception as e:
            logger.error("保存模型失败", model_name=model_name, error=str(e), exc_info=True)
            return False
    
    def load_model(self, model_name: str, filepath: str) -> bool:
        """加载模型"""
        try:
            tf = self._lazy_import_tensorflow()
            model = tf.keras.models.load_model(filepath)
            self._models[model_name] = model
            logger.info("模型加载成功", model_name=model_name, file_path=str(filepath))
            return True
            
        except Exception as e:
            logger.error("加载模型失败", model_name=model_name, error=str(e), exc_info=True)
            return False
    
    def list_models(self) -> List[str]:
        """获取所有模型名称"""
        return list(self._models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model_name not in self._models:
            return {'success': False, 'error': f'模型 {model_name} 不存在'}
        
        try:
            model = self._models[model_name]
            return {
                'success': True,
                'model_name': model_name,
                'layers': len(model.layers),
                'parameters': model.count_params(),
                'trainable_parameters': sum([tf.size(w).numpy() for w in model.trainable_weights]),
                'input_shape': model.input_shape if hasattr(model, 'input_shape') else None,
                'output_shape': model.output_shape if hasattr(model, 'output_shape') else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def cleanup(self):
        """清理资源"""
        try:
            if self._tf:
                # 清除会话
                self._tf.keras.backend.clear_session()
            
            self._models.clear()
            self._initialized = False
            logger.info("TensorFlow资源清理完成")
            
        except Exception as e:
            logger.error("TensorFlow资源清理失败", error=str(e), exc_info=True)

# 创建全局TensorFlow服务实例
tensorflow_service = TensorFlowService()

def get_tensorflow_service() -> TensorFlowService:
    """获取TensorFlow服务实例"""
    return tensorflow_service

# 便捷函数
def create_simple_model(name: str, input_dim: int, hidden_layers: List[int], output_dim: int) -> bool:
    """创建简单的全连接模型"""
    layers = []
    
    # 输入层
    layers.append({
        'type': 'dense',
        'params': {'units': hidden_layers[0], 'activation': 'relu', 'input_shape': (input_dim,)}
    })
    
    # 隐藏层
    for units in hidden_layers[1:]:
        layers.append({
            'type': 'dense',
            'params': {'units': units, 'activation': 'relu'}
        })
    
    # 输出层
    layers.append({
        'type': 'dense',
        'params': {'units': output_dim}
    })
    
    model_config = {
        'type': 'sequential',
        'layers': layers,
        'compile': {
            'optimizer': 'adam',
            'loss': 'mse',
            'metrics': ['mae']
        }
    }
    
    return tensorflow_service.create_model(name, model_config)

# 测试函数
def test_tensorflow_module():
    """测试TensorFlow模块功能"""
    try:
        # 初始化
        if not tensorflow_service.initialize():
            return False
        
        # 创建测试模型
        success = create_simple_model('test_model', 10, [64, 32], 1)
        if not success:
            return False
        
        # 生成测试数据
        import numpy as np
        test_data = np.random.random((100, 10))
        test_labels = np.random.random((100, 1))
        
        # 训练模型
        result = tensorflow_service.train_model('test_model', test_data, test_labels, epochs=5)
        if not result['success']:
            return False
        
        # 预测
        pred_result = tensorflow_service.predict('test_model', test_data[:5])
        if not pred_result['success']:
            return False
        
        # 获取模型信息
        info = tensorflow_service.get_model_info('test_model')
        logger.info("模型信息", model_info=info)
        
        return True
        
    except Exception as e:
        logger.error("测试失败", error=str(e), exc_info=True)
        return False
    finally:
        tensorflow_service.cleanup()

if __name__ == "__main__":
    logger.info("TensorFlow模块测试开始")
    success = test_tensorflow_module()
    logger.info("TensorFlow模块测试结束", success=success)
