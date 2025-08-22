"""
GPU加速优化器
实现高性能GPU训练策略，包括混合精度训练、批处理优化和内存管理
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import threading
from dataclasses import dataclass
from enum import Enum


class GPUStrategy(Enum):
    """GPU加速策略类型"""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu" 
    MIXED_PRECISION = "mixed_precision"
    XLA_COMPILATION = "xla_compilation"


@dataclass
class GPUConfig:
    """GPU配置参数"""
    strategy: GPUStrategy = GPUStrategy.SINGLE_GPU
    enable_mixed_precision: bool = True
    enable_xla: bool = True
    memory_growth: bool = True
    memory_limit: Optional[int] = None  # MB
    batch_size_multiplier: int = 2  # GPU批处理倍数
    prefetch_buffer_size: int = 4
    parallel_calls: int = tf.data.AUTOTUNE


class GPUAccelerator:
    """GPU加速器"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.strategy = None
        self.mixed_precision_policy = None
        
        self._setup_gpu_strategy()
        self._setup_mixed_precision()
        self._setup_xla_compilation()
        
        # 性能监控
        self.performance_metrics = {
            "batch_times": [],
            "memory_usage": [],
            "gpu_utilization": [],
            "throughput": []
        }
    
    def _setup_gpu_strategy(self):
        """设置GPU策略"""
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if not physical_devices:
            print("警告: 未检测到GPU设备，将使用CPU")
            self.strategy = tf.distribute.get_strategy()
            return
        
        # 配置GPU内存增长
        if self.config.memory_growth:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except Exception as e:
                    print(f"GPU内存增长配置失败: {e}")
        
        # 设置内存限制
        if self.config.memory_limit:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_limit(
                        device, self.config.memory_limit
                    )
                except Exception as e:
                    print(f"GPU内存限制配置失败: {e}")
        
        # 选择分布策略
        if self.config.strategy == GPUStrategy.MULTI_GPU and len(physical_devices) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
            print(f"使用多GPU策略，设备数量: {len(physical_devices)}")
        else:
            self.strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            print(f"使用单GPU策略: {physical_devices[0].name}")
    
    def _setup_mixed_precision(self):
        """设置混合精度训练"""
        if self.config.enable_mixed_precision:
            try:
                self.mixed_precision_policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(self.mixed_precision_policy)
                print("启用混合精度训练 (float16)")
            except Exception as e:
                print(f"混合精度配置失败: {e}")
                self.mixed_precision_policy = None
    
    def _setup_xla_compilation(self):
        """设置XLA编译优化"""
        if self.config.enable_xla:
            try:
                tf.config.optimizer.set_jit(True)
                print("启用XLA编译优化")
            except Exception as e:
                print(f"XLA编译配置失败: {e}")
    
    def create_optimized_dataset(self, data: List[Any]) -> tf.data.Dataset:
        """创建优化的数据集"""
        dataset = tf.data.Dataset.from_tensor_slices(data)
        
        # 数据预处理优化
        dataset = dataset.batch(
            self.config.batch_size_multiplier,
            drop_remainder=True
        )
        dataset = dataset.prefetch(self.config.prefetch_buffer_size)
        
        # 并行化处理
        dataset = dataset.map(
            self._preprocess_data,
            num_parallel_calls=self.config.parallel_calls
        )
        
        return dataset
    
    def _preprocess_data(self, batch):
        """数据预处理函数"""
        # 数据类型转换
        if self.mixed_precision_policy:
            batch = tf.cast(batch, tf.float16)
        
        return batch
    
    def create_distributed_model(self, model_fn, *args, **kwargs):
        """创建分布式模型"""
        with self.strategy.scope():
            model = model_fn(*args, **kwargs)
            return model
    
    def create_distributed_optimizer(self, optimizer_class, **kwargs):
        """创建分布式优化器"""
        with self.strategy.scope():
            optimizer = optimizer_class(**kwargs)
            
            # 混合精度优化器包装
            if self.mixed_precision_policy:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
            return optimizer
    
    @tf.function(experimental_relax_shapes=True)
    def distributed_train_step(self, model, optimizer, loss_fn, inputs, targets):
        """分布式训练步骤"""
        def step_fn(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)
                
                # 混合精度损失缩放
                if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                    scaled_loss = optimizer.get_scaled_loss(loss)
                else:
                    scaled_loss = loss
            
            # 计算梯度
            gradients = tape.gradient(scaled_loss, model.trainable_variables)
            
            # 混合精度梯度反缩放
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                gradients = optimizer.get_unscaled_gradients(gradients)
            
            # 梯度裁剪
            gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
            
            # 应用梯度
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return loss
        
        return self.strategy.run(step_fn, args=(inputs, targets))
    
    def benchmark_performance(self, model, sample_data, num_iterations: int = 100) -> Dict[str, float]:
        """性能基准测试"""
        print(f"开始GPU性能基准测试 ({num_iterations} 次迭代)")
        
        batch_times = []
        memory_usages = []
        
        # 预热
        for _ in range(10):
            with tf.device('/GPU:0'):
                _ = model(sample_data)
        
        # 性能测试
        for i in range(num_iterations):
            start_time = time.time()
            
            with tf.device('/GPU:0'):
                output = model(sample_data)
            
            # 同步GPU操作
            tf.reduce_sum(output).numpy()
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # 记录内存使用
            try:
                memory_info = tf.config.experimental.get_memory_info('/GPU:0')
                memory_usages.append(memory_info['current'] / 1024 / 1024)  # MB
            except:
                pass
        
        # 计算统计指标
        avg_batch_time = np.mean(batch_times)
        throughput = sample_data.shape[0] / avg_batch_time  # samples/sec
        
        results = {
            "average_batch_time": avg_batch_time,
            "throughput": throughput,
            "memory_usage": np.mean(memory_usages) if memory_usages else 0,
            "gpu_utilization": self._get_gpu_utilization()
        }
        
        print(f"基准测试结果:")
        print(f"  平均批处理时间: {avg_batch_time:.4f}s")
        print(f"  吞吐量: {throughput:.2f} samples/sec") 
        print(f"  内存使用: {results['memory_usage']:.2f} MB")
        print(f"  GPU利用率: {results['gpu_utilization']:.2f}%")
        
        return results
    
    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except:
            return 0.0
    
    def optimize_model_for_inference(self, model):
        """优化模型用于推理"""
        # 转换为TensorRT（如果可用）
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            
            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=None,
                input_saved_model_tags=None,
                input_saved_model_signature_key=None,
                conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                    precision_mode=trt.TrtPrecisionMode.FP16,
                    max_workspace_size_bytes=1 << 25,
                    maximum_cached_engines=100
                )
            )
            
            print("应用TensorRT优化")
            return converter.convert()
            
        except ImportError:
            print("TensorRT不可用，跳过优化")
            return model
    
    def create_performance_monitor(self) -> 'PerformanceMonitor':
        """创建性能监控器"""
        return PerformanceMonitor(self)
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            "cpu_count": tf.config.threading.get_inter_op_parallelism_threads(),
            "gpu_devices": []
        }
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        for device in gpu_devices:
            device_info = {
                "name": device.name,
                "device_type": device.device_type
            }
            
            try:
                # 获取GPU详细信息
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                gpu_id = int(device.name.split(':')[-1])
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                
                device_info.update({
                    "memory_total": nvml.nvmlDeviceGetMemoryInfo(handle).total,
                    "memory_free": nvml.nvmlDeviceGetMemoryInfo(handle).free,
                    "temperature": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                    "power_usage": nvml.nvmlDeviceGetPowerUsage(handle),
                })
            except:
                pass
            
            info["gpu_devices"].append(device_info)
        
        return info


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, accelerator: GPUAccelerator):
        self.accelerator = accelerator
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
    
    def start_monitoring(self, interval: float = 1.0):
        """开始性能监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print("开始GPU性能监控")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("停止GPU性能监控")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append({
                    "timestamp": time.time(),
                    **metrics
                })
                
                # 保持历史记录在合理范围内
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
            except Exception as e:
                print(f"性能监控错误: {e}")
            
            time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """收集性能指标"""
        metrics = {}
        
        try:
            # GPU内存使用
            memory_info = tf.config.experimental.get_memory_info('/GPU:0')
            metrics["memory_used"] = memory_info['current'] / 1024 / 1024  # MB
            metrics["memory_peak"] = memory_info['peak'] / 1024 / 1024  # MB
        except:
            pass
        
        try:
            # GPU利用率
            metrics["gpu_utilization"] = self.accelerator._get_gpu_utilization()
        except:
            pass
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # 最近100个数据点
        
        summary = {}
        for key in ["memory_used", "memory_peak", "gpu_utilization"]:
            values = [m.get(key, 0) for m in recent_metrics if key in m]
            if values:
                summary[key] = {
                    "current": values[-1],
                    "average": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values)
                }
        
        return summary


def create_optimized_gpu_config(model_size: str = "medium") -> GPUConfig:
    """根据模型大小创建优化的GPU配置"""
    configs = {
        "small": GPUConfig(
            enable_mixed_precision=True,
            enable_xla=True,
            batch_size_multiplier=4,
            prefetch_buffer_size=8
        ),
        "medium": GPUConfig(
            enable_mixed_precision=True,
            enable_xla=True,
            batch_size_multiplier=2,
            prefetch_buffer_size=4
        ),
        "large": GPUConfig(
            enable_mixed_precision=True,
            enable_xla=False,  # 大模型可能不适合XLA
            batch_size_multiplier=1,
            prefetch_buffer_size=2,
            memory_limit=8192  # 8GB限制
        )
    }
    
    return configs.get(model_size, configs["medium"])