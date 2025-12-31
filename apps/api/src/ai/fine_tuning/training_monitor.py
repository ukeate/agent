"""
训练监控模块
提供训练过程中的指标监控、日志记录、异常检测等功能
"""

import time
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from collections import defaultdict, deque
import threading

from src.core.logging import get_logger
logger = get_logger(__name__)

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir: Optional[str] = None, enable_wandb: bool = False):
        """
        初始化训练监控器
        
        Args:
            log_dir: 日志目录
            enable_wandb: 是否启用Weights & Biases
        """
        self.log_dir = log_dir or "./logs"
        self.enable_wandb = enable_wandb
        self.start_time = time.time()
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = get_logger(__name__)
        
        # 监控数据
        self.metrics = defaultdict(list)
        self.events = []
        self.alerts = []
        
        # 异常检测参数
        self.metric_windows = defaultdict(lambda: deque(maxlen=100))  # 最近100个值
        self.alert_thresholds = {
            'loss_spike': 2.0,      # 损失突增阈值
            'memory_usage': 0.95,    # 内存使用率阈值
            'gradient_norm': 10.0,   # 梯度范数阈值
        }
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 初始化wandb
        if self.enable_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """初始化Weights & Biases"""
        try:
            import wandb
            wandb.init(
                project="lora-fine-tuning",
                dir=self.log_dir,
                tags=["lora", "fine-tuning"]
            )
            self.logger.info("Weights & Biases 初始化成功")
        except ImportError:
            self.logger.warning("Weights & Biases 未安装，跳过初始化")
            self.enable_wandb = False
        except Exception as e:
            self.logger.error(f"Weights & Biases 初始化失败: {e}")
            self.enable_wandb = False
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            step: 步数
        """
        with self._lock:
            timestamp = time.time()
            metric_data = {
                'timestamp': timestamp,
                'value': value,
                'step': step or len(self.metrics[name])
            }
            
            self.metrics[name].append(metric_data)
            
            # 记录到日志
            self.logger.info(f"指标 {name}: {value} (步数: {metric_data['step']})")
            
            # 发送到wandb
            if self.enable_wandb:
                try:
                    import wandb
                    wandb.log({name: value}, step=metric_data['step'])
                except Exception as e:
                    self.logger.error(f"Wandb记录失败: {e}")
            
            # 异常检测
            self._detect_anomaly(name, value)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        记录事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        with self._lock:
            event = {
                'timestamp': time.time(),
                'type': event_type,
                'data': data
            }
            
            self.events.append(event)
            
            # 记录到日志
            self.logger.info(f"事件 {event_type}: {data}")
    
    def log_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """
        记录告警
        
        Args:
            alert_type: 告警类型
            message: 告警消息
            severity: 严重程度 (info/warning/error)
        """
        with self._lock:
            alert = {
                'timestamp': time.time(),
                'type': alert_type,
                'message': message,
                'severity': severity
            }
            
            self.alerts.append(alert)
            
            # 记录到日志
            log_func = getattr(self.logger, severity, self.logger.info)
            log_func(f"告警 [{alert_type}]: {message}")
    
    def _detect_anomaly(self, metric_name: str, value: float):
        """异常检测"""
        self.metric_windows[metric_name].append(value)
        window = self.metric_windows[metric_name]
        
        if len(window) < 10:  # 需要足够的历史数据
            return
        
        # 损失突增检测
        if 'loss' in metric_name.lower():
            recent_avg = sum(list(window)[-5:]) / 5
            historical_avg = sum(list(window)[:-5]) / (len(window) - 5)
            
            if recent_avg > historical_avg * self.alert_thresholds['loss_spike']:
                self.log_alert(
                    'loss_spike',
                    f"{metric_name} 突然增加: {recent_avg:.4f} vs {historical_avg:.4f}",
                    'warning'
                )
        
        # 内存使用率检测
        if 'memory_utilization' in metric_name:
            if value > self.alert_thresholds['memory_usage'] * 100:
                self.log_alert(
                    'high_memory_usage',
                    f"内存使用率过高: {value:.1f}%",
                    'warning'
                )
        
        # 梯度范数检测
        if 'grad_norm' in metric_name:
            if value > self.alert_thresholds['gradient_norm']:
                self.log_alert(
                    'large_gradient',
                    f"梯度范数过大: {value:.4f}",
                    'warning'
                )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            summary = {}
            
            for metric_name, metric_data in self.metrics.items():
                if not metric_data:
                    continue
                
                values = [item['value'] for item in metric_data]
                summary[metric_name] = {
                    'count': len(values),
                    'latest': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'trend': self._calculate_trend(values)
                }
            
            return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 计算最近10个值的趋势
        recent_values = values[-10:] if len(values) >= 10 else values
        
        if len(recent_values) < 2:
            return 'stable'
        
        # 简单线性回归
        n = len(recent_values)
        x = list(range(n))
        y = recent_values
        
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                (n * sum(x[i] * x[i] for i in range(n)) - sum(x) * sum(x))
        
        if abs(slope) < 0.001:  # 阈值可调整
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        with self._lock:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 获取最新指标
            latest_metrics = {}
            for metric_name, metric_data in self.metrics.items():
                if metric_data:
                    latest_metrics[metric_name] = metric_data[-1]['value']
            
            # 计算训练进度
            current_epoch = latest_metrics.get('epoch', 0)
            total_epochs = latest_metrics.get('total_epochs', 1)
            progress_percentage = (current_epoch / total_epochs) * 100 if total_epochs > 0 else 0
            
            return {
                'elapsed_time': elapsed_time,
                'elapsed_time_formatted': self._format_time(elapsed_time),
                'current_epoch': current_epoch,
                'total_epochs': total_epochs,
                'progress_percentage': progress_percentage,
                'latest_metrics': latest_metrics,
                'event_count': len(self.events),
                'alert_count': len(self.alerts)
            }
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def save_report(self, filepath: Optional[str] = None):
        """保存训练报告"""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"training_report_{int(time.time())}.json")
        
        report = {
            'metadata': {
                'generated_at': utc_now().isoformat(),
                'training_duration': time.time() - self.start_time
            },
            'metrics_summary': self.get_metrics_summary(),
            'training_progress': self.get_training_progress(),
            'events': self.events,
            'alerts': self.alerts,
            'raw_metrics': dict(self.metrics)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"训练报告已保存: {filepath}")
        return filepath
    
    def plot_metrics(self, metrics: List[str], save_path: Optional[str] = None):
        """绘制指标图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
        except ImportError:
            self.logger.warning("matplotlib未安装，无法绘制图表")
            return
        
        if not metrics:
            metrics = list(self.metrics.keys())
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metrics):
            if metric_name not in self.metrics:
                continue
            
            data = self.metrics[metric_name]
            if not data:
                continue
            
            timestamps = [datetime.fromtimestamp(item['timestamp']) for item in data]
            values = [item['value'] for item in data]
            
            axes[i].plot(timestamps, values, marker='o', markersize=2)
            axes[i].set_title(f"{metric_name}")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)
            
            # 格式化x轴时间
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            axes[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, f"metrics_plot_{int(time.time())}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"指标图表已保存: {save_path}")
        return save_path
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """获取实时状态"""
        return {
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'metrics_count': len(self.metrics),
            'events_count': len(self.events),
            'alerts_count': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'training_progress': self.get_training_progress()
        }
    
    def set_alert_threshold(self, alert_type: str, threshold: float):
        """设置告警阈值"""
        self.alert_thresholds[alert_type] = threshold
        self.logger.info(f"设置告警阈值 {alert_type}: {threshold}")
    
    def clear_history(self):
        """清除历史数据"""
        with self._lock:
            self.metrics.clear()
            self.events.clear()
            self.alerts.clear()
            self.metric_windows.clear()
            self.start_time = time.time()
        
        self.logger.info("历史数据已清除")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type:
            self.log_alert(
                'training_exception',
                f"训练过程中发生异常: {exc_type.__name__}: {exc_val}",
                'error'
            )
        
        # 保存最终报告
        self.save_report()
        
        # 关闭wandb
        if self.enable_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                self.logger.error(f"Wandb关闭失败: {e}")
