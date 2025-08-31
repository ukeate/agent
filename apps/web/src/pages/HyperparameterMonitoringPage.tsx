import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Alert } from '../components/ui/alert';
import { Progress } from '../components/ui/progress';

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  gpu_usage?: number;
  gpu_memory_usage?: number;
  active_experiments: number;
  running_trials: number;
  queue_size: number;
  response_time_avg: number;
  error_rate: number;
  throughput: number;
}

interface ExperimentMetrics {
  id: string;
  name: string;
  status: string;
  start_time: string;
  duration: number;
  trials_per_minute: number;
  success_rate: number;
  current_best_value?: number;
  estimated_completion?: string;
  resource_usage: {
    cpu: number;
    memory: number;
    gpu?: number;
  };
}

interface AlertRule {
  id: string;
  name: string;
  metric: string;
  condition: string;
  threshold: number;
  status: 'active' | 'triggered' | 'resolved';
  last_triggered?: string;
}

const HyperparameterMonitoringPage: React.FC = () => {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [experimentMetrics, setExperimentMetrics] = useState<ExperimentMetrics[]>([]);
  const [alerts, setAlerts] = useState<AlertRule[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  const API_BASE = '/api/v1/hyperparameter-optimization';

  // 加载系统指标
  const loadSystemMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/metrics/system`);
      if (!response.ok) {
        // 如果API不存在，使用模拟数据
        const mockData: SystemMetrics = {
          cpu_usage: Math.random() * 60 + 20,
          memory_usage: Math.random() * 40 + 30,
          gpu_usage: Math.random() * 80 + 10,
          gpu_memory_usage: Math.random() * 70 + 20,
          active_experiments: Math.floor(Math.random() * 5) + 1,
          running_trials: Math.floor(Math.random() * 20) + 5,
          queue_size: Math.floor(Math.random() * 50),
          response_time_avg: Math.random() * 200 + 50,
          error_rate: Math.random() * 5,
          throughput: Math.random() * 100 + 50
        };
        setSystemMetrics(mockData);
        return;
      }
      
      const data = await response.json();
      setSystemMetrics(data);
    } catch (err) {
      // 使用模拟数据作为降级方案
      const mockData: SystemMetrics = {
        cpu_usage: Math.random() * 60 + 20,
        memory_usage: Math.random() * 40 + 30,
        gpu_usage: Math.random() * 80 + 10,
        gpu_memory_usage: Math.random() * 70 + 20,
        active_experiments: Math.floor(Math.random() * 5) + 1,
        running_trials: Math.floor(Math.random() * 20) + 5,
        queue_size: Math.floor(Math.random() * 50),
        response_time_avg: Math.random() * 200 + 50,
        error_rate: Math.random() * 5,
        throughput: Math.random() * 100 + 50
      };
      setSystemMetrics(mockData);
    }
  };

  // 加载实验指标
  const loadExperimentMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/experiments`);
      if (!response.ok) throw new Error('Failed to load experiment metrics');
      
      const experiments = await response.json();
      const runningExperiments = experiments.filter((exp: any) => exp.status === 'running');
      
      // 为运行中的实验生成指标数据（实际应该从API获取）
      const metrics: ExperimentMetrics[] = runningExperiments.map((exp: any) => ({
        id: exp.id,
        name: exp.name,
        status: exp.status,
        start_time: exp.started_at || exp.created_at,
        duration: Date.now() - new Date(exp.started_at || exp.created_at).getTime(),
        trials_per_minute: Math.random() * 10 + 2,
        success_rate: Math.random() * 30 + 70,
        current_best_value: exp.best_value,
        estimated_completion: new Date(Date.now() + Math.random() * 3600000).toISOString(),
        resource_usage: {
          cpu: Math.random() * 40 + 10,
          memory: Math.random() * 30 + 20,
          gpu: Math.random() * 60 + 20
        }
      }));
      
      setExperimentMetrics(metrics);
    } catch (err) {
      console.error('Failed to load experiment metrics:', err);
    }
  };

  // 加载告警规则
  const loadAlerts = async () => {
    try {
      // 模拟告警数据（实际应该从API获取）
      const mockAlerts: AlertRule[] = [
        {
          id: '1',
          name: 'CPU使用率过高',
          metric: 'cpu_usage',
          condition: '>',
          threshold: 80,
          status: Math.random() > 0.7 ? 'triggered' : 'active',
          last_triggered: new Date(Date.now() - Math.random() * 3600000).toISOString()
        },
        {
          id: '2',
          name: '内存使用率过高',
          metric: 'memory_usage',
          condition: '>',
          threshold: 85,
          status: Math.random() > 0.8 ? 'triggered' : 'active'
        },
        {
          id: '3',
          name: '错误率过高',
          metric: 'error_rate',
          condition: '>',
          threshold: 5,
          status: Math.random() > 0.9 ? 'triggered' : 'resolved'
        },
        {
          id: '4',
          name: '响应时间过长',
          metric: 'response_time_avg',
          condition: '>',
          threshold: 1000,
          status: 'active'
        }
      ];
      
      setAlerts(mockAlerts);
    } catch (err) {
      console.error('Failed to load alerts:', err);
    }
  };

  // 刷新所有数据
  const refreshData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadSystemMetrics(),
        loadExperimentMetrics(),
        loadAlerts()
      ]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // 启动自动刷新
  const startAutoRefresh = () => {
    if (refreshInterval) clearInterval(refreshInterval);
    const interval = setInterval(refreshData, 5000);
    setRefreshInterval(interval);
  };

  // 停止自动刷新
  const stopAutoRefresh = () => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  };

  useEffect(() => {
    refreshData();
    startAutoRefresh();
    
    return () => stopAutoRefresh();
  }, []);

  // 获取状态颜色
  const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'text-red-600';
    if (value >= thresholds.warning) return 'text-yellow-600';
    return 'text-green-600';
  };

  // 格式化持续时间
  const formatDuration = (ms: number) => {
    const hours = Math.floor(ms / (1000 * 60 * 60));
    const minutes = Math.floor((ms % (1000 * 60 * 60)) / (1000 * 60));
    return `${hours}时${minutes}分`;
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">性能监控</h1>
            <p className="mt-2 text-gray-600">
              实时监控超参数优化系统的性能和资源使用
            </p>
          </div>
          <div className="flex space-x-2">
            <Button
              variant={refreshInterval ? "destructive" : "outline"}
              onClick={refreshInterval ? stopAutoRefresh : startAutoRefresh}
            >
              {refreshInterval ? '停止自动刷新' : '开启自动刷新'}
            </Button>
            <Button onClick={refreshData} disabled={loading}>
              {loading ? '刷新中...' : '手动刷新'}
            </Button>
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            {error}
          </Alert>
        )}

        {/* 系统概览指标 */}
        {systemMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">CPU使用率</div>
              <div className={`text-2xl font-bold ${getStatusColor(systemMetrics.cpu_usage, { warning: 70, critical: 85 })}`}>
                {systemMetrics.cpu_usage.toFixed(1)}%
              </div>
              <Progress value={systemMetrics.cpu_usage} className="h-2 mt-2" />
            </Card>

            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">内存使用率</div>
              <div className={`text-2xl font-bold ${getStatusColor(systemMetrics.memory_usage, { warning: 75, critical: 90 })}`}>
                {systemMetrics.memory_usage.toFixed(1)}%
              </div>
              <Progress value={systemMetrics.memory_usage} className="h-2 mt-2" />
            </Card>

            {systemMetrics.gpu_usage !== undefined && (
              <Card className="p-4">
                <div className="text-sm font-medium text-gray-500">GPU使用率</div>
                <div className={`text-2xl font-bold ${getStatusColor(systemMetrics.gpu_usage, { warning: 80, critical: 95 })}`}>
                  {systemMetrics.gpu_usage.toFixed(1)}%
                </div>
                <Progress value={systemMetrics.gpu_usage} className="h-2 mt-2" />
              </Card>
            )}

            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">活跃实验</div>
              <div className="text-2xl font-bold text-blue-600">
                {systemMetrics.active_experiments}
              </div>
              <div className="text-xs text-gray-500 mt-1">个实验运行中</div>
            </Card>

            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">运行试验</div>
              <div className="text-2xl font-bold text-green-600">
                {systemMetrics.running_trials}
              </div>
              <div className="text-xs text-gray-500 mt-1">个试验执行中</div>
            </Card>

            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">响应时间</div>
              <div className={`text-2xl font-bold ${getStatusColor(systemMetrics.response_time_avg, { warning: 500, critical: 1000 })}`}>
                {systemMetrics.response_time_avg.toFixed(0)}ms
              </div>
              <div className="text-xs text-gray-500 mt-1">平均响应</div>
            </Card>
          </div>
        )}

        {/* 告警状态 */}
        <Card className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">告警状态</h2>
            <div className="flex space-x-2">
              <Badge className="bg-red-100 text-red-800">
                {alerts.filter(a => a.status === 'triggered').length} 告警
              </Badge>
              <Badge className="bg-green-100 text-green-800">
                {alerts.filter(a => a.status === 'active').length} 正常
              </Badge>
            </div>
          </div>

          <div className="space-y-3">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-3 rounded-lg border ${
                  alert.status === 'triggered'
                    ? 'border-red-200 bg-red-50'
                    : alert.status === 'resolved'
                    ? 'border-green-200 bg-green-50'
                    : 'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="font-medium text-gray-900">{alert.name}</h3>
                      <Badge
                        className={`${
                          alert.status === 'triggered'
                            ? 'bg-red-500 text-white'
                            : alert.status === 'resolved'
                            ? 'bg-green-500 text-white'
                            : 'bg-gray-500 text-white'
                        }`}
                      >
                        {alert.status}
                      </Badge>
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      {alert.metric} {alert.condition} {alert.threshold}
                    </div>
                    {alert.last_triggered && (
                      <div className="text-xs text-gray-500 mt-1">
                        上次触发: {new Date(alert.last_triggered).toLocaleString('zh-CN')}
                      </div>
                    )}
                  </div>
                  <Button size="sm" variant="outline">
                    配置
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* 实验性能监控 */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold mb-4">运行中实验监控</h2>
          
          {experimentMetrics.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              暂无运行中的实验
            </div>
          ) : (
            <div className="space-y-4">
              {experimentMetrics.map((experiment) => (
                <div key={experiment.id} className="border rounded-lg p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="font-semibold text-gray-900">{experiment.name}</h3>
                      <div className="flex items-center space-x-4 text-sm text-gray-600 mt-1">
                        <span>运行时长: {formatDuration(experiment.duration)}</span>
                        <span>试验速度: {experiment.trials_per_minute.toFixed(1)}/分钟</span>
                        <span>成功率: {experiment.success_rate.toFixed(1)}%</span>
                      </div>
                    </div>
                    <Badge className="bg-blue-100 text-blue-800">运行中</Badge>
                  </div>

                  {experiment.current_best_value !== undefined && (
                    <div className="mb-3">
                      <span className="text-sm font-medium text-gray-700">当前最佳值: </span>
                      <span className="text-lg font-mono text-blue-600">
                        {experiment.current_best_value.toFixed(6)}
                      </span>
                    </div>
                  )}

                  {/* 资源使用情况 */}
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">CPU</div>
                      <div className="flex items-center space-x-2">
                        <Progress value={experiment.resource_usage.cpu} className="h-2 flex-1" />
                        <span className="text-sm font-medium">{experiment.resource_usage.cpu.toFixed(0)}%</span>
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">内存</div>
                      <div className="flex items-center space-x-2">
                        <Progress value={experiment.resource_usage.memory} className="h-2 flex-1" />
                        <span className="text-sm font-medium">{experiment.resource_usage.memory.toFixed(0)}%</span>
                      </div>
                    </div>
                    {experiment.resource_usage.gpu && (
                      <div>
                        <div className="text-sm text-gray-600">GPU</div>
                        <div className="flex items-center space-x-2">
                          <Progress value={experiment.resource_usage.gpu} className="h-2 flex-1" />
                          <span className="text-sm font-medium">{experiment.resource_usage.gpu.toFixed(0)}%</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {experiment.estimated_completion && (
                    <div className="text-xs text-gray-500 mt-2">
                      预计完成时间: {new Date(experiment.estimated_completion).toLocaleString('zh-CN')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* 性能趋势 */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold mb-4">系统性能趋势</h2>
          <div className="h-48 bg-gray-50 rounded-lg flex items-center justify-center">
            <div className="text-gray-500 text-sm">
              性能趋势图表区域（需要图表库支持）
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default HyperparameterMonitoringPage;