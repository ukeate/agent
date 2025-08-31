import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Alert } from '../components/ui/alert';
import { Progress } from '../components/ui/progress';
import { Input } from '../components/ui/input';

interface TrialTask {
  id: string;
  experiment_id: string;
  experiment_name: string;
  trial_number: number;
  parameters: Record<string, any>;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: 'low' | 'normal' | 'high' | 'critical';
  resource_requirements: {
    cpu_cores: number;
    memory_gb: number;
    gpu_count?: number;
    estimated_duration_minutes: number;
  };
  queue_position?: number;
  start_time?: string;
  end_time?: string;
  duration?: number;
  result_value?: number;
  error_message?: string;
  worker_id?: string;
}

interface SchedulerConfig {
  max_concurrent_trials: number;
  cpu_limit_per_trial: number;
  memory_limit_per_trial: number;
  gpu_limit_per_trial: number;
  scheduling_algorithm: 'fifo' | 'priority' | 'fair_share' | 'shortest_job_first';
  timeout_minutes: number;
  retry_count: number;
  auto_scaling_enabled: boolean;
  preemption_enabled: boolean;
}

interface SchedulerStats {
  total_tasks: number;
  queued_tasks: number;
  running_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  average_queue_time: number;
  average_execution_time: number;
  throughput_per_hour: number;
  resource_utilization: number;
}

interface Worker {
  id: string;
  name: string;
  status: 'active' | 'idle' | 'offline' | 'error';
  current_task?: string;
  cpu_usage: number;
  memory_usage: number;
  gpu_usage?: number;
  tasks_completed: number;
  last_heartbeat: string;
}

const HyperparameterSchedulerPage: React.FC = () => {
  const [tasks, setTasks] = useState<TrialTask[]>([]);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [config, setConfig] = useState<SchedulerConfig | null>(null);
  const [stats, setStats] = useState<SchedulerStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'queue' | 'workers' | 'config' | 'stats'>('queue');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const API_BASE = '/api/v1/hyperparameter-optimization';

  // 加载调度器数据
  const loadSchedulerData = async () => {
    try {
      setLoading(true);
      
      // 模拟任务队列数据
      const mockTasks: TrialTask[] = [];
      for (let i = 1; i <= 20; i++) {
        const statuses: TrialTask['status'][] = ['queued', 'running', 'completed', 'failed'];
        const priorities: TrialTask['priority'][] = ['low', 'normal', 'high', 'critical'];
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        
        mockTasks.push({
          id: `task-${i}`,
          experiment_id: `exp-${Math.ceil(i / 3)}`,
          experiment_name: `实验 ${Math.ceil(i / 3)}`,
          trial_number: i,
          parameters: {
            learning_rate: (Math.random() * 0.01 + 0.001).toFixed(4),
            batch_size: Math.floor(Math.random() * 64 + 16),
            hidden_units: Math.floor(Math.random() * 256 + 64)
          },
          status,
          priority: priorities[Math.floor(Math.random() * priorities.length)],
          resource_requirements: {
            cpu_cores: Math.floor(Math.random() * 4 + 1),
            memory_gb: Math.floor(Math.random() * 8 + 2),
            gpu_count: Math.random() > 0.5 ? Math.floor(Math.random() * 2 + 1) : undefined,
            estimated_duration_minutes: Math.floor(Math.random() * 120 + 30)
          },
          queue_position: status === 'queued' ? Math.floor(Math.random() * 10 + 1) : undefined,
          start_time: status !== 'queued' ? new Date(Date.now() - Math.random() * 3600000).toISOString() : undefined,
          end_time: status === 'completed' || status === 'failed' ? new Date(Date.now() - Math.random() * 1800000).toISOString() : undefined,
          duration: status === 'completed' || status === 'failed' ? Math.random() * 3600 + 300 : undefined,
          result_value: status === 'completed' ? Math.random() * 0.95 + 0.05 : undefined,
          error_message: status === 'failed' ? 'Out of memory error' : undefined,
          worker_id: status === 'running' ? `worker-${Math.floor(Math.random() * 4 + 1)}` : undefined
        });
      }

      // 模拟工作节点数据
      const mockWorkers: Worker[] = [];
      for (let i = 1; i <= 5; i++) {
        const statuses: Worker['status'][] = ['active', 'idle', 'offline', 'error'];
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        
        mockWorkers.push({
          id: `worker-${i}`,
          name: `计算节点-${i}`,
          status,
          current_task: status === 'active' ? `task-${Math.floor(Math.random() * 20 + 1)}` : undefined,
          cpu_usage: status === 'offline' ? 0 : Math.random() * 80 + 10,
          memory_usage: status === 'offline' ? 0 : Math.random() * 70 + 20,
          gpu_usage: Math.random() > 0.5 ? (status === 'offline' ? 0 : Math.random() * 90 + 5) : undefined,
          tasks_completed: Math.floor(Math.random() * 100 + 10),
          last_heartbeat: new Date(Date.now() - Math.random() * 60000).toISOString()
        });
      }

      // 模拟调度器配置
      const mockConfig: SchedulerConfig = {
        max_concurrent_trials: 10,
        cpu_limit_per_trial: 4,
        memory_limit_per_trial: 8,
        gpu_limit_per_trial: 2,
        scheduling_algorithm: 'priority',
        timeout_minutes: 120,
        retry_count: 3,
        auto_scaling_enabled: true,
        preemption_enabled: false
      };

      // 模拟统计数据
      const mockStats: SchedulerStats = {
        total_tasks: mockTasks.length,
        queued_tasks: mockTasks.filter(t => t.status === 'queued').length,
        running_tasks: mockTasks.filter(t => t.status === 'running').length,
        completed_tasks: mockTasks.filter(t => t.status === 'completed').length,
        failed_tasks: mockTasks.filter(t => t.status === 'failed').length,
        average_queue_time: Math.random() * 600 + 120,
        average_execution_time: Math.random() * 3600 + 1800,
        throughput_per_hour: Math.random() * 20 + 5,
        resource_utilization: Math.random() * 40 + 60
      };

      setTasks(mockTasks);
      setWorkers(mockWorkers);
      setConfig(mockConfig);
      setStats(mockStats);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSchedulerData();
    // 设置定时刷新
    const interval = setInterval(loadSchedulerData, 5000);
    return () => clearInterval(interval);
  }, []);

  // 取消任务
  const cancelTask = async (taskId: string) => {
    try {
      // 模拟API调用
      setTasks(prev => prev.map(task => 
        task.id === taskId ? { ...task, status: 'cancelled' } : task
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel task');
    }
  };

  // 重试任务
  const retryTask = async (taskId: string) => {
    try {
      // 模拟API调用
      setTasks(prev => prev.map(task => 
        task.id === taskId ? { ...task, status: 'queued', error_message: undefined } : task
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retry task');
    }
  };

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
      case 'active':
        return 'bg-blue-500 text-white';
      case 'completed':
        return 'bg-green-500 text-white';
      case 'failed':
      case 'error':
        return 'bg-red-500 text-white';
      case 'queued':
      case 'idle':
        return 'bg-yellow-500 text-white';
      case 'cancelled':
      case 'offline':
        return 'bg-gray-500 text-white';
      default:
        return 'bg-gray-500 text-white';
    }
  };

  // 获取优先级颜色
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'bg-red-100 text-red-800';
      case 'high':
        return 'bg-orange-100 text-orange-800';
      case 'normal':
        return 'bg-blue-100 text-blue-800';
      case 'low':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // 格式化时间差
  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}时${minutes}分`;
  };

  // 过滤任务
  const filteredTasks = filterStatus === 'all' ? tasks : tasks.filter(task => task.status === filterStatus);

  // 渲染任务队列
  const renderTaskQueue = () => (
    <div className="space-y-4">
      {/* 过滤器 */}
      <div className="flex space-x-4 items-center">
        <span className="text-sm font-medium text-gray-700">状态过滤:</span>
        <select
          className="border border-gray-300 rounded-md px-3 py-1 text-sm"
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
        >
          <option value="all">全部</option>
          <option value="queued">队列中</option>
          <option value="running">运行中</option>
          <option value="completed">已完成</option>
          <option value="failed">失败</option>
          <option value="cancelled">已取消</option>
        </select>
      </div>

      {/* 任务列表 */}
      <div className="space-y-3">
        {filteredTasks.map((task) => (
          <Card key={task.id} className="p-4">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="font-semibold text-gray-900">
                    {task.experiment_name} - 试验 #{task.trial_number}
                  </h3>
                  <Badge className={getStatusColor(task.status)}>
                    {task.status}
                  </Badge>
                  <Badge className={getPriorityColor(task.priority)}>
                    {task.priority}
                  </Badge>
                  {task.queue_position && (
                    <Badge className="bg-purple-100 text-purple-800">
                      队列位置: {task.queue_position}
                    </Badge>
                  )}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                  <div>
                    <span className="text-gray-500">CPU:</span>
                    <span className="ml-1 font-medium">{task.resource_requirements.cpu_cores}核</span>
                  </div>
                  <div>
                    <span className="text-gray-500">内存:</span>
                    <span className="ml-1 font-medium">{task.resource_requirements.memory_gb}GB</span>
                  </div>
                  {task.resource_requirements.gpu_count && (
                    <div>
                      <span className="text-gray-500">GPU:</span>
                      <span className="ml-1 font-medium">{task.resource_requirements.gpu_count}卡</span>
                    </div>
                  )}
                  <div>
                    <span className="text-gray-500">预计时长:</span>
                    <span className="ml-1 font-medium">{task.resource_requirements.estimated_duration_minutes}分钟</span>
                  </div>
                </div>

                {/* 参数显示 */}
                <div className="text-xs text-gray-600 mb-2">
                  参数: {Object.entries(task.parameters).map(([key, value]) => `${key}=${value}`).join(', ')}
                </div>

                {/* 时间和结果信息 */}
                <div className="flex items-center space-x-4 text-xs text-gray-500">
                  {task.start_time && (
                    <span>开始: {new Date(task.start_time).toLocaleString('zh-CN')}</span>
                  )}
                  {task.duration && (
                    <span>耗时: {formatDuration(task.duration)}</span>
                  )}
                  {task.result_value && (
                    <span className="font-medium text-blue-600">结果: {task.result_value.toFixed(4)}</span>
                  )}
                  {task.worker_id && (
                    <span>工作节点: {task.worker_id}</span>
                  )}
                </div>

                {task.error_message && (
                  <div className="text-xs text-red-600 mt-2">
                    错误: {task.error_message}
                  </div>
                )}
              </div>

              {/* 操作按钮 */}
              <div className="flex space-x-2 ml-4">
                {task.status === 'queued' && (
                  <Button size="sm" variant="outline" onClick={() => cancelTask(task.id)}>
                    取消
                  </Button>
                )}
                {task.status === 'failed' && (
                  <Button size="sm" variant="outline" onClick={() => retryTask(task.id)}>
                    重试
                  </Button>
                )}
                <Button size="sm" variant="outline">
                  详情
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );

  // 渲染工作节点
  const renderWorkers = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {workers.map((worker) => (
        <Card key={worker.id} className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">{worker.name}</h3>
              <div className="flex items-center space-x-2 mt-1">
                <Badge className={getStatusColor(worker.status)}>
                  {worker.status}
                </Badge>
                <span className="text-sm text-gray-500">
                  已完成 {worker.tasks_completed} 个任务
                </span>
              </div>
            </div>
            <div className="text-xs text-gray-500">
              最后心跳: {new Date(worker.last_heartbeat).toLocaleString('zh-CN')}
            </div>
          </div>

          {worker.current_task && (
            <div className="mb-4 p-3 bg-blue-50 rounded-lg">
              <div className="text-sm font-medium text-blue-900">
                当前任务: {worker.current_task}
              </div>
            </div>
          )}

          {/* 资源使用情况 */}
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>CPU使用率</span>
                <span>{worker.cpu_usage.toFixed(1)}%</span>
              </div>
              <Progress value={worker.cpu_usage} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>内存使用率</span>
                <span>{worker.memory_usage.toFixed(1)}%</span>
              </div>
              <Progress value={worker.memory_usage} className="h-2" />
            </div>
            {worker.gpu_usage !== undefined && (
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>GPU使用率</span>
                  <span>{worker.gpu_usage.toFixed(1)}%</span>
                </div>
                <Progress value={worker.gpu_usage} className="h-2" />
              </div>
            )}
          </div>
        </Card>
      ))}
    </div>
  );

  // 渲染配置
  const renderConfig = () => (
    <div className="space-y-6">
      {config && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">调度器配置</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  最大并发试验数
                </label>
                <Input
                  type="number"
                  value={config.max_concurrent_trials}
                  onChange={(e) => setConfig({
                    ...config,
                    max_concurrent_trials: parseInt(e.target.value)
                  })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  单个试验CPU限制
                </label>
                <Input
                  type="number"
                  value={config.cpu_limit_per_trial}
                  onChange={(e) => setConfig({
                    ...config,
                    cpu_limit_per_trial: parseInt(e.target.value)
                  })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  单个试验内存限制 (GB)
                </label>
                <Input
                  type="number"
                  value={config.memory_limit_per_trial}
                  onChange={(e) => setConfig({
                    ...config,
                    memory_limit_per_trial: parseInt(e.target.value)
                  })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  单个试验GPU限制
                </label>
                <Input
                  type="number"
                  value={config.gpu_limit_per_trial}
                  onChange={(e) => setConfig({
                    ...config,
                    gpu_limit_per_trial: parseInt(e.target.value)
                  })}
                />
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  调度算法
                </label>
                <select
                  className="w-full p-2 border border-gray-300 rounded-md"
                  value={config.scheduling_algorithm}
                  onChange={(e) => setConfig({
                    ...config,
                    scheduling_algorithm: e.target.value as any
                  })}
                >
                  <option value="fifo">先进先出 (FIFO)</option>
                  <option value="priority">优先级调度</option>
                  <option value="fair_share">公平共享</option>
                  <option value="shortest_job_first">最短作业优先</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  超时时间 (分钟)
                </label>
                <Input
                  type="number"
                  value={config.timeout_minutes}
                  onChange={(e) => setConfig({
                    ...config,
                    timeout_minutes: parseInt(e.target.value)
                  })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  重试次数
                </label>
                <Input
                  type="number"
                  value={config.retry_count}
                  onChange={(e) => setConfig({
                    ...config,
                    retry_count: parseInt(e.target.value)
                  })}
                />
              </div>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.auto_scaling_enabled}
                    onChange={(e) => setConfig({
                      ...config,
                      auto_scaling_enabled: e.target.checked
                    })}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700">启用自动扩缩容</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.preemption_enabled}
                    onChange={(e) => setConfig({
                      ...config,
                      preemption_enabled: e.target.checked
                    })}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700">启用任务抢占</span>
                </label>
              </div>
            </div>
          </div>
          
          <div className="flex justify-end mt-6">
            <Button>保存配置</Button>
          </div>
        </Card>
      )}
    </div>
  );

  // 渲染统计
  const renderStats = () => (
    <div className="space-y-6">
      {stats && (
        <>
          {/* 任务统计 */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <Card className="p-4 text-center">
              <div className="text-2xl font-bold text-gray-900">{stats.total_tasks}</div>
              <div className="text-sm text-gray-500">总任务数</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-2xl font-bold text-yellow-600">{stats.queued_tasks}</div>
              <div className="text-sm text-gray-500">队列中</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-2xl font-bold text-blue-600">{stats.running_tasks}</div>
              <div className="text-sm text-gray-500">运行中</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-2xl font-bold text-green-600">{stats.completed_tasks}</div>
              <div className="text-sm text-gray-500">已完成</div>
            </Card>
            <Card className="p-4 text-center">
              <div className="text-2xl font-bold text-red-600">{stats.failed_tasks}</div>
              <div className="text-sm text-gray-500">失败</div>
            </Card>
          </div>

          {/* 性能指标 */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">平均排队时间</div>
              <div className="text-2xl font-bold text-gray-900">
                {formatDuration(stats.average_queue_time)}
              </div>
            </Card>
            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">平均执行时间</div>
              <div className="text-2xl font-bold text-gray-900">
                {formatDuration(stats.average_execution_time)}
              </div>
            </Card>
            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">每小时吞吐量</div>
              <div className="text-2xl font-bold text-gray-900">
                {stats.throughput_per_hour.toFixed(1)}
              </div>
            </Card>
            <Card className="p-4">
              <div className="text-sm font-medium text-gray-500">资源利用率</div>
              <div className="text-2xl font-bold text-gray-900">
                {stats.resource_utilization.toFixed(1)}%
              </div>
              <Progress value={stats.resource_utilization} className="h-2 mt-2" />
            </Card>
          </div>
        </>
      )}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">试验调度器</h1>
            <p className="mt-2 text-gray-600">
              管理和监控超参数优化试验的调度和执行
            </p>
          </div>
          <Button onClick={loadSchedulerData} disabled={loading}>
            {loading ? '刷新中...' : '刷新'}
          </Button>
        </div>

        {error && (
          <Alert variant="destructive">
            {error}
          </Alert>
        )}

        {/* 标签页导航 */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { key: 'queue', label: '任务队列', count: tasks.length },
              { key: 'workers', label: '工作节点', count: workers.length },
              { key: 'config', label: '调度配置' },
              { key: 'stats', label: '统计信息' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
                {tab.count !== undefined && (
                  <span className="ml-2 bg-gray-100 text-gray-900 py-0.5 px-2 rounded-full text-xs">
                    {tab.count}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* 内容区域 */}
        <div>
          {activeTab === 'queue' && renderTaskQueue()}
          {activeTab === 'workers' && renderWorkers()}
          {activeTab === 'config' && renderConfig()}
          {activeTab === 'stats' && renderStats()}
        </div>
      </div>
    </div>
  );
};

export default HyperparameterSchedulerPage;