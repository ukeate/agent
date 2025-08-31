import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Progress } from '../components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/Tabs';
import { 
  GitBranch, Upload, Download, RefreshCw, Activity,
  Clock, CheckCircle, AlertTriangle, Pause, Play,
  ArrowUpDown, Database, Network, Settings, Zap,
  BarChart3, TrendingUp, Timer, Hash, Code
} from 'lucide-react';

// 同步引擎相关的数据结构
interface SyncTask {
  id: string;
  session_id: string;
  direction: 'upload' | 'download' | 'bidirectional';
  priority: number;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'paused' | 'cancelled';
  progress: number;
  total_operations: number;
  completed_operations: number;
  failed_operations: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  retry_count: number;
  checkpoint_data: any;
  estimated_completion: string;
  throughput_ops_per_second: number;
}

interface SyncOperation {
  id: string;
  operation_type: 'PUT' | 'PATCH' | 'DELETE' | 'CLEAR';
  table_name: string;
  object_id: string;
  timestamp: string;
  status: 'pending' | 'synced' | 'failed' | 'conflict';
  data_size_bytes: number;
  retry_count: number;
  conflict_detected: boolean;
}

interface SyncStatistics {
  active_tasks: number;
  queued_tasks: number;
  total_tasks: number;
  status_distribution: Record<string, number>;
  priority_distribution: Record<string, number>;
  total_synced_operations: number;
  total_failed_operations: number;
  total_conflicts_resolved: number;
  last_sync_time?: string;
  sync_efficiency: number;
  average_throughput: number;
  network_usage_mb: number;
}

// 增量同步算法演示
interface DeltaDemo {
  base_state: any;
  local_changes: any;
  remote_changes: any;
  calculated_delta: any;
  merge_result: any;
  conflicts: Array<{
    path: string;
    local_value: any;
    remote_value: any;
    resolution: string;
  }>;
  algorithm_steps: string[];
}

// 优先级队列演示
interface PriorityQueueDemo {
  queue_capacity: number;
  tasks: Array<{
    id: string;
    priority: number;
    weight: number;
    arrival_time: string;
    execution_order: number;
  }>;
  scheduling_algorithm: 'priority_first' | 'weighted_fair' | 'round_robin';
}

const SyncEngineLearningPage: React.FC = () => {
  const [syncTasks, setSyncTasks] = useState<SyncTask[]>([]);
  const [syncOperations, setSyncOperations] = useState<SyncOperation[]>([]);
  const [syncStats, setSyncStats] = useState<SyncStatistics | null>(null);
  const [deltaDemo, setDeltaDemo] = useState<DeltaDemo | null>(null);
  const [priorityQueueDemo, setPriorityQueueDemo] = useState<PriorityQueueDemo | null>(null);
  const [selectedTask, setSelectedTask] = useState<SyncTask | null>(null);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);

  // 同步引擎算法实现 - 教学版本
  const SyncEngineAlgorithms = {
    // 增量同步算法
    calculateDelta: (baseState: any, currentState: any) => {
      const delta: any = {};
      const steps: string[] = [];
      
      steps.push('开始计算增量 (Delta):');
      steps.push(`Base State: ${JSON.stringify(baseState)}`);
      steps.push(`Current State: ${JSON.stringify(currentState)}`);

      // 比较对象字段
      const allKeys = new Set([
        ...Object.keys(baseState || {}),
        ...Object.keys(currentState || {})
      ]);

      for (const key of allKeys) {
        const baseValue = baseState?.[key];
        const currentValue = currentState?.[key];

        if (baseValue === undefined && currentValue !== undefined) {
          // 新增字段
          delta[key] = { operation: 'add', value: currentValue };
          steps.push(`字段 '${key}': 新增 -> ${JSON.stringify(currentValue)}`);
        } else if (baseValue !== undefined && currentValue === undefined) {
          // 删除字段
          delta[key] = { operation: 'delete', old_value: baseValue };
          steps.push(`字段 '${key}': 删除 <- ${JSON.stringify(baseValue)}`);
        } else if (baseValue !== currentValue) {
          // 修改字段
          delta[key] = { operation: 'update', old_value: baseValue, new_value: currentValue };
          steps.push(`字段 '${key}': 修改 ${JSON.stringify(baseValue)} -> ${JSON.stringify(currentValue)}`);
        } else {
          steps.push(`字段 '${key}': 无变化`);
        }
      }

      steps.push(`计算完成，Delta: ${JSON.stringify(delta)}`);
      return { delta, steps };
    },

    // 三路合并算法（用于冲突解决）
    threeWayMerge: (base: any, local: any, remote: any) => {
      const merged: any = { ...base };
      const conflicts: Array<{path: string, local_value: any, remote_value: any, resolution: string}> = [];
      const steps: string[] = [];

      steps.push('开始三路合并:');
      steps.push(`Base: ${JSON.stringify(base)}`);
      steps.push(`Local: ${JSON.stringify(local)}`);
      steps.push(`Remote: ${JSON.stringify(remote)}`);

      const allKeys = new Set([
        ...Object.keys(base || {}),
        ...Object.keys(local || {}),
        ...Object.keys(remote || {})
      ]);

      for (const key of allKeys) {
        const baseValue = base?.[key];
        const localValue = local?.[key];
        const remoteValue = remote?.[key];

        steps.push(`\n处理字段 '${key}':`);
        steps.push(`  Base: ${JSON.stringify(baseValue)}`);
        steps.push(`  Local: ${JSON.stringify(localValue)}`);
        steps.push(`  Remote: ${JSON.stringify(remoteValue)}`);

        if (localValue === remoteValue) {
          // 本地和远程相同
          merged[key] = localValue;
          steps.push(`  -> 本地和远程相同，使用: ${JSON.stringify(localValue)}`);
        } else if (localValue === baseValue) {
          // 本地未变，使用远程
          merged[key] = remoteValue;
          steps.push(`  -> 本地未变，使用远程: ${JSON.stringify(remoteValue)}`);
        } else if (remoteValue === baseValue) {
          // 远程未变，使用本地
          merged[key] = localValue;
          steps.push(`  -> 远程未变，使用本地: ${JSON.stringify(localValue)}`);
        } else {
          // 冲突：两边都有修改
          conflicts.push({
            path: key,
            local_value: localValue,
            remote_value: remoteValue,
            resolution: 'manual_required'
          });
          merged[key] = localValue; // 默认使用本地值
          steps.push(`  -> 冲突检测：两边都有修改，需要手动解决`);
        }
      }

      return { merged, conflicts, steps };
    },

    // 优先级队列调度算法
    scheduleByPriority: (tasks: Array<{id: string, priority: number}>, algorithm: string) => {
      const steps: string[] = [];
      steps.push(`使用 ${algorithm} 算法调度任务:`);

      let scheduledTasks: Array<{id: string, priority: number, weight: number, execution_order: number}> = [];

      switch (algorithm) {
        case 'priority_first':
          // 优先级优先调度
          const sortedByPriority = [...tasks].sort((a, b) => a.priority - b.priority);
          scheduledTasks = sortedByPriority.map((task, index) => ({
            ...task,
            weight: 1.0 / task.priority,
            execution_order: index + 1
          }));
          steps.push('优先级调度：数值越小优先级越高');
          break;

        case 'weighted_fair':
          // 加权公平调度
          const weights = tasks.map(task => 1.0 / task.priority);
          const totalWeight = weights.reduce((sum, w) => sum + w, 0);
          scheduledTasks = tasks.map((task, index) => ({
            ...task,
            weight: weights[index] / totalWeight,
            execution_order: index + 1
          })).sort((a, b) => b.weight - a.weight);
          steps.push('加权公平调度：根据权重分配执行时间');
          break;

        case 'round_robin':
          // 轮询调度
          scheduledTasks = tasks.map((task, index) => ({
            ...task,
            weight: 1.0 / tasks.length,
            execution_order: index + 1
          }));
          steps.push('轮询调度：每个任务获得相等的执行时间');
          break;
      }

      return { scheduledTasks, steps };
    }
  };

  // 生成模拟数据
  const generateMockData = () => {
    const tasks: SyncTask[] = [
      {
        id: 'task-001',
        session_id: 'session-123',
        direction: 'upload',
        priority: 1,
        status: 'in_progress',
        progress: 75.5,
        total_operations: 1200,
        completed_operations: 906,
        failed_operations: 12,
        created_at: '2025-01-17T10:30:00Z',
        started_at: '2025-01-17T10:31:00Z',
        retry_count: 1,
        checkpoint_data: { last_batch: 9, completed_at: '2025-01-17T10:45:00Z' },
        estimated_completion: '2025-01-17T11:15:00Z',
        throughput_ops_per_second: 15.2
      },
      {
        id: 'task-002',
        session_id: 'session-456',
        direction: 'bidirectional',
        priority: 2,
        status: 'pending',
        progress: 0,
        total_operations: 800,
        completed_operations: 0,
        failed_operations: 0,
        created_at: '2025-01-17T10:35:00Z',
        retry_count: 0,
        checkpoint_data: {},
        estimated_completion: '2025-01-17T11:30:00Z',
        throughput_ops_per_second: 0
      },
      {
        id: 'task-003',
        session_id: 'session-789',
        direction: 'download',
        priority: 3,
        status: 'completed',
        progress: 100,
        total_operations: 450,
        completed_operations: 445,
        failed_operations: 5,
        created_at: '2025-01-17T09:15:00Z',
        started_at: '2025-01-17T09:16:00Z',
        completed_at: '2025-01-17T09:45:00Z',
        retry_count: 0,
        checkpoint_data: {},
        estimated_completion: '2025-01-17T09:45:00Z',
        throughput_ops_per_second: 25.8
      }
    ];

    const operations: SyncOperation[] = [
      {
        id: 'op-001',
        operation_type: 'PUT',
        table_name: 'user_data',
        object_id: 'user-123',
        timestamp: '2025-01-17T10:45:30Z',
        status: 'synced',
        data_size_bytes: 1024,
        retry_count: 0,
        conflict_detected: false
      },
      {
        id: 'op-002',
        operation_type: 'PATCH',
        table_name: 'documents',
        object_id: 'doc-456',
        timestamp: '2025-01-17T10:46:15Z',
        status: 'conflict',
        data_size_bytes: 2048,
        retry_count: 2,
        conflict_detected: true
      },
      {
        id: 'op-003',
        operation_type: 'DELETE',
        table_name: 'temp_files',
        object_id: 'temp-789',
        timestamp: '2025-01-17T10:47:00Z',
        status: 'pending',
        data_size_bytes: 512,
        retry_count: 0,
        conflict_detected: false
      }
    ];

    const stats: SyncStatistics = {
      active_tasks: 1,
      queued_tasks: 1,
      total_tasks: 3,
      status_distribution: {
        'in_progress': 1,
        'pending': 1,
        'completed': 1
      },
      priority_distribution: {
        '1': 1,
        '2': 1,
        '3': 1
      },
      total_synced_operations: 2458,
      total_failed_operations: 47,
      total_conflicts_resolved: 23,
      last_sync_time: '2025-01-17T10:47:30Z',
      sync_efficiency: 0.981,
      average_throughput: 18.5,
      network_usage_mb: 125.6
    };

    setSyncTasks(tasks);
    setSyncOperations(operations);
    setSyncStats(stats);
    setSelectedTask(tasks[0]);

    // 生成增量同步演示
    const baseState = { name: 'John', age: 30, city: 'New York' };
    const localChanges = { name: 'John', age: 31, city: 'New York', occupation: 'Engineer' };
    const remoteChanges = { name: 'John Smith', age: 30, city: 'San Francisco' };

    const deltaCalc = SyncEngineAlgorithms.calculateDelta(baseState, localChanges);
    const mergeResult = SyncEngineAlgorithms.threeWayMerge(baseState, localChanges, remoteChanges);

    setDeltaDemo({
      base_state: baseState,
      local_changes: localChanges,
      remote_changes: remoteChanges,
      calculated_delta: deltaCalc.delta,
      merge_result: mergeResult.merged,
      conflicts: mergeResult.conflicts,
      algorithm_steps: [...deltaCalc.steps, '', '=== 三路合并 ===', ...mergeResult.steps]
    });

    // 生成优先级队列演示
    const queueTasks = [
      { id: 'task-A', priority: 1 },
      { id: 'task-B', priority: 3 },
      { id: 'task-C', priority: 2 },
      { id: 'task-D', priority: 1 },
      { id: 'task-E', priority: 4 }
    ];

    const scheduling = SyncEngineAlgorithms.scheduleByPriority(queueTasks, 'priority_first');
    setPriorityQueueDemo({
      queue_capacity: 10,
      tasks: scheduling.scheduledTasks.map(task => ({
        ...task,
        arrival_time: new Date(Date.now() - Math.random() * 60000).toISOString()
      })),
      scheduling_algorithm: 'priority_first'
    });
  };

  const simulateTaskProgress = async (taskId: string) => {
    setIsSimulationRunning(true);
    
    // 模拟任务进度更新
    const updateInterval = setInterval(() => {
      setSyncTasks(prev => prev.map(task => {
        if (task.id === taskId && task.status === 'in_progress') {
          const newProgress = Math.min(100, task.progress + Math.random() * 5);
          const newCompleted = Math.floor((newProgress / 100) * task.total_operations);
          
          return {
            ...task,
            progress: newProgress,
            completed_operations: newCompleted,
            throughput_ops_per_second: 12 + Math.random() * 8
          };
        }
        return task;
      }));
    }, 1000);

    // 停止模拟
    setTimeout(() => {
      clearInterval(updateInterval);
      setIsSimulationRunning(false);
      
      setSyncTasks(prev => prev.map(task => {
        if (task.id === taskId) {
          return {
            ...task,
            status: 'completed',
            progress: 100,
            completed_operations: task.total_operations,
            completed_at: new Date().toISOString()
          };
        }
        return task;
      }));
    }, 10000);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Clock className="h-4 w-4" />;
      case 'in_progress': return <Activity className="h-4 w-4 animate-spin" />;
      case 'completed': return <CheckCircle className="h-4 w-4" />;
      case 'failed': return <AlertTriangle className="h-4 w-4" />;
      case 'paused': return <Pause className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-500';
      case 'in_progress': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      case 'paused': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'upload': return <Upload className="h-4 w-4" />;
      case 'download': return <Download className="h-4 w-4" />;
      case 'bidirectional': return <ArrowUpDown className="h-4 w-4" />;
      default: return <RefreshCw className="h-4 w-4" />;
    }
  };

  const getPriorityColor = (priority: number) => {
    switch (priority) {
      case 1: return 'bg-red-500'; // Critical
      case 2: return 'bg-orange-500'; // High
      case 3: return 'bg-blue-500'; // Normal
      case 4: return 'bg-green-500'; // Low
      case 5: return 'bg-gray-500'; // Background
      default: return 'bg-gray-500';
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  useEffect(() => {
    generateMockData();
  }, []);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">同步引擎学习系统</h1>
        <div className="flex space-x-2">
          <Button onClick={() => window.location.reload()} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            重置演示
          </Button>
        </div>
      </div>

      {/* 同步统计概览 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">活跃任务</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{syncStats?.active_tasks || 0}</div>
            <p className="text-xs text-muted-foreground">
              队列中: {syncStats?.queued_tasks || 0}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">同步效率</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{((syncStats?.sync_efficiency || 0) * 100).toFixed(1)}%</div>
            <Progress value={(syncStats?.sync_efficiency || 0) * 100} className="w-full mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">平均吞吐量</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{syncStats?.average_throughput.toFixed(1)} ops/s</div>
            <p className="text-xs text-muted-foreground">
              网络使用: {syncStats?.network_usage_mb.toFixed(1)} MB
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">冲突解决</CardTitle>
            <GitBranch className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{syncStats?.total_conflicts_resolved || 0}</div>
            <p className="text-xs text-muted-foreground">
              失败操作: {syncStats?.total_failed_operations || 0}
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="tasks" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="tasks">任务管理</TabsTrigger>
          <TabsTrigger value="delta">增量同步</TabsTrigger>
          <TabsTrigger value="priority">优先级调度</TabsTrigger>
          <TabsTrigger value="architecture">架构设计</TabsTrigger>
          <TabsTrigger value="implementation">实现细节</TabsTrigger>
        </TabsList>

        <TabsContent value="tasks" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 任务列表 */}
            <div className="space-y-4">
              <h3 className="font-medium">同步任务列表:</h3>
              {syncTasks.map((task) => (
                <Card 
                  key={task.id}
                  className={`cursor-pointer transition-all ${
                    selectedTask?.id === task.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                  }`}
                  onClick={() => setSelectedTask(task)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {getDirectionIcon(task.direction)}
                        <span className="font-medium text-sm">{task.id}</span>
                        <Badge className={getPriorityColor(task.priority)}>
                          P{task.priority}
                        </Badge>
                      </div>
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(task.status)}
                        <Badge className={getStatusColor(task.status)}>
                          {task.status}
                        </Badge>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span>进度: {task.progress.toFixed(1)}%</span>
                        <span>{task.completed_operations}/{task.total_operations}</span>
                      </div>
                      <Progress value={task.progress} className="w-full" />
                      
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>吞吐量: {task.throughput_ops_per_second.toFixed(1)} ops/s</span>
                        <span>失败: {task.failed_operations}</span>
                      </div>
                    </div>

                    {task.status === 'in_progress' && (
                      <Button 
                        size="sm" 
                        className="w-full mt-2"
                        onClick={(e) => {
                          e.stopPropagation();
                          simulateTaskProgress(task.id);
                        }}
                        disabled={isSimulationRunning}
                      >
                        {isSimulationRunning ? '模拟中...' : '模拟进度'}
                      </Button>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* 任务详情 */}
            {selectedTask && (
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      {getDirectionIcon(selectedTask.direction)}
                      <span className="ml-2">{selectedTask.id}</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="font-medium">会话ID:</p>
                        <p className="text-muted-foreground font-mono">{selectedTask.session_id}</p>
                      </div>
                      <div>
                        <p className="font-medium">同步方向:</p>
                        <p className="text-muted-foreground">{selectedTask.direction}</p>
                      </div>
                      <div>
                        <p className="font-medium">优先级:</p>
                        <Badge className={getPriorityColor(selectedTask.priority)}>
                          Priority {selectedTask.priority}
                        </Badge>
                      </div>
                      <div>
                        <p className="font-medium">重试次数:</p>
                        <p className="text-muted-foreground">{selectedTask.retry_count}</p>
                      </div>
                    </div>

                    <div>
                      <p className="font-medium text-sm mb-2">执行统计:</p>
                      <div className="bg-gray-50 rounded p-3 space-y-2">
                        <div className="flex justify-between text-xs">
                          <span>总操作数:</span>
                          <span>{selectedTask.total_operations}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span>已完成:</span>
                          <span className="text-green-600">{selectedTask.completed_operations}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span>失败:</span>
                          <span className="text-red-600">{selectedTask.failed_operations}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span>成功率:</span>
                          <span className="text-blue-600">
                            {selectedTask.total_operations > 0 ? 
                              ((selectedTask.completed_operations / selectedTask.total_operations) * 100).toFixed(1) : 0}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <p className="font-medium text-sm mb-2">时间信息:</p>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span>创建时间:</span>
                          <span>{new Date(selectedTask.created_at).toLocaleString()}</span>
                        </div>
                        {selectedTask.started_at && (
                          <div className="flex justify-between">
                            <span>开始时间:</span>
                            <span>{new Date(selectedTask.started_at).toLocaleString()}</span>
                          </div>
                        )}
                        {selectedTask.completed_at && (
                          <div className="flex justify-between">
                            <span>完成时间:</span>
                            <span>{new Date(selectedTask.completed_at).toLocaleString()}</span>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <span>预计完成:</span>
                          <span>{new Date(selectedTask.estimated_completion).toLocaleString()}</span>
                        </div>
                      </div>
                    </div>

                    {Object.keys(selectedTask.checkpoint_data).length > 0 && (
                      <div>
                        <p className="font-medium text-sm mb-2">检查点数据:</p>
                        <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto">
                          {JSON.stringify(selectedTask.checkpoint_data, null, 2)}
                        </pre>
                      </div>
                    )}

                    {selectedTask.error_message && (
                      <div className="bg-red-50 border border-red-200 rounded p-3">
                        <p className="font-medium text-red-800 text-sm">错误信息:</p>
                        <p className="text-red-600 text-xs">{selectedTask.error_message}</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </div>

          {/* 操作列表 */}
          <Card>
            <CardHeader>
              <CardTitle>同步操作详情</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">操作ID</th>
                      <th className="text-left p-2">类型</th>
                      <th className="text-left p-2">表名</th>
                      <th className="text-left p-2">对象ID</th>
                      <th className="text-left p-2">状态</th>
                      <th className="text-left p-2">数据大小</th>
                      <th className="text-left p-2">重试</th>
                      <th className="text-left p-2">冲突</th>
                    </tr>
                  </thead>
                  <tbody>
                    {syncOperations.map((op) => (
                      <tr key={op.id} className="border-b text-sm">
                        <td className="p-2 font-mono">{op.id}</td>
                        <td className="p-2">
                          <Badge variant="outline">{op.operation_type}</Badge>
                        </td>
                        <td className="p-2">{op.table_name}</td>
                        <td className="p-2 font-mono">{op.object_id}</td>
                        <td className="p-2">
                          <Badge className={getStatusColor(op.status)}>
                            {op.status}
                          </Badge>
                        </td>
                        <td className="p-2">{formatBytes(op.data_size_bytes)}</td>
                        <td className="p-2">{op.retry_count}</td>
                        <td className="p-2">
                          {op.conflict_detected ? (
                            <AlertTriangle className="h-4 w-4 text-red-500" />
                          ) : (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="delta" className="space-y-4">
          {deltaDemo && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Hash className="h-5 w-5 mr-2" />
                  增量同步算法演示
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
                  <div>
                    <h4 className="font-medium mb-2">基础状态 (Base)</h4>
                    <div className="bg-gray-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(deltaDemo.base_state, null, 2)}
                      </pre>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">本地更改 (Local)</h4>
                    <div className="bg-blue-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(deltaDemo.local_changes, null, 2)}
                      </pre>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">远程更改 (Remote)</h4>
                    <div className="bg-green-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(deltaDemo.remote_changes, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <div>
                    <h4 className="font-medium mb-2">计算的增量 (Delta)</h4>
                    <div className="bg-yellow-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(deltaDemo.calculated_delta, null, 2)}
                      </pre>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">合并结果</h4>
                    <div className="bg-purple-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(deltaDemo.merge_result, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>

                {deltaDemo.conflicts.length > 0 && (
                  <div className="mb-6">
                    <h4 className="font-medium mb-2">检测到的冲突:</h4>
                    <div className="space-y-2">
                      {deltaDemo.conflicts.map((conflict, index) => (
                        <div key={index} className="bg-red-50 border border-red-200 rounded p-3">
                          <div className="flex items-center space-x-2 mb-2">
                            <AlertTriangle className="h-4 w-4 text-red-500" />
                            <span className="font-medium text-red-800">字段冲突: {conflict.path}</span>
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                              <span className="font-medium">本地值:</span>
                              <span className="ml-2">{JSON.stringify(conflict.local_value)}</span>
                            </div>
                            <div>
                              <span className="font-medium">远程值:</span>
                              <span className="ml-2">{JSON.stringify(conflict.remote_value)}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <h4 className="font-medium mb-2">算法执行步骤:</h4>
                  <div className="bg-gray-800 text-green-400 p-4 rounded text-xs overflow-x-auto max-h-60">
                    {deltaDemo.algorithm_steps.map((step, index) => (
                      <div key={index}>{step}</div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="priority" className="space-y-4">
          {priorityQueueDemo && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="h-5 w-5 mr-2" />
                  优先级调度算法演示
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="bg-blue-50 border border-blue-200 rounded p-4">
                    <h4 className="font-medium text-blue-800 mb-2">调度算法: {priorityQueueDemo.scheduling_algorithm}</h4>
                    <p className="text-sm text-blue-600">
                      队列容量: {priorityQueueDemo.queue_capacity} | 
                      当前任务数: {priorityQueueDemo.tasks.length}
                    </p>
                  </div>

                  <div>
                    <h4 className="font-medium mb-3">任务调度顺序:</h4>
                    <div className="space-y-2">
                      {priorityQueueDemo.tasks.map((task, index) => (
                        <div key={index} className="flex items-center justify-between border rounded p-3">
                          <div className="flex items-center space-x-4">
                            <Badge variant="outline">#{task.execution_order}</Badge>
                            <span className="font-medium">{task.id}</span>
                            <Badge className={getPriorityColor(task.priority)}>
                              P{task.priority}
                            </Badge>
                          </div>
                          <div className="flex items-center space-x-4 text-sm">
                            <span>权重: {task.weight.toFixed(3)}</span>
                            <span className="text-muted-foreground">
                              到达: {new Date(task.arrival_time).toLocaleTimeString()}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-green-50 border border-green-200 rounded p-4">
                      <h5 className="font-medium text-green-800 mb-2">优先级优先</h5>
                      <ul className="text-sm text-green-600 space-y-1">
                        <li>• 数值越小优先级越高</li>
                        <li>• 关键任务优先执行</li>
                        <li>• 可能导致低优先级任务饥饿</li>
                      </ul>
                    </div>

                    <div className="bg-orange-50 border border-orange-200 rounded p-4">
                      <h5 className="font-medium text-orange-800 mb-2">加权公平调度</h5>
                      <ul className="text-sm text-orange-600 space-y-1">
                        <li>• 根据权重分配执行时间</li>
                        <li>• 平衡优先级和公平性</li>
                        <li>• 避免任务饥饿问题</li>
                      </ul>
                    </div>

                    <div className="bg-purple-50 border border-purple-200 rounded p-4">
                      <h5 className="font-medium text-purple-800 mb-2">轮询调度</h5>
                      <ul className="text-sm text-purple-600 space-y-1">
                        <li>• 所有任务平等对待</li>
                        <li>• 简单公平的分配策略</li>
                        <li>• 忽略任务重要性差异</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="architecture" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Network className="h-5 w-5 mr-2" />
                同步引擎架构设计
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-blue-50 border border-blue-200 rounded p-4">
                    <h4 className="font-medium text-blue-800 mb-3">核心组件</h4>
                    <ul className="text-sm text-blue-600 space-y-2">
                      <li><strong>同步引擎 (SyncEngine)</strong>: 任务调度和执行</li>
                      <li><strong>向量时钟管理器</strong>: 因果关系跟踪</li>
                      <li><strong>增量计算器</strong>: Delta计算和应用</li>
                      <li><strong>冲突检测器</strong>: 并发冲突识别</li>
                      <li><strong>冲突解决器</strong>: 自动和手动冲突解决</li>
                      <li><strong>网络监控器</strong>: 连接质量评估</li>
                    </ul>
                  </div>

                  <div className="bg-green-50 border border-green-200 rounded p4">
                    <h4 className="font-medium text-green-800 mb-3">数据流设计</h4>
                    <ul className="text-sm text-green-600 space-y-2">
                      <li><strong>操作记录</strong>: 本地操作日志</li>
                      <li><strong>状态快照</strong>: 检查点和恢复</li>
                      <li><strong>同步批次</strong>: 批量操作处理</li>
                      <li><strong>冲突记录</strong>: 冲突历史追踪</li>
                      <li><strong>元数据管理</strong>: 版本和校验信息</li>
                      <li><strong>性能度量</strong>: 吞吐量和延迟统计</li>
                    </ul>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-3">系统架构图:</h4>
                  <div className="bg-gray-50 border rounded p-6">
                    <div className="text-center space-y-4">
                      <div className="flex justify-center space-x-4">
                        <div className="bg-blue-100 border-2 border-blue-300 rounded px-4 py-2">
                          <Database className="h-6 w-6 mx-auto mb-1" />
                          <span className="text-xs">本地存储</span>
                        </div>
                        <div className="flex flex-col justify-center">
                          <ArrowUpDown className="h-6 w-6" />
                        </div>
                        <div className="bg-green-100 border-2 border-green-300 rounded px-4 py-2">
                          <Settings className="h-6 w-6 mx-auto mb-1" />
                          <span className="text-xs">同步引擎</span>
                        </div>
                        <div className="flex flex-col justify-center">
                          <ArrowUpDown className="h-6 w-6" />
                        </div>
                        <div className="bg-purple-100 border-2 border-purple-300 rounded px-4 py-2">
                          <Network className="h-6 w-6 mx-auto mb-1" />
                          <span className="text-xs">远程服务器</span>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 text-xs">
                        <div>
                          <p className="font-medium">本地层</p>
                          <p>操作日志、状态管理、离线存储</p>
                        </div>
                        <div>
                          <p className="font-medium">同步层</p>
                          <p>任务调度、冲突解决、断点续传</p>
                        </div>
                        <div>
                          <p className="font-medium">网络层</p>
                          <p>传输优化、重试机制、负载均衡</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-orange-50 border border-orange-200 rounded p-4">
                  <h4 className="font-medium text-orange-800 mb-2">设计原则</h4>
                  <ul className="text-sm text-orange-600 space-y-1">
                    <li>• <strong>最终一致性</strong>: 保证数据最终收敛到一致状态</li>
                    <li>• <strong>分区容错</strong>: 网络分区时仍能正常工作</li>
                    <li>• <strong>可恢复性</strong>: 支持断点续传和故障恢复</li>
                    <li>• <strong>可观测性</strong>: 详细的监控和调试信息</li>
                    <li>• <strong>可扩展性</strong>: 支持多种同步策略和冲突解决方案</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="implementation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Code className="h-5 w-5 mr-2" />
                核心算法实现
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h4 className="font-medium mb-2">同步任务调度器:</h4>
                  <pre className="bg-gray-800 text-green-400 p-4 rounded text-sm overflow-x-auto">
{`class SyncEngine:
    def __init__(self):
        self.active_tasks = {}
        self.task_queue = []
        self.priority_weights = {
            SyncPriority.CRITICAL: 1.0,
            SyncPriority.HIGH: 0.8,
            SyncPriority.NORMAL: 0.6,
            SyncPriority.LOW: 0.4,
            SyncPriority.BACKGROUND: 0.2
        }
    
    async def execute_sync_task(self, task_id: str) -> SyncResult:
        task = self.get_task(task_id)
        task.status = SyncStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        try:
            if task.direction == SyncDirection.UPLOAD:
                result = await self._execute_upload_task(task)
            elif task.direction == SyncDirection.DOWNLOAD:
                result = await self._execute_download_task(task)
            else:
                result = await self._execute_bidirectional_task(task)
            
            task.status = SyncStatus.COMPLETED
            return result
            
        except Exception as e:
            task.status = SyncStatus.FAILED
            if task.retry_count < task.max_retries:
                await self._schedule_retry(task)
            return self._create_error_result(task, e)`}
                  </pre>
                </div>

                <div>
                  <h4 className="font-medium mb-2">增量计算算法:</h4>
                  <pre className="bg-gray-800 text-green-400 p-4 rounded text-sm overflow-x-auto">
{`class DeltaCalculator:
    def calculate_delta(self, base_state: Dict, current_state: Dict) -> Dict:
        delta = {}
        
        # 获取所有字段
        all_keys = set(base_state.keys()) | set(current_state.keys())
        
        for key in all_keys:
            base_value = base_state.get(key)
            current_value = current_state.get(key)
            
            if base_value is None and current_value is not None:
                delta[key] = {'op': 'add', 'value': current_value}
            elif base_value is not None and current_value is None:
                delta[key] = {'op': 'delete', 'old_value': base_value}
            elif base_value != current_value:
                delta[key] = {
                    'op': 'update',
                    'old_value': base_value,
                    'new_value': current_value
                }
        
        return delta
    
    def apply_delta(self, base_state: Dict, delta: Dict) -> Dict:
        new_state = base_state.copy()
        
        for key, operation in delta.items():
            if operation['op'] == 'add':
                new_state[key] = operation['value']
            elif operation['op'] == 'delete':
                new_state.pop(key, None)
            elif operation['op'] == 'update':
                new_state[key] = operation['new_value']
        
        return new_state`}
                  </pre>
                </div>

                <div>
                  <h4 className="font-medium mb-2">断点续传机制:</h4>
                  <pre className="bg-gray-800 text-green-400 p-4 rounded text-sm overflow-x-auto">
{`class CheckpointManager:
    def __init__(self, checkpoint_interval: int = 50):
        self.checkpoint_interval = checkpoint_interval
    
    def create_checkpoint(self, task: SyncTask, batch_index: int):
        """创建检查点"""
        if batch_index % self.checkpoint_interval == 0:
            checkpoint_data = {
                'completed_batches': batch_index,
                'completed_operations': task.completed_operations,
                'failed_operations': task.failed_operations,
                'last_successful_operation': task.get_last_successful_operation(),
                'checkpoint_time': datetime.utcnow().isoformat(),
                'partial_batch_state': task.get_current_batch_state()
            }
            
            task.checkpoint_data = checkpoint_data
            self._persist_checkpoint(task.id, checkpoint_data)
    
    def restore_from_checkpoint(self, task: SyncTask) -> bool:
        """从检查点恢复"""
        if not task.checkpoint_data:
            return False
        
        # 恢复任务状态
        task.completed_operations = task.checkpoint_data['completed_operations']
        task.failed_operations = task.checkpoint_data['failed_operations']
        
        # 计算剩余操作
        completed_batches = task.checkpoint_data['completed_batches']
        task.operation_ids = task.operation_ids[completed_batches * self.batch_size:]
        
        return True`}
                  </pre>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
                  <h4 className="font-medium text-yellow-800 mb-2">性能优化技巧:</h4>
                  <ul className="text-sm text-yellow-600 space-y-1">
                    <li>• <strong>批量处理</strong>: 减少网络往返次数，提高吞吐量</li>
                    <li>• <strong>并发控制</strong>: 限制同时进行的同步任务数量</li>
                    <li>• <strong>增量同步</strong>: 只同步变化的数据，减少传输量</li>
                    <li>• <strong>压缩传输</strong>: 对大数据块进行压缩，节省带宽</li>
                    <li>• <strong>优先级调度</strong>: 重要数据优先同步，提升用户体验</li>
                    <li>• <strong>自适应重试</strong>: 根据网络状况调整重试策略</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SyncEngineLearningPage;