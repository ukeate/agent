import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Alert } from '../components/ui/alert';
import { Progress } from '../components/ui/progress';
import { Input } from '../components/ui/input';

interface ResourcePool {
  id: string;
  name: string;
  type: 'cpu' | 'gpu';
  total_capacity: number;
  used_capacity: number;
  available_capacity: number;
  status: 'healthy' | 'degraded' | 'critical';
  nodes: ResourceNode[];
}

interface ResourceNode {
  id: string;
  name: string;
  type: 'cpu' | 'gpu';
  status: 'online' | 'offline' | 'maintenance';
  cpu_cores: number;
  cpu_usage: number;
  memory_total: number;
  memory_used: number;
  gpu_count?: number;
  gpu_memory_total?: number;
  gpu_memory_used?: number;
  current_experiments: string[];
  temperature?: number;
  power_usage?: number;
}

interface ResourceAllocation {
  experiment_id: string;
  experiment_name: string;
  resource_type: 'cpu' | 'gpu';
  allocated_cores: number;
  allocated_memory: number;
  allocated_gpu?: number;
  priority: 'low' | 'normal' | 'high';
  start_time: string;
  estimated_end_time?: string;
  status: 'running' | 'queued' | 'completed';
}

interface ResourceQuota {
  user_id: string;
  user_name: string;
  cpu_quota: number;
  cpu_used: number;
  memory_quota: number;
  memory_used: number;
  gpu_quota: number;
  gpu_used: number;
  active_experiments: number;
  max_experiments: number;
}

const HyperparameterResourcesPage: React.FC = () => {
  const [resourcePools, setResourcePools] = useState<ResourcePool[]>([]);
  const [allocations, setAllocations] = useState<ResourceAllocation[]>([]);
  const [quotas, setQuotas] = useState<ResourceQuota[]>([]);
  const [selectedPool, setSelectedPool] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'pools' | 'allocations' | 'quotas'>('pools');

  // 模拟API调用
  const loadResourceData = async () => {
    try {
      setLoading(true);
      
      // 模拟资源池数据
      const mockPools: ResourcePool[] = [
        {
          id: 'cpu-pool-1',
          name: 'CPU集群',
          type: 'cpu',
          total_capacity: 128,
          used_capacity: 64,
          available_capacity: 64,
          status: 'healthy',
          nodes: [
            {
              id: 'cpu-node-1',
              name: 'worker-01',
              type: 'cpu',
              status: 'online',
              cpu_cores: 32,
              cpu_usage: 75,
              memory_total: 128,
              memory_used: 64,
              current_experiments: ['exp-1', 'exp-3']
            },
            {
              id: 'cpu-node-2',
              name: 'worker-02',
              type: 'cpu',
              status: 'online',
              cpu_cores: 32,
              cpu_usage: 45,
              memory_total: 128,
              memory_used: 48,
              current_experiments: ['exp-2']
            },
            {
              id: 'cpu-node-3',
              name: 'worker-03',
              type: 'cpu',
              status: 'maintenance',
              cpu_cores: 32,
              cpu_usage: 0,
              memory_total: 128,
              memory_used: 0,
              current_experiments: []
            }
          ]
        },
        {
          id: 'gpu-pool-1',
          name: 'GPU集群',
          type: 'gpu',
          total_capacity: 8,
          used_capacity: 6,
          available_capacity: 2,
          status: 'degraded',
          nodes: [
            {
              id: 'gpu-node-1',
              name: 'gpu-01',
              type: 'gpu',
              status: 'online',
              cpu_cores: 16,
              cpu_usage: 60,
              memory_total: 64,
              memory_used: 32,
              gpu_count: 4,
              gpu_memory_total: 40,
              gpu_memory_used: 32,
              current_experiments: ['exp-4', 'exp-5'],
              temperature: 72,
              power_usage: 280
            },
            {
              id: 'gpu-node-2',
              name: 'gpu-02',
              type: 'gpu',
              status: 'online',
              cpu_cores: 16,
              cpu_usage: 80,
              memory_total: 64,
              memory_used: 45,
              gpu_count: 4,
              gpu_memory_total: 40,
              gpu_memory_used: 38,
              current_experiments: ['exp-6'],
              temperature: 78,
              power_usage: 320
            }
          ]
        }
      ];

      // 模拟资源分配数据
      const mockAllocations: ResourceAllocation[] = [
        {
          experiment_id: 'exp-1',
          experiment_name: '神经网络超参数优化',
          resource_type: 'cpu',
          allocated_cores: 16,
          allocated_memory: 32,
          priority: 'high',
          start_time: new Date(Date.now() - 3600000).toISOString(),
          estimated_end_time: new Date(Date.now() + 1800000).toISOString(),
          status: 'running'
        },
        {
          experiment_id: 'exp-4',
          experiment_name: '深度学习模型调优',
          resource_type: 'gpu',
          allocated_cores: 8,
          allocated_memory: 16,
          allocated_gpu: 2,
          priority: 'normal',
          start_time: new Date(Date.now() - 7200000).toISOString(),
          estimated_end_time: new Date(Date.now() + 3600000).toISOString(),
          status: 'running'
        },
        {
          experiment_id: 'exp-7',
          experiment_name: '强化学习参数搜索',
          resource_type: 'gpu',
          allocated_cores: 4,
          allocated_memory: 8,
          allocated_gpu: 1,
          priority: 'low',
          start_time: new Date(Date.now() + 1800000).toISOString(),
          status: 'queued'
        }
      ];

      // 模拟资源配额数据
      const mockQuotas: ResourceQuota[] = [
        {
          user_id: 'user-1',
          user_name: '张三',
          cpu_quota: 32,
          cpu_used: 16,
          memory_quota: 64,
          memory_used: 32,
          gpu_quota: 2,
          gpu_used: 2,
          active_experiments: 2,
          max_experiments: 5
        },
        {
          user_id: 'user-2',
          user_name: '李四',
          cpu_quota: 16,
          cpu_used: 8,
          memory_quota: 32,
          memory_used: 16,
          gpu_quota: 1,
          gpu_used: 1,
          active_experiments: 1,
          max_experiments: 3
        }
      ];

      setResourcePools(mockPools);
      setAllocations(mockAllocations);
      setQuotas(mockQuotas);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadResourceData();
    // 设置定时刷新
    const interval = setInterval(loadResourceData, 10000);
    return () => clearInterval(interval);
  }, []);

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'online':
      case 'running':
        return 'bg-green-500 text-white';
      case 'degraded':
      case 'maintenance':
      case 'queued':
        return 'bg-yellow-500 text-white';
      case 'critical':
      case 'offline':
        return 'bg-red-500 text-white';
      default:
        return 'bg-gray-500 text-white';
    }
  };

  // 获取优先级颜色
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'normal':
        return 'bg-blue-100 text-blue-800';
      case 'low':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // 格式化时间差
  const formatTimeDiff = (timeStr: string) => {
    const diff = new Date(timeStr).getTime() - Date.now();
    const hours = Math.floor(Math.abs(diff) / (1000 * 60 * 60));
    const minutes = Math.floor((Math.abs(diff) % (1000 * 60 * 60)) / (1000 * 60));
    
    if (diff > 0) {
      return `${hours}时${minutes}分后`;
    } else {
      return `${hours}时${minutes}分前`;
    }
  };

  // 渲染资源池视图
  const renderResourcePools = () => (
    <div className="space-y-6">
      {/* 资源池概览 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {resourcePools.map((pool) => (
          <Card key={pool.id} className="p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{pool.name}</h3>
                <div className="flex items-center space-x-2 mt-1">
                  <Badge className={getStatusColor(pool.status)}>
                    {pool.status}
                  </Badge>
                  <Badge className="bg-blue-100 text-blue-800">
                    {pool.type.toUpperCase()}
                  </Badge>
                </div>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setSelectedPool(selectedPool === pool.id ? null : pool.id)}
              >
                {selectedPool === pool.id ? '收起' : '展开'}
              </Button>
            </div>

            {/* 资源使用概览 */}
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>总容量</span>
                  <span>{pool.used_capacity}/{pool.total_capacity} {pool.type === 'cpu' ? '核心' : '卡'}</span>
                </div>
                <Progress 
                  value={(pool.used_capacity / pool.total_capacity) * 100} 
                  className="h-3" 
                />
              </div>
              
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <div className="text-lg font-semibold text-gray-900">{pool.total_capacity}</div>
                  <div className="text-gray-500">总容量</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-semibold text-blue-600">{pool.used_capacity}</div>
                  <div className="text-gray-500">已使用</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-semibold text-green-600">{pool.available_capacity}</div>
                  <div className="text-gray-500">可用</div>
                </div>
              </div>
            </div>

            {/* 节点详情 */}
            {selectedPool === pool.id && (
              <div className="mt-6 border-t pt-4">
                <h4 className="font-medium text-gray-900 mb-3">节点详情</h4>
                <div className="space-y-3">
                  {pool.nodes.map((node) => (
                    <div key={node.id} className="bg-gray-50 rounded-lg p-3">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <span className="font-medium text-gray-900">{node.name}</span>
                          <Badge className={`${getStatusColor(node.status)} ml-2`}>
                            {node.status}
                          </Badge>
                        </div>
                        {node.temperature && (
                          <span className="text-sm text-gray-600">
                            {node.temperature}°C
                          </span>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="text-gray-600">CPU: {node.cpu_usage.toFixed(0)}%</div>
                          <Progress value={node.cpu_usage} className="h-2 mt-1" />
                        </div>
                        <div>
                          <div className="text-gray-600">
                            内存: {((node.memory_used / node.memory_total) * 100).toFixed(0)}%
                          </div>
                          <Progress value={(node.memory_used / node.memory_total) * 100} className="h-2 mt-1" />
                        </div>
                        {node.gpu_count && (
                          <>
                            <div>
                              <div className="text-gray-600">GPU数量: {node.gpu_count}</div>
                            </div>
                            <div>
                              <div className="text-gray-600">
                                GPU内存: {node.gpu_memory_used && node.gpu_memory_total ? 
                                  ((node.gpu_memory_used / node.gpu_memory_total) * 100).toFixed(0) : 0}%
                              </div>
                              {node.gpu_memory_used && node.gpu_memory_total && (
                                <Progress value={(node.gpu_memory_used / node.gpu_memory_total) * 100} className="h-2 mt-1" />
                              )}
                            </div>
                          </>
                        )}
                      </div>
                      
                      {node.current_experiments.length > 0 && (
                        <div className="mt-2">
                          <span className="text-xs text-gray-500">
                            运行实验: {node.current_experiments.join(', ')}
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        ))}
      </div>
    </div>
  );

  // 渲染资源分配视图
  const renderResourceAllocations = () => (
    <div className="space-y-4">
      {allocations.length === 0 ? (
        <Card className="p-8 text-center">
          <div className="text-gray-500">暂无资源分配记录</div>
        </Card>
      ) : (
        allocations.map((allocation) => (
          <Card key={allocation.experiment_id} className="p-4">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="font-semibold text-gray-900">{allocation.experiment_name}</h3>
                  <Badge className={getStatusColor(allocation.status)}>
                    {allocation.status}
                  </Badge>
                  <Badge className={getPriorityColor(allocation.priority)}>
                    {allocation.priority}
                  </Badge>
                  <Badge className="bg-purple-100 text-purple-800">
                    {allocation.resource_type.toUpperCase()}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">CPU核心:</span>
                    <span className="ml-1 font-medium">{allocation.allocated_cores}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">内存:</span>
                    <span className="ml-1 font-medium">{allocation.allocated_memory}GB</span>
                  </div>
                  {allocation.allocated_gpu && (
                    <div>
                      <span className="text-gray-500">GPU:</span>
                      <span className="ml-1 font-medium">{allocation.allocated_gpu}卡</span>
                    </div>
                  )}
                  <div>
                    <span className="text-gray-500">开始时间:</span>
                    <span className="ml-1 font-medium">{formatTimeDiff(allocation.start_time)}</span>
                  </div>
                </div>
                
                {allocation.estimated_end_time && (
                  <div className="text-xs text-gray-500 mt-2">
                    预计结束: {formatTimeDiff(allocation.estimated_end_time)}
                  </div>
                )}
              </div>
              
              <div className="flex space-x-2 ml-4">
                {allocation.status === 'running' && (
                  <Button size="sm" variant="outline">
                    调整
                  </Button>
                )}
                <Button size="sm" variant="outline">
                  详情
                </Button>
              </div>
            </div>
          </Card>
        ))
      )}
    </div>
  );

  // 渲染资源配额视图
  const renderResourceQuotas = () => (
    <div className="space-y-4">
      {quotas.map((quota) => (
        <Card key={quota.user_id} className="p-4">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="font-semibold text-gray-900">{quota.user_name}</h3>
              <div className="text-sm text-gray-600">
                活跃实验: {quota.active_experiments}/{quota.max_experiments}
              </div>
            </div>
            <Button size="sm" variant="outline">
              编辑配额
            </Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* CPU配额 */}
            <div>
              <div className="text-sm font-medium text-gray-700 mb-2">CPU核心</div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>已用/配额</span>
                <span>{quota.cpu_used}/{quota.cpu_quota}</span>
              </div>
              <Progress value={(quota.cpu_used / quota.cpu_quota) * 100} className="h-3" />
            </div>
            
            {/* 内存配额 */}
            <div>
              <div className="text-sm font-medium text-gray-700 mb-2">内存 (GB)</div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>已用/配额</span>
                <span>{quota.memory_used}/{quota.memory_quota}</span>
              </div>
              <Progress value={(quota.memory_used / quota.memory_quota) * 100} className="h-3" />
            </div>
            
            {/* GPU配额 */}
            <div>
              <div className="text-sm font-medium text-gray-700 mb-2">GPU卡</div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>已用/配额</span>
                <span>{quota.gpu_used}/{quota.gpu_quota}</span>
              </div>
              <Progress value={(quota.gpu_used / quota.gpu_quota) * 100} className="h-3" />
            </div>
          </div>
        </Card>
      ))}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">资源管理</h1>
            <p className="mt-2 text-gray-600">
              管理和监控超参数优化系统的计算资源
            </p>
          </div>
          <Button onClick={loadResourceData} disabled={loading}>
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
              { key: 'pools', label: '资源池', count: resourcePools.length },
              { key: 'allocations', label: '资源分配', count: allocations.length },
              { key: 'quotas', label: '用户配额', count: quotas.length }
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
                {tab.count > 0 && (
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
          {activeTab === 'pools' && renderResourcePools()}
          {activeTab === 'allocations' && renderResourceAllocations()}
          {activeTab === 'quotas' && renderResourceQuotas()}
        </div>
      </div>
    </div>
  );
};

export default HyperparameterResourcesPage;