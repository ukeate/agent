import React, { useState, useEffect } from 'react';

interface BackpressureMetrics {
  strategy: string;
  status: 'active' | 'inactive' | 'triggering';
  queue_size: number;
  max_queue_size: number;
  throughput: number;
  target_throughput: number;
  drop_rate: number;
  backpressure_ratio: number;
}

interface CircuitBreakerState {
  name: string;
  state: 'CLOSED' | 'OPEN' | 'HALF_OPEN';
  failure_count: number;
  failure_threshold: number;
  success_count: number;
  last_failure_time?: string;
  next_attempt_time?: string;
}

interface FlowControlData {
  backpressure_controllers: BackpressureMetrics[];
  circuit_breakers: CircuitBreakerState[];
  flow_stats: {
    total_tasks_processed: number;
    tasks_dropped: number;
    average_latency_ms: number;
    current_throughput: number;
  };
}

const FlowControlPage: React.FC = () => {
  const [data, setData] = useState<FlowControlData | null>(null);
  const [loading, setLoading] = useState(true);

  // 模拟实时数据更新
  useEffect(() => {
    const generateData = (): FlowControlData => ({
      backpressure_controllers: [
        {
          strategy: 'QueueBasedBackpressure',
          status: Math.random() > 0.7 ? 'triggering' : 'active',
          queue_size: Math.floor(Math.random() * 8000) + 1000,
          max_queue_size: 10000,
          throughput: Math.floor(Math.random() * 500) + 200,
          target_throughput: 600,
          drop_rate: Math.random() * 5,
          backpressure_ratio: Math.random() * 0.3
        },
        {
          strategy: 'ThroughputBasedBackpressure',
          status: Math.random() > 0.8 ? 'triggering' : 'active',
          queue_size: Math.floor(Math.random() * 5000) + 500,
          max_queue_size: 8000,
          throughput: Math.floor(Math.random() * 400) + 150,
          target_throughput: 500,
          drop_rate: Math.random() * 3,
          backpressure_ratio: Math.random() * 0.4
        },
        {
          strategy: 'AdaptiveBackpressure',
          status: 'active',
          queue_size: Math.floor(Math.random() * 3000) + 200,
          max_queue_size: 6000,
          throughput: Math.floor(Math.random() * 600) + 300,
          target_throughput: 800,
          drop_rate: Math.random() * 2,
          backpressure_ratio: Math.random() * 0.2
        }
      ],
      circuit_breakers: [
        {
          name: 'DatabaseConnection',
          state: Math.random() > 0.9 ? 'OPEN' : 'CLOSED',
          failure_count: Math.floor(Math.random() * 3),
          failure_threshold: 5,
          success_count: Math.floor(Math.random() * 100) + 50,
          last_failure_time: new Date(Date.now() - Math.random() * 300000).toISOString(),
        },
        {
          name: 'ExternalAPICall',
          state: Math.random() > 0.8 ? 'HALF_OPEN' : 'CLOSED',
          failure_count: Math.floor(Math.random() * 2),
          failure_threshold: 3,
          success_count: Math.floor(Math.random() * 200) + 100,
        },
        {
          name: 'AgentPoolAccess',
          state: 'CLOSED',
          failure_count: 0,
          failure_threshold: 10,
          success_count: Math.floor(Math.random() * 500) + 200,
        }
      ],
      flow_stats: {
        total_tasks_processed: Math.floor(Math.random() * 50000) + 10000,
        tasks_dropped: Math.floor(Math.random() * 500) + 50,
        average_latency_ms: Math.floor(Math.random() * 200) + 50,
        current_throughput: Math.floor(Math.random() * 800) + 200,
      }
    });

    setData(generateData());
    setLoading(false);

    const interval = setInterval(() => {
      setData(generateData());
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100';
      case 'triggering': return 'text-red-600 bg-red-100';
      case 'inactive': return 'text-gray-600 bg-gray-100';
      case 'CLOSED': return 'text-green-600 bg-green-100';
      case 'OPEN': return 'text-red-600 bg-red-100';
      case 'HALF_OPEN': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading) {
    return <div className="p-6">加载流控监控系统...</div>;
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">流控与背压监控系统</h1>
        <p className="text-gray-600 mb-4">
          实时监控背压机制和熔断器状态 - 展示flow control和backpressure机制的技术实现
        </p>

        {/* 流控统计概览 */}
        {data && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <h2 className="text-lg font-semibold mb-3">流控统计概览</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <span className="block text-sm text-gray-500">处理任务总数</span>
                <span className="text-2xl font-bold text-blue-600">
                  {data.flow_stats.total_tasks_processed.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">丢弃任务数</span>
                <span className="text-2xl font-bold text-red-600">
                  {data.flow_stats.tasks_dropped.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">平均延迟</span>
                <span className="text-2xl font-bold text-green-600">
                  {data.flow_stats.average_latency_ms}ms
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">当前吞吐量</span>
                <span className="text-2xl font-bold text-purple-600">
                  {data.flow_stats.current_throughput}/s
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* 背压控制器 */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-3 border-b">
            <h2 className="text-lg font-semibold">背压控制器</h2>
          </div>
          <div className="p-4 space-y-4">
            {data?.backpressure_controllers.map((controller, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium">{controller.strategy}</h3>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(controller.status)}`}>
                    {controller.status.toUpperCase()}
                  </span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">队列使用率:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${controller.queue_size / controller.max_queue_size > 0.8 ? 'bg-red-500' : 'bg-blue-500'}`}
                          style={{width: `${(controller.queue_size / controller.max_queue_size) * 100}%`}}
                        />
                      </div>
                      <span>{controller.queue_size}/{controller.max_queue_size}</span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600">吞吐量:</span>
                    <span>{controller.throughput}/{controller.target_throughput} tasks/s</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600">丢弃率:</span>
                    <span className={controller.drop_rate > 3 ? 'text-red-600 font-medium' : 'text-gray-900'}>
                      {controller.drop_rate.toFixed(2)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600">背压比例:</span>
                    <span>{(controller.backpressure_ratio * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 熔断器状态 */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-3 border-b">
            <h2 className="text-lg font-semibold">熔断器状态</h2>
          </div>
          <div className="p-4 space-y-4">
            {data?.circuit_breakers.map((breaker, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium">{breaker.name}</h3>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(breaker.state)}`}>
                    {breaker.state}
                  </span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">失败次数:</span>
                    <span className={breaker.failure_count >= breaker.failure_threshold ? 'text-red-600 font-medium' : 'text-gray-900'}>
                      {breaker.failure_count}/{breaker.failure_threshold}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600">成功次数:</span>
                    <span className="text-green-600">{breaker.success_count}</span>
                  </div>
                  
                  {breaker.last_failure_time && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">最后失败:</span>
                      <span className="text-xs">
                        {new Date(breaker.last_failure_time).toLocaleTimeString()}
                      </span>
                    </div>
                  )}
                  
                  {breaker.next_attempt_time && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">下次尝试:</span>
                      <span className="text-xs">
                        {new Date(breaker.next_attempt_time).toLocaleTimeString()}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 技术实现说明 */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3">技术实现说明</h3>
        <div className="text-sm text-gray-700 space-y-2">
          <p><strong>背压控制策略:</strong> QueueBased（队列）、ThroughputBased（吞吐量）、Adaptive（自适应）</p>
          <p><strong>流控机制:</strong> 通过flow_control.py实现多种背压策略和任务处理器</p>
          <p><strong>熔断器模式:</strong> 实现CLOSED/OPEN/HALF_OPEN状态转换，防止级联故障</p>
          <p><strong>实时监控:</strong> 队列使用率、吞吐量、丢弃率等关键指标实时展示</p>
          <p><strong>自适应调节:</strong> 根据系统负载动态调整背压阈值和处理策略</p>
        </div>
      </div>
    </div>
  );
};

export default FlowControlPage;