import React, { useState, useEffect } from 'react';

interface MetricPoint {
  timestamp: string;
  value: number;
}

interface Metric {
  name: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
  description: string;
  unit: string;
  current_value: number | null;
  points: MetricPoint[];
  statistics: {
    count: number;
    min: number;
    max: number;
    avg: number;
    median: number;
    latest: number;
  };
}

interface Alert {
  id: string;
  metric_name: string;
  level: 'info' | 'warning' | 'critical' | 'emergency';
  message: string;
  timestamp: string;
  threshold: number;
  actual_value: number;
  resolved: boolean;
}

interface DashboardData {
  timestamp: string;
  metrics: Record<string, Metric>;
  alerts: {
    active: Alert[];
    recent: Alert[];
  };
  summary: {
    system_status: 'healthy' | 'warning' | 'critical';
    total_metrics: number;
    active_alerts: number;
    critical_alerts: number;
    key_metrics: Record<string, { value: number | null; unit: string }>;
  };
}

const MonitoringDashboardPage: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string>('system_cpu_usage');
  const [loading, setLoading] = useState(true);

  // 模拟实时数据更新
  useEffect(() => {
    const generateDashboardData = (): DashboardData => {
      const now = new Date();
      const metrics: Record<string, Metric> = {};

      // 系统指标
      const systemMetrics = [
        {
          name: 'system_cpu_usage',
          type: 'gauge' as const,
          description: 'CPU使用率',
          unit: '%',
          baseValue: 45,
          variation: 30
        },
        {
          name: 'system_memory_usage', 
          type: 'gauge' as const,
          description: '内存使用率',
          unit: '%',
          baseValue: 65,
          variation: 20
        },
        {
          name: 'agent_pool_active',
          type: 'gauge' as const,
          description: '活跃智能体数量',
          unit: 'count',
          baseValue: 8,
          variation: 5
        },
        {
          name: 'task_completion_rate',
          type: 'gauge' as const,
          description: '任务完成率',
          unit: '%',
          baseValue: 92,
          variation: 8
        },
        {
          name: 'flow_control_throughput',
          type: 'gauge' as const,
          description: '流控吞吐量',
          unit: 'tasks/sec',
          baseValue: 450,
          variation: 200
        },
        {
          name: 'event_bus_messages',
          type: 'counter' as const,
          description: '事件总数',
          unit: 'count',
          baseValue: 15430,
          variation: 50
        },
        {
          name: 'error_count_by_severity',
          type: 'counter' as const,
          description: '按严重性分组的错误数',
          unit: 'count',
          baseValue: 23,
          variation: 5
        }
      ];

      systemMetrics.forEach(metric => {
        const currentValue = metric.baseValue + (Math.random() - 0.5) * metric.variation;
        const points = Array.from({ length: 60 }, (_, i) => ({
          timestamp: new Date(now.getTime() - (59 - i) * 60000).toISOString(),
          value: metric.baseValue + (Math.random() - 0.5) * metric.variation
        }));

        const values = points.map(p => p.value);
        metrics[metric.name] = {
          name: metric.name,
          type: metric.type,
          description: metric.description,
          unit: metric.unit,
          current_value: currentValue,
          points,
          statistics: {
            count: values.length,
            min: Math.min(...values),
            max: Math.max(...values),
            avg: values.reduce((a, b) => a + b, 0) / values.length,
            median: values.sort((a, b) => a - b)[Math.floor(values.length / 2)],
            latest: values[values.length - 1]
          }
        };
      });

      const alerts: Alert[] = [
        {
          id: 'alert-001',
          metric_name: 'system_cpu_usage',
          level: 'warning',
          message: 'CPU使用率过高: 91.2% > 90%',
          timestamp: new Date(now.getTime() - 300000).toISOString(),
          threshold: 90,
          actual_value: 91.2,
          resolved: false
        },
        {
          id: 'alert-002', 
          metric_name: 'task_completion_rate',
          level: 'critical',
          message: '任务完成率过低: 85.3% < 90%',
          timestamp: new Date(now.getTime() - 180000).toISOString(),
          threshold: 90,
          actual_value: 85.3,
          resolved: false
        }
      ];

      const activeAlerts = alerts.filter(a => !a.resolved);
      const criticalAlerts = activeAlerts.filter(a => a.level === 'critical' || a.level === 'emergency');

      return {
        timestamp: now.toISOString(),
        metrics,
        alerts: {
          active: activeAlerts,
          recent: alerts.slice(0, 10)
        },
        summary: {
          system_status: criticalAlerts.length > 0 ? 'critical' : 
                        activeAlerts.length > 0 ? 'warning' : 'healthy',
          total_metrics: Object.keys(metrics).length,
          active_alerts: activeAlerts.length,
          critical_alerts: criticalAlerts.length,
          key_metrics: {
            system_cpu_usage: { value: metrics.system_cpu_usage?.current_value || null, unit: '%' },
            system_memory_usage: { value: metrics.system_memory_usage?.current_value || null, unit: '%' },
            agent_pool_active: { value: metrics.agent_pool_active?.current_value || null, unit: 'count' },
            task_completion_rate: { value: metrics.task_completion_rate?.current_value || null, unit: '%' },
            flow_control_throughput: { value: metrics.flow_control_throughput?.current_value || null, unit: 'tasks/sec' }
          }
        }
      };
    };

    setDashboardData(generateDashboardData());
    setLoading(false);

    const interval = setInterval(() => {
      setDashboardData(generateDashboardData());
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'info': return 'bg-blue-100 text-blue-800';
      case 'warning': return 'bg-yellow-100 text-yellow-800';
      case 'critical': return 'bg-red-100 text-red-800';
      case 'emergency': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return <div className="p-6">加载监控仪表板...</div>;
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">企业级监控仪表板</h1>
        <p className="text-gray-600 mb-4">
          实时指标监控和告警系统 - 展示monitoring dashboard的技术实现
        </p>

        {/* 系统状态概览 */}
        {dashboardData && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">系统状态概览</h2>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">状态:</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(dashboardData.summary.system_status)}`}>
                  {dashboardData.summary.system_status.toUpperCase()}
                </span>
                <span className="text-sm text-gray-500 ml-4">
                  更新时间: {new Date(dashboardData.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <span className="block text-sm text-gray-500">监控指标</span>
                <span className="text-2xl font-bold">{dashboardData.summary.total_metrics}</span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">活跃告警</span>
                <span className={`text-2xl font-bold ${dashboardData.summary.active_alerts > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {dashboardData.summary.active_alerts}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">严重告警</span>
                <span className={`text-2xl font-bold ${dashboardData.summary.critical_alerts > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {dashboardData.summary.critical_alerts}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">CPU使用率</span>
                <span className="text-2xl font-bold text-blue-600">
                  {dashboardData.summary.key_metrics.system_cpu_usage.value?.toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">内存使用率</span>
                <span className="text-2xl font-bold text-purple-600">
                  {dashboardData.summary.key_metrics.system_memory_usage.value?.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* 关键指标卡片 */}
        {dashboardData && (
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
            {Object.entries(dashboardData.summary.key_metrics).map(([name, metric]) => (
              <div key={name} className="bg-white rounded-lg shadow p-4">
                <div className="text-sm text-gray-500 mb-1">
                  {dashboardData.metrics[name]?.description || name}
                </div>
                <div className="text-2xl font-bold mb-1">
                  {metric.value?.toFixed(1)} {metric.unit}
                </div>
                <div className="text-xs text-gray-400">
                  {dashboardData.metrics[name]?.type}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 指标详情 */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="px-4 py-3 border-b">
              <h2 className="text-lg font-semibold">指标详情</h2>
            </div>
            <div className="p-4">
              {dashboardData && (
                <>
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">选择指标</label>
                    <select
                      value={selectedMetric}
                      onChange={(e) => setSelectedMetric(e.target.value)}
                      className="border border-gray-300 rounded px-3 py-2 w-full max-w-sm"
                    >
                      {Object.entries(dashboardData.metrics).map(([name, metric]) => (
                        <option key={name} value={name}>
                          {metric.description} ({metric.unit})
                        </option>
                      ))}
                    </select>
                  </div>

                  {dashboardData.metrics[selectedMetric] && (
                    <div>
                      <div className="mb-4">
                        <h3 className="font-medium text-gray-900 mb-2">
                          {dashboardData.metrics[selectedMetric].description}
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="block text-gray-500">当前值</span>
                            <span className="font-medium">
                              {dashboardData.metrics[selectedMetric].current_value?.toFixed(2)} {dashboardData.metrics[selectedMetric].unit}
                            </span>
                          </div>
                          <div>
                            <span className="block text-gray-500">平均值</span>
                            <span className="font-medium">
                              {dashboardData.metrics[selectedMetric].statistics.avg.toFixed(2)} {dashboardData.metrics[selectedMetric].unit}
                            </span>
                          </div>
                          <div>
                            <span className="block text-gray-500">最小值</span>
                            <span className="font-medium">
                              {dashboardData.metrics[selectedMetric].statistics.min.toFixed(2)} {dashboardData.metrics[selectedMetric].unit}
                            </span>
                          </div>
                          <div>
                            <span className="block text-gray-500">最大值</span>
                            <span className="font-medium">
                              {dashboardData.metrics[selectedMetric].statistics.max.toFixed(2)} {dashboardData.metrics[selectedMetric].unit}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* 简化的时间序列图表 */}
                      <div className="bg-gray-50 rounded p-4">
                        <h4 className="font-medium mb-2">最近60分钟趋势</h4>
                        <div className="flex items-end h-32 gap-1">
                          {dashboardData.metrics[selectedMetric].points.slice(-30).map((point, index) => {
                            const max = dashboardData.metrics[selectedMetric].statistics.max;
                            const min = dashboardData.metrics[selectedMetric].statistics.min;
                            const height = max > min ? ((point.value - min) / (max - min)) * 100 : 50;
                            return (
                              <div
                                key={index}
                                className="bg-blue-500 rounded-t"
                                style={{
                                  height: `${Math.max(height, 2)}%`,
                                  width: '100%',
                                }}
                                title={`${point.value.toFixed(2)} ${dashboardData.metrics[selectedMetric].unit}`}
                              />
                            );
                          })}
                        </div>
                        <div className="text-xs text-gray-500 mt-2">
                          最近30个数据点（每2分钟一个）
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* 活跃告警 */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-3 border-b">
            <h2 className="text-lg font-semibold">活跃告警</h2>
          </div>
          <div className="p-4">
            {dashboardData?.alerts.active.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <div className="text-green-600 mb-2">✓</div>
                <div>无活跃告警</div>
              </div>
            ) : (
              <div className="space-y-3">
                {dashboardData?.alerts.active.map((alert) => (
                  <div key={alert.id} className="border rounded-lg p-3">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAlertColor(alert.level)}`}>
                          {alert.level.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    <p className="text-sm text-gray-900 mb-2">{alert.message}</p>
                    <div className="text-xs text-gray-500">
                      指标: {alert.metric_name} | 阈值: {alert.threshold} | 实际值: {alert.actual_value}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 技术实现说明 */}
      <div className="mt-8 bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3">技术实现说明</h3>
        <div className="text-sm text-gray-700 space-y-2">
          <p><strong>指标收集:</strong> 通过MetricCollector收集系统、应用、流控、事件等15+内置指标</p>
          <p><strong>告警系统:</strong> 7个默认告警规则，支持自定义阈值和持续时间条件</p>
          <p><strong>实时更新:</strong> 仪表板数据每秒更新，告警评估每30秒执行一次</p>
          <p><strong>数据存储:</strong> 指标数据保留1000个数据点，告警历史保留1000条记录</p>
          <p><strong>可视化:</strong> 简化版时间序列图表展示指标趋势变化</p>
        </div>
      </div>
    </div>
  );
};

export default MonitoringDashboardPage;