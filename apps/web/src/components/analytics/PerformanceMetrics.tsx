import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Progress } from '../ui/Progress';
import { Alert } from '../ui/Alert';

interface PerformanceMetricsProps {
  metrics: {
    total_connections?: number;
    active_connections?: number;
    messages_sent?: number;
    messages_failed?: number;
    subscription_stats?: Record<string, number>;
  };
}

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  threshold: number;
  status: 'good' | 'warning' | 'critical';
  description: string;
}

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ metrics }) => {
  const [selectedCategory, setSelectedCategory] = useState<'overview' | 'websocket' | 'system' | 'recommendations'>('overview');

  // 模拟系统性能指标
  const systemMetrics: SystemMetric[] = [
    {
      name: 'CPU使用率',
      value: 45.2,
      unit: '%',
      threshold: 80,
      status: 'good',
      description: '系统CPU使用率正常'
    },
    {
      name: '内存使用率',
      value: 67.8,
      unit: '%',
      threshold: 85,
      status: 'good',
      description: '内存使用在正常范围内'
    },
    {
      name: '事件处理延迟',
      value: 12.5,
      unit: 'ms',
      threshold: 100,
      status: 'good',
      description: '事件处理响应时间优秀'
    },
    {
      name: '数据库连接池',
      value: 15,
      unit: '连接',
      threshold: 100,
      status: 'good',
      description: '数据库连接数正常'
    },
    {
      name: '缓存命中率',
      value: 89.3,
      unit: '%',
      threshold: 70,
      status: 'good',
      description: '缓存效率良好'
    },
    {
      name: '异常检测延迟',
      value: 234,
      unit: 'ms',
      threshold: 500,
      status: 'good',
      description: '异常检测性能正常'
    }
  ];

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    const colors = {
      good: 'text-green-600 bg-green-100',
      warning: 'text-yellow-600 bg-yellow-100',
      critical: 'text-red-600 bg-red-100'
    };
    return colors[status as keyof typeof colors] || colors.good;
  };

  // 获取状态图标
  const getStatusIcon = (status: string) => {
    const icons = {
      good: '✅',
      warning: '⚠️',
      critical: '🚨'
    };
    return icons[status as keyof typeof icons] || icons.good;
  };

  // 计算整体健康评分
  const calculateHealthScore = () => {
    const goodMetrics = systemMetrics.filter(m => m.status === 'good').length;
    const warningMetrics = systemMetrics.filter(m => m.status === 'warning').length;
    const criticalMetrics = systemMetrics.filter(m => m.status === 'critical').length;
    
    const score = (goodMetrics * 100 + warningMetrics * 60 + criticalMetrics * 20) / systemMetrics.length;
    return Math.round(score);
  };

  // WebSocket 性能分析
  const websocketPerformance = {
    connectionRate: metrics.total_connections ? 
      ((metrics.active_connections || 0) / metrics.total_connections * 100).toFixed(1) : '0',
    messageSuccessRate: metrics.messages_sent ? 
      (((metrics.messages_sent - (metrics.messages_failed || 0)) / metrics.messages_sent) * 100).toFixed(1) : '100',
    throughput: (metrics.messages_sent || 0) / 60, // 每分钟消息数
  };

  const healthScore = calculateHealthScore();

  return (
    <div className="space-y-6">
      {/* 类别选择 */}
      <Card className="p-4">
        <div className="flex space-x-1">
          {[
            { id: 'overview', label: '总览', icon: '📊' },
            { id: 'websocket', label: 'WebSocket', icon: '🔗' },
            { id: 'system', label: '系统指标', icon: '🖥️' },
            { id: 'recommendations', label: '优化建议', icon: '💡' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedCategory(tab.id as any)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                selectedCategory === tab.id
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </Card>

      {/* 总览 */}
      {selectedCategory === 'overview' && (
        <div className="space-y-6">
          {/* 健康评分 */}
          <Card className="p-6">
            <div className="text-center">
              <div className="relative inline-block">
                <div className="w-32 h-32 mx-auto mb-4">
                  <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                    <path
                      className="text-gray-300"
                      strokeWidth="3"
                      fill="none"
                      strokeDasharray="100, 100"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      stroke="currentColor"
                    />
                    <path
                      className={healthScore >= 80 ? 'text-green-500' : healthScore >= 60 ? 'text-yellow-500' : 'text-red-500'}
                      strokeWidth="3"
                      fill="none"
                      strokeLinecap="round"
                      strokeDasharray={`${healthScore}, 100`}
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      stroke="currentColor"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl font-bold text-gray-900">{healthScore}</span>
                  </div>
                </div>
                <h3 className="text-lg font-semibold text-gray-900">系统健康评分</h3>
                <p className="text-sm text-gray-600">
                  {healthScore >= 80 ? '系统运行优秀' : 
                   healthScore >= 60 ? '系统运行良好' : '系统需要关注'}
                </p>
              </div>
            </div>
          </Card>

          {/* 关键指标概览 */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">
                  {systemMetrics.filter(m => m.status === 'good').length}
                </p>
                <p className="text-sm text-gray-600">正常指标</p>
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-yellow-600">
                  {systemMetrics.filter(m => m.status === 'warning').length}
                </p>
                <p className="text-sm text-gray-600">警告指标</p>
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-red-600">
                  {systemMetrics.filter(m => m.status === 'critical').length}
                </p>
                <p className="text-sm text-gray-600">严重指标</p>
              </div>
            </Card>
            
            <Card className="p-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">
                  {metrics.active_connections || 0}
                </p>
                <p className="text-sm text-gray-600">活跃连接</p>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* WebSocket 性能 */}
      {selectedCategory === 'websocket' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="p-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-blue-600">
                  {metrics.total_connections || 0}
                </p>
                <p className="text-sm text-gray-600">总连接数</p>
                <p className="text-xs text-gray-500 mt-1">
                  活跃: {metrics.active_connections || 0}
                </p>
              </div>
            </Card>

            <Card className="p-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-green-600">
                  {websocketPerformance.messageSuccessRate}%
                </p>
                <p className="text-sm text-gray-600">消息成功率</p>
                <p className="text-xs text-gray-500 mt-1">
                  失败: {metrics.messages_failed || 0}
                </p>
              </div>
            </Card>

            <Card className="p-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-purple-600">
                  {websocketPerformance.throughput.toFixed(1)}
                </p>
                <p className="text-sm text-gray-600">消息吞吐量</p>
                <p className="text-xs text-gray-500 mt-1">消息/分钟</p>
              </div>
            </Card>
          </div>

          {/* 订阅统计 */}
          {metrics.subscription_stats && (
            <Card className="p-6">
              <h4 className="font-semibold text-gray-900 mb-4">订阅分布</h4>
              <div className="space-y-3">
                {Object.entries(metrics.subscription_stats).map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700 capitalize">
                      {type.replace(/_/g, ' ')}
                    </span>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full"
                          style={{
                            width: `${(count / Math.max(...Object.values(metrics.subscription_stats || {}))) * 100}%`
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-8">
                        {count}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* 系统指标详情 */}
      {selectedCategory === 'system' && (
        <div className="space-y-4">
          {systemMetrics.map((metric) => (
            <Card key={metric.name} className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <h5 className="font-medium text-gray-900">{metric.name}</h5>
                    <Badge className={`text-xs ${getStatusColor(metric.status)}`}>
                      {getStatusIcon(metric.status)} {metric.status}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600">{metric.description}</p>
                </div>
                
                <div className="text-right ml-6">
                  <p className="text-2xl font-bold text-gray-900">
                    {metric.value}
                    <span className="text-sm font-normal text-gray-500 ml-1">
                      {metric.unit}
                    </span>
                  </p>
                  <div className="w-32 mt-2">
                    <Progress
                      value={metric.unit === '%' ? metric.value : (metric.value / metric.threshold) * 100}
                      max={100}
                      className="h-2"
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    阈值: {metric.threshold} {metric.unit}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* 优化建议 */}
      {selectedCategory === 'recommendations' && (
        <div className="space-y-4">
          <Alert variant="default">
            <h4 className="font-semibold mb-2">📊 数据收集优化</h4>
            <p className="text-sm mb-2">
              当前事件缓冲区使用率较低，可以适当增加批处理大小以提高效率。
            </p>
            <p className="text-xs text-gray-600">
              建议：将批处理大小从 100 增加到 200
            </p>
          </Alert>

          <Alert variant="default">
            <h4 className="font-semibold mb-2">🚀 性能表现良好</h4>
            <p className="text-sm mb-2">
              WebSocket连接稳定，消息传输成功率达到 {websocketPerformance.messageSuccessRate}%，
              系统整体运行状况良好。
            </p>
          </Alert>

          <Alert variant="warning">
            <h4 className="font-semibold mb-2">⚡ 连接池优化</h4>
            <p className="text-sm mb-2">
              建议监控数据库连接池使用情况，在高峰期可能需要调整连接池大小。
            </p>
            <p className="text-xs text-gray-600">
              当前连接池大小：20，建议考虑扩容至 30
            </p>
          </Alert>

          <Card className="p-6">
            <h4 className="font-semibold text-gray-900 mb-4">性能优化清单</h4>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-bold">
                  ✓
                </div>
                <span className="text-sm text-gray-700">启用数据压缩传输</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-bold">
                  ✓
                </div>
                <span className="text-sm text-gray-700">配置缓存策略</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-yellow-100 text-yellow-600 rounded-full flex items-center justify-center text-xs font-bold">
                  !
                </div>
                <span className="text-sm text-gray-700">优化数据库索引</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-yellow-100 text-yellow-600 rounded-full flex items-center justify-center text-xs font-bold">
                  !
                </div>
                <span className="text-sm text-gray-700">实现连接池监控</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-gray-100 text-gray-600 rounded-full flex items-center justify-center text-xs font-bold">
                  ○
                </div>
                <span className="text-sm text-gray-700">部署负载均衡</span>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};