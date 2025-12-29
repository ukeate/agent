import React from 'react';
import { Card } from '../ui/card';
import { Progress } from '../ui/progress';

interface PerformanceMetricsProps {
  metrics: {
    active_connections?: number;
    messages_sent?: number;
    messages_failed?: number;
    subscription_stats?: Record<string, number>;
  };
}

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ metrics }) => {
  const activeConnections = metrics.active_connections || 0;
  const messagesSent = metrics.messages_sent || 0;
  const messagesFailed = metrics.messages_failed || 0;
  const successRate = messagesSent ? ((messagesSent - messagesFailed) / messagesSent) * 100 : 100;

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">WebSocket 指标</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-gray-600">活跃连接</div>
            <div className="text-2xl font-bold text-gray-900">{activeConnections}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">发送消息</div>
            <div className="text-2xl font-bold text-gray-900">{messagesSent}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">失败消息</div>
            <div className="text-2xl font-bold text-gray-900">{messagesFailed}</div>
          </div>
        </div>

        <div className="mt-6">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
            <span>消息成功率</span>
            <span>{successRate.toFixed(1)}%</span>
          </div>
          <Progress value={successRate} max={100} className="h-2" />
        </div>
      </Card>
    </div>
  );
};

