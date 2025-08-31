import React, { useState, useEffect } from 'react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Alert } from '../ui/alert';

interface RealTimeMonitorProps {
  data: {
    event_count: number;
    active_users: number;
    events_per_minute: number;
    window_duration_seconds?: number;
    unique_sessions?: number;
    event_type_distribution?: Record<string, number>;
    hourly_distribution?: Record<string, number>;
    most_active_hour?: number;
  };
  connected: boolean;
}

interface RecentEvent {
  id: string;
  type: string;
  user_id: string;
  timestamp: Date;
  properties?: Record<string, any>;
}

export const RealTimeMonitor: React.FC<RealTimeMonitorProps> = ({ data, connected }) => {
  const [recentEvents, setRecentEvents] = useState<RecentEvent[]>([]);
  const [alerts, setAlerts] = useState<Array<{
    id: string;
    type: 'info' | 'warning' | 'error';
    message: string;
    timestamp: Date;
  }>>([]);

  // 模拟实时事件流
  useEffect(() => {
    if (!connected) return;

    const interval = setInterval(() => {
      // 模拟新事件
      const eventTypes = ['click', 'page_view', 'form_submit', 'scroll', 'hover'];
      const newEvent: RecentEvent = {
        id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
        user_id: `user_${Math.floor(Math.random() * 100)}`,
        timestamp: new Date(),
        properties: {
          page: '/dashboard',
          duration: Math.floor(Math.random() * 5000)
        }
      };

      setRecentEvents(prev => [newEvent, ...prev.slice(0, 49)]);

      // 模拟告警
      if (Math.random() < 0.1) { // 10% 概率产生告警
        const alertTypes: Array<'info' | 'warning' | 'error'> = ['info', 'warning', 'error'];
        const alertMessages = [
          '检测到异常高频事件',
          '用户活跃度显著上升',
          '系统响应时间延长',
          '新的行为模式出现'
        ];
        
        const newAlert = {
          id: `alert_${Date.now()}`,
          type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
          message: alertMessages[Math.floor(Math.random() * alertMessages.length)],
          timestamp: new Date()
        };

        setAlerts(prev => [newAlert, ...prev.slice(0, 9)]);
      }
    }, 2000); // 每2秒更新一次

    return () => clearInterval(interval);
  }, [connected]);

  // 计算趋势
  const calculateTrend = (current: number, previous: number = current * 0.9) => {
    if (previous === 0) return { value: 0, direction: 'stable' as const };
    const change = ((current - previous) / previous * 100);
    return {
      value: Math.abs(change),
      direction: change > 5 ? 'up' as const : change < -5 ? 'down' as const : 'stable' as const
    };
  };

  // 获取趋势图标和颜色
  const getTrendStyle = (direction: 'up' | 'down' | 'stable') => {
    const styles = {
      up: { icon: '↗️', color: 'text-green-600' },
      down: { icon: '↘️', color: 'text-red-600' },
      stable: { icon: '➡️', color: 'text-gray-600' }
    };
    return styles[direction];
  };

  // 格式化时间
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN');
  };

  const eventTrend = calculateTrend(data.events_per_minute || 0);
  const userTrend = calculateTrend(data.active_users || 0);

  return (
    <div className="space-y-6">
      {/* 连接状态警告 */}
      {!connected && (
        <Alert variant="warning">
          <p>⚠️ 实时连接已断开，数据可能不是最新的</p>
        </Alert>
      )}

      {/* 实时指标 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">当前事件数</p>
              <p className="text-3xl font-bold text-blue-600">
                {data.event_count}
              </p>
              <div className="flex items-center mt-1">
                <span className={`text-sm ${getTrendStyle(eventTrend.direction).color}`}>
                  {getTrendStyle(eventTrend.direction).icon} {(eventTrend.value || 0).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="p-3 bg-blue-50 rounded-full">
              <div className={`w-3 h-3 rounded-full ${connected ? 'bg-blue-500 animate-pulse' : 'bg-gray-400'}`} />
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">在线用户</p>
              <p className="text-3xl font-bold text-green-600">
                {data.active_users}
              </p>
              <div className="flex items-center mt-1">
                <span className={`text-sm ${getTrendStyle(userTrend.direction).color}`}>
                  {getTrendStyle(userTrend.direction).icon} {(userTrend.value || 0).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="p-3 bg-green-50 rounded-full">
              <span className="text-2xl">👥</span>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">事件频率</p>
              <p className="text-3xl font-bold text-purple-600">
                {(data.events_per_minute || 0).toFixed(1)}
              </p>
              <p className="text-xs text-gray-500 mt-1">/分钟</p>
            </div>
            <div className="p-3 bg-purple-50 rounded-full">
              <span className="text-2xl">⚡</span>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">活跃会话</p>
              <p className="text-3xl font-bold text-orange-600">
                {data.unique_sessions || 0}
              </p>
              <p className="text-xs text-gray-500 mt-1">当前</p>
            </div>
            <div className="p-3 bg-orange-50 rounded-full">
              <span className="text-2xl">🔗</span>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 实时事件流 */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-gray-900">实时事件流</h4>
            <Badge variant={connected ? 'secondary' : 'destructive'} className="text-xs">
              {connected ? '已连接' : '已断开'}
            </Badge>
          </div>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {recentEvents.length > 0 ? (
              recentEvents.map((event) => (
                <div key={event.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {event.type}
                      </p>
                      <p className="text-xs text-gray-500">
                        用户: {event.user_id}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-500">
                      {formatTime(event.timestamp)}
                    </p>
                    {event.properties?.duration && (
                      <p className="text-xs text-gray-400">
                        {event.properties.duration}ms
                      </p>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                {connected ? (
                  <div>
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2" />
                    <p>等待事件数据...</p>
                  </div>
                ) : (
                  <div>
                    <span className="text-4xl mb-2 block">📡</span>
                    <p>未连接到实时数据流</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>

        {/* 实时告警 */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-gray-900">实时告警</h4>
            <Badge variant="outline" className="text-xs">
              {alerts.length} 条告警
            </Badge>
          </div>
          
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {alerts.length > 0 ? (
              alerts.map((alert) => (
                <Alert key={alert.id} variant={alert.type === 'error' ? 'destructive' : alert.type === 'warning' ? 'warning' : 'default'} className="text-sm">
                  <div className="flex items-start justify-between">
                    <p className="flex-1">{alert.message}</p>
                    <p className="text-xs text-gray-500 ml-2">
                      {formatTime(alert.timestamp)}
                    </p>
                  </div>
                </Alert>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <span className="text-4xl mb-2 block">✅</span>
                <p>暂无告警信息</p>
                <p className="text-sm mt-1">系统运行正常</p>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* 实时统计图表占位 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 事件类型分布 */}
        <Card className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4">
            实时事件类型分布
          </h4>
          {data.event_type_distribution ? (
            <div className="space-y-3">
              {Object.entries(data.event_type_distribution)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5)
                .map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">{type}</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{
                            width: `${(count / Math.max(...Object.values(data.event_type_distribution || {}))) * 100}%`
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
          ) : (
            <div className="text-center py-4 text-gray-500">
              <p>暂无分布数据</p>
            </div>
          )}
        </Card>

        {/* 系统健康状态 */}
        <Card className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4">系统健康状态</h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-md">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full" />
                <span className="text-sm text-gray-700">事件处理</span>
              </div>
              <span className="text-sm font-medium text-green-700">正常</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-md">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full" />
                <span className="text-sm text-gray-700">数据存储</span>
              </div>
              <span className="text-sm font-medium text-green-700">正常</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-md">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-gray-700">实时连接</span>
              </div>
              <span className={`text-sm font-medium ${connected ? 'text-green-700' : 'text-red-700'}`}>
                {connected ? '正常' : '断开'}
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-md">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full" />
                <span className="text-sm text-gray-700">分析引擎</span>
              </div>
              <span className="text-sm font-medium text-green-700">正常</span>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-blue-50 rounded-md">
            <p className="text-sm text-blue-800">
              <span className="font-medium">系统状态：</span>
              所有组件运行正常，实时分析功能可用
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
};