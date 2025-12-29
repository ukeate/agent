import React from 'react';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { Alert } from '../ui/alert';

type RealtimeStats = {
  event_count: number;
  active_users: number;
  events_per_minute: number;
  window_duration_seconds?: number;
  unique_sessions?: number;
  event_type_distribution?: Record<string, number>;
  hourly_distribution?: Record<string, number>;
  most_active_hour?: number;
};

type RealtimeEvent = {
  id?: string;
  event_type?: string;
  user_id?: string;
  session_id?: string;
  timestamp?: string;
  properties?: Record<string, any>;
};

interface RealTimeMonitorProps {
  stats: RealtimeStats;
  events: RealtimeEvent[];
  connected: boolean;
}

const formatTime = (value?: string) => {
  if (!value) return '';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleTimeString('zh-CN');
};

export const RealTimeMonitor: React.FC<RealTimeMonitorProps> = ({ stats, events, connected }) => {
  return (
    <div className="space-y-6">
      {!connected && (
        <Alert variant="warning">
          <p>实时连接已断开</p>
        </Alert>
      )}

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-6">
          <p className="text-sm font-medium text-gray-600">当前事件数</p>
          <p className="text-3xl font-bold text-blue-600">{stats.event_count}</p>
          {stats.window_duration_seconds ? (
            <p className="text-xs text-gray-500 mt-1">窗口: {stats.window_duration_seconds}s</p>
          ) : null}
        </Card>

        <Card className="p-6">
          <p className="text-sm font-medium text-gray-600">在线用户</p>
          <p className="text-3xl font-bold text-green-600">{stats.active_users}</p>
        </Card>

        <Card className="p-6">
          <p className="text-sm font-medium text-gray-600">事件频率</p>
          <p className="text-3xl font-bold text-purple-600">{stats.events_per_minute.toFixed(1)}</p>
          <p className="text-xs text-gray-500 mt-1">/分钟</p>
        </Card>

        <Card className="p-6">
          <p className="text-sm font-medium text-gray-600">活跃会话</p>
          <p className="text-3xl font-bold text-orange-600">{stats.unique_sessions || 0}</p>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-semibold text-gray-900">实时事件流</h4>
            <Badge variant={connected ? 'secondary' : 'destructive'} className="text-xs">
              {connected ? '已连接' : '已断开'}
            </Badge>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {events.length ? (
              events.map((event, idx) => (
                <div key={event.id || `${event.timestamp || 't'}_${idx}`} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-gray-400'}`} />
                    <div>
                      <p className="text-sm font-medium text-gray-900">{event.event_type || 'unknown'}</p>
                      <p className="text-xs text-gray-500">用户: {event.user_id || '-'}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-500">{formatTime(event.timestamp)}</p>
                    {event.session_id ? <p className="text-xs text-gray-400">{event.session_id}</p> : null}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>{connected ? '等待事件数据...' : '未连接到实时数据流'}</p>
              </div>
            )}
          </div>
        </Card>

        <Card className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4">实时事件类型分布</h4>
          {stats.event_type_distribution ? (
            <div className="space-y-3">
              {Object.entries(stats.event_type_distribution)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 8)
                .map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">{type}</span>
                    <span className="text-sm font-medium text-gray-900">{count}</span>
                  </div>
                ))}
            </div>
          ) : (
            <div className="text-center py-4 text-gray-500">
              <p>暂无分布数据</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

