import React, { useState, useEffect } from 'react';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/button';
import { Input } from '../../components/ui/input';
import { Badge } from '../../components/ui/badge';
import { behaviorAnalyticsService } from '../../services/behaviorAnalyticsService';

import { logger } from '../../utils/logger'
interface BehaviorEvent {
  event_id: string;
  user_id: string;
  session_id: string;
  event_type: string;
  timestamp: string;
  properties: Record<string, any>;
  context?: Record<string, any>;
}

interface EventFilter {
  user_id?: string;
  session_id?: string;
  event_type?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
  offset?: number;
}

export const EventDataManagePage: React.FC = () => {
  const [events, setEvents] = useState<BehaviorEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<EventFilter>({
    limit: 50,
    offset: 0
  });
  const [totalEvents, setTotalEvents] = useState(0);
  const [eventTypes, setEventTypes] = useState<string[]>([]);

  // 获取事件数据
  const fetchEvents = async () => {
    setLoading(true);
    try {
      const response = await behaviorAnalyticsService.getEvents(filter);
      setEvents(response.events || []);
      setTotalEvents(response.total || 0);
      
      // 提取事件类型用于筛选
      const types = [...new Set(response.events?.map(e => e.event_type) || [])];
      setEventTypes(types);
    } catch (error) {
      logger.error('获取事件数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [filter]);

  // 格式化时间戳
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN');
  };

  // 获取事件类型颜色
  const getEventTypeColor = (eventType: string) => {
    const colors: Record<string, string> = {
      'click': 'bg-blue-100 text-blue-800',
      'page_view': 'bg-green-100 text-green-800',
      'form_submit': 'bg-orange-100 text-orange-800',
      'error': 'bg-red-100 text-red-800',
      'api_call': 'bg-purple-100 text-purple-800',
    };
    return colors[eventType] || 'bg-gray-100 text-gray-800';
  };

  // 导出事件数据
  const handleExport = async () => {
    try {
      const blob = await behaviorAnalyticsService.exportEvents('csv', {
        user_id: filter.user_id,
        start_time: filter.start_time,
        end_time: filter.end_time,
        limit: filter.limit
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `events_${new Date().toISOString()}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      logger.error('导出失败:', error);
    }
  };

  // 分页处理
  const handlePageChange = (page: number) => {
    setFilter(prev => ({
      ...prev,
      offset: (page - 1) * (prev.limit || 50)
    }));
  };

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">事件数据管理</h1>
          <p className="text-sm text-gray-600 mt-1">
            管理和查询用户行为事件数据，支持筛选、导出和批量操作
          </p>
        </div>
        <div className="flex space-x-3">
          <Button onClick={handleExport} variant="outline">
            📥 导出数据
          </Button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{totalEvents}</p>
            <p className="text-sm text-gray-600">总事件数</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">{eventTypes.length}</p>
            <p className="text-sm text-gray-600">事件类型数</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">{events.length}</p>
            <p className="text-sm text-gray-600">当前显示</p>
          </div>
        </Card>
      </div>

      {/* 筛选器 */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">数据筛选</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">用户ID</label>
            <Input
              placeholder="输入用户ID"
              value={filter.user_id || ''}
              onChange={(e) => setFilter(prev => ({ ...prev, user_id: e.target.value || undefined }))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">会话ID</label>
            <Input
              placeholder="输入会话ID"
              value={filter.session_id || ''}
              onChange={(e) => setFilter(prev => ({ ...prev, session_id: e.target.value || undefined }))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">事件类型</label>
            <select
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              value={filter.event_type || ''}
              onChange={(e) => setFilter(prev => ({ ...prev, event_type: e.target.value || undefined }))}
            >
              <option value="">全部类型</option>
              {eventTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">开始时间</label>
            <Input
              type="datetime-local"
              value={filter.start_time || ''}
              onChange={(e) => setFilter(prev => ({ ...prev, start_time: e.target.value || undefined }))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">结束时间</label>
            <Input
              type="datetime-local"
              value={filter.end_time || ''}
              onChange={(e) => setFilter(prev => ({ ...prev, end_time: e.target.value || undefined }))}
            />
          </div>
        </div>
        <div className="mt-4 flex space-x-2">
          <Button onClick={fetchEvents}>
            🔍 应用筛选
          </Button>
          <Button 
            variant="outline"
            onClick={() => {
              setFilter({ limit: 50, offset: 0 });
            }}
          >
            🔄 重置
          </Button>
        </div>
      </Card>

      {/* 事件列表 */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">事件列表</h3>
        </div>

        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2 text-gray-600">加载中...</p>
          </div>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {events.map((event) => (
              <div
                key={event.event_id}
                className="p-4 border rounded-md transition-colors border-gray-200 hover:border-gray-300"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <Badge className={getEventTypeColor(event.event_type)}>
                          {event.event_type}
                        </Badge>
                        <span className="text-sm text-gray-500">
                          ID: {event.event_id.substring(0, 8)}...
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">用户:</span>
                          <span className="ml-1 font-medium">{event.user_id}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">会话:</span>
                          <span className="ml-1 font-medium">{event.session_id.substring(0, 8)}...</span>
                        </div>
                      </div>
                      {Object.keys(event.properties || {}).length > 0 && (
                        <div className="mt-2">
                          <details className="text-sm">
                            <summary className="text-gray-500 cursor-pointer">
                              属性 ({Object.keys(event.properties).length} 项)
                            </summary>
                            <pre className="mt-1 p-2 bg-gray-50 rounded text-xs overflow-x-auto">
                              {JSON.stringify(event.properties, null, 2)}
                            </pre>
                          </details>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-right text-sm text-gray-500">
                    {formatTimestamp(event.timestamp)}
                  </div>
                </div>
              </div>
            ))}
            
            {events.length === 0 && !loading && (
              <div className="text-center py-8 text-gray-500">
                <p>暂无事件数据</p>
                <p className="text-sm mt-1">尝试调整筛选条件</p>
              </div>
            )}
          </div>
        )}

        {/* 分页 */}
        {totalEvents > (filter.limit || 50) && (
          <div className="mt-6 flex items-center justify-between">
            <div className="text-sm text-gray-600">
              显示 {(filter.offset || 0) + 1} - {Math.min((filter.offset || 0) + (filter.limit || 50), totalEvents)} 条，
              共 {totalEvents} 条记录
            </div>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                disabled={(filter.offset || 0) === 0}
                onClick={() => handlePageChange(Math.floor((filter.offset || 0) / (filter.limit || 50)))}
              >
                上一页
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={(filter.offset || 0) + (filter.limit || 50) >= totalEvents}
                onClick={() => handlePageChange(Math.floor((filter.offset || 0) / (filter.limit || 50)) + 2)}
              >
                下一页
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default EventDataManagePage;
