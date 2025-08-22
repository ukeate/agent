import React from 'react';
import { Card } from '../ui/Card';
import { Progress } from '../ui/Progress';
import { Badge } from '../ui/Badge';

interface BehaviorOverviewProps {
  data: {
    total_events?: number;
    unique_users?: number;
    unique_sessions?: number;
    events_per_minute?: number;
    event_type_distribution?: Record<string, number>;
    hourly_distribution?: Record<string, number>;
    most_active_hour?: number;
  };
  realtime: {
    event_count: number;
    active_users: number;
    events_per_minute: number;
  };
}

export const BehaviorOverview: React.FC<BehaviorOverviewProps> = ({ data, realtime }) => {
  // 计算增长率（示例数据）
  const getGrowthRate = (current: number, previous?: number) => {
    const currentNum = Number(current || 0);
    const previousNum = Number(previous || currentNum * 0.9);
    if (previousNum === 0) return 0;
    return ((currentNum - previousNum) / previousNum * 100).toFixed(1);
  };

  // 格式化数字
  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  // 获取最活跃的事件类型
  const getTopEventTypes = () => {
    if (!data.event_type_distribution) return [];
    
    return Object.entries(data.event_type_distribution)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([type, count]) => ({
        type,
        count,
        percentage: ((count / (data.total_events || 1)) * 100).toFixed(1)
      }));
  };

  // 获取时段分布数据
  const getHourlyData = () => {
    if (!data.hourly_distribution) return [];
    
    return Object.entries(data.hourly_distribution)
      .map(([hour, count]) => ({
        hour: parseInt(hour),
        count,
        label: `${hour}:00`
      }))
      .sort((a, b) => a.hour - b.hour);
  };

  const topEventTypes = getTopEventTypes();
  const hourlyData = getHourlyData();
  const maxHourlyCount = Math.max(...hourlyData.map(h => h.count));

  return (
    <div className="space-y-6">
      {/* 关键指标卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* 总事件数 */}
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">总事件数</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatNumber(data.total_events || 0)}
              </p>
              <p className="text-xs text-green-600 flex items-center mt-1">
                <span className="mr-1">↗</span>
                {getGrowthRate(data.total_events || 0)}% vs 昨天
              </p>
            </div>
            <div className="p-3 bg-blue-50 rounded-full">
              <span className="text-2xl">📊</span>
            </div>
          </div>
        </Card>

        {/* 活跃用户数 */}
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">活跃用户</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatNumber(data.unique_users || 0)}
              </p>
              <p className="text-xs text-green-600 flex items-center mt-1">
                <span className="mr-1">↗</span>
                {getGrowthRate(data.unique_users || 0)}% vs 昨天
              </p>
            </div>
            <div className="p-3 bg-green-50 rounded-full">
              <span className="text-2xl">👥</span>
            </div>
          </div>
        </Card>

        {/* 会话数 */}
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">总会话数</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatNumber(data.unique_sessions || 0)}
              </p>
              <p className="text-xs text-green-600 flex items-center mt-1">
                <span className="mr-1">↗</span>
                {getGrowthRate(data.unique_sessions || 0)}% vs 昨天
              </p>
            </div>
            <div className="p-3 bg-purple-50 rounded-full">
              <span className="text-2xl">🔗</span>
            </div>
          </div>
        </Card>

        {/* 每分钟事件数 */}
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">事件频率</p>
              <p className="text-2xl font-bold text-gray-900">
                {Number(data.events_per_minute || 0).toFixed(1)}/min
              </p>
              <p className="text-xs text-orange-600 flex items-center mt-1">
                <span className="mr-1">⚡</span>
                实时: {Number(realtime.events_per_minute || 0).toFixed(1)}/min
              </p>
            </div>
            <div className="p-3 bg-orange-50 rounded-full">
              <span className="text-2xl">⚡</span>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 事件类型分布 */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            热门事件类型
          </h3>
          <div className="space-y-4">
            {topEventTypes.map((event, index) => (
              <div key={event.type} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <Badge variant={index < 3 ? 'default' : 'secondary'}>
                      #{index + 1}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {event.type}
                    </p>
                    <p className="text-xs text-gray-500">
                      {event.count} 次事件
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-900">
                    {event.percentage}%
                  </p>
                  <div className="w-16 mt-1">
                    <Progress 
                      value={parseFloat(event.percentage)} 
                      max={100}
                      className="h-2"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* 时段分布 */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            24小时活跃度分布
          </h3>
          <div className="space-y-2">
            {hourlyData.map((hour) => (
              <div key={hour.hour} className="flex items-center space-x-3">
                <div className="w-12 text-xs text-gray-600 flex-shrink-0">
                  {hour.label}
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          hour.hour === data.most_active_hour
                            ? 'bg-blue-500'
                            : 'bg-gray-400'
                        }`}
                        style={{
                          width: `${(hour.count / maxHourlyCount) * 100}%`,
                        }}
                      />
                    </div>
                    <div className="w-8 text-xs text-gray-600 text-right">
                      {hour.count}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          {data.most_active_hour !== undefined && (
            <div className="mt-4 p-3 bg-blue-50 rounded-md">
              <p className="text-sm text-blue-800">
                <span className="font-medium">最活跃时段：</span>
                {data.most_active_hour}:00 - {data.most_active_hour + 1}:00
              </p>
            </div>
          )}
        </Card>
      </div>

      {/* 实时状态 */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          实时状态
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-md">
            <p className="text-2xl font-bold text-green-600">
              {realtime.event_count}
            </p>
            <p className="text-sm text-green-800">当前活动事件</p>
          </div>
          <div className="text-center p-4 bg-blue-50 rounded-md">
            <p className="text-2xl font-bold text-blue-600">
              {realtime.active_users}
            </p>
            <p className="text-sm text-blue-800">在线用户</p>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-md">
            <p className="text-2xl font-bold text-purple-600">
              {Number(realtime.events_per_minute || 0).toFixed(1)}
            </p>
            <p className="text-sm text-purple-800">事件/分钟</p>
          </div>
        </div>
      </Card>
    </div>
  );
};