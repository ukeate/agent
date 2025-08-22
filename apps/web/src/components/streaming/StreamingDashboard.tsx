/**
 * 流式处理监控面板
 * 
 * 提供流式处理系统的实时监控和管理界面
 */

import React, { useState, useEffect } from 'react';
import { 
  streamingService, 
  StreamingMetrics, 
  BackpressureStatus, 
  FlowControlMetrics, 
  QueueStatus, 
  SessionMetrics 
} from '../../services/streamingService';

interface DashboardState {
  systemMetrics: StreamingMetrics | null;
  backpressureStatus: BackpressureStatus | null;
  flowControlMetrics: FlowControlMetrics | null;
  queueStatus: QueueStatus | null;
  sessions: Record<string, SessionMetrics>;
  healthStatus: any;
  loading: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export const StreamingDashboard: React.FC = () => {
  const [state, setState] = useState<DashboardState>({
    systemMetrics: null,
    backpressureStatus: null,
    flowControlMetrics: null,
    queueStatus: null,
    sessions: {},
    healthStatus: null,
    loading: true,
    error: null,
    lastUpdate: null
  });

  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5秒

  const fetchAllData = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));

      const [
        systemResponse,
        backpressureResponse,
        flowControlResponse,
        queueResponse,
        sessionsResponse,
        healthResponse
      ] = await Promise.all([
        streamingService.getSystemMetrics(),
        streamingService.getBackpressureStatus(),
        streamingService.getFlowControlMetrics(),
        streamingService.getQueueStatus(),
        streamingService.getSessions(),
        streamingService.getHealthStatus()
      ]);

      setState(prev => ({
        ...prev,
        systemMetrics: systemResponse.system_metrics,
        backpressureStatus: backpressureResponse.backpressure_status || null,
        flowControlMetrics: flowControlResponse.flow_control_metrics,
        queueStatus: queueResponse,
        sessions: sessionsResponse.sessions,
        healthStatus: healthResponse,
        loading: false,
        lastUpdate: new Date()
      }));
    } catch (error) {
      console.error('获取流式处理数据失败:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : '未知错误'
      }));
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchAllData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const getThrottleLevelColor = (level: string) => {
    switch (level) {
      case 'none': return 'text-green-600';
      case 'light': return 'text-yellow-500';
      case 'moderate': return 'text-orange-500';
      case 'heavy': return 'text-red-500';
      case 'severe': return 'text-red-700';
      default: return 'text-gray-500';
    }
  };

  const getThrottleLevelBg = (level: string) => {
    switch (level) {
      case 'none': return 'bg-green-50 border-green-200';
      case 'light': return 'bg-yellow-50 border-yellow-200';
      case 'moderate': return 'bg-orange-50 border-orange-200';
      case 'heavy': return 'bg-red-50 border-red-200';
      case 'severe': return 'bg-red-100 border-red-300';
      default: return 'bg-gray-50 border-gray-200';
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (state.loading && !state.lastUpdate) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-48 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* 头部控制 */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">流式处理监控面板</h1>
          {state.lastUpdate && (
            <p className="text-sm text-gray-500">
              最后更新: {state.lastUpdate.toLocaleTimeString()}
            </p>
          )}
        </div>
        
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="mr-2"
            />
            自动刷新
          </label>
          
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="border border-gray-300 rounded px-3 py-1"
            disabled={!autoRefresh}
          >
            <option value={1000}>1秒</option>
            <option value={5000}>5秒</option>
            <option value={10000}>10秒</option>
            <option value={30000}>30秒</option>
          </select>
          
          <button
            onClick={fetchAllData}
            disabled={state.loading}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {state.loading ? '刷新中...' : '立即刷新'}
          </button>
        </div>
      </div>

      {state.error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">错误</h3>
              <div className="mt-2 text-sm text-red-700">
                <p>{state.error}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 系统状态概览 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* 健康状态 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">服务状态</p>
              <p className={`text-2xl font-bold ${
                state.healthStatus?.status === 'healthy' ? 'text-green-600' : 'text-red-600'
              }`}>
                {state.healthStatus?.status === 'healthy' ? '正常' : '异常'}
              </p>
            </div>
            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
              state.healthStatus?.status === 'healthy' ? 'bg-green-100' : 'bg-red-100'
            }`}>
              <div className={`w-6 h-6 rounded-full ${
                state.healthStatus?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
            </div>
          </div>
          {state.systemMetrics && (
            <p className="text-xs text-gray-500 mt-2">
              运行时间: {formatUptime(state.systemMetrics.uptime)}
            </p>
          )}
        </div>

        {/* 活跃会话 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">活跃会话</p>
              <p className="text-2xl font-bold text-blue-600">
                {state.systemMetrics?.active_sessions || 0}
              </p>
            </div>
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            总会话: {state.systemMetrics?.total_sessions_created || 0}
          </p>
        </div>

        {/* Token处理量 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Token处理量</p>
              <p className="text-2xl font-bold text-purple-600">
                {state.systemMetrics?.total_tokens_processed?.toLocaleString() || 0}
              </p>
            </div>
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            事件: {state.systemMetrics?.total_events_processed?.toLocaleString() || 0}
          </p>
        </div>

        {/* 背压状态 */}
        <div className={`p-6 rounded-lg shadow border-2 ${
          state.backpressureStatus ? 
            getThrottleLevelBg(state.backpressureStatus.throttle_level) : 
            'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">背压状态</p>
              <p className={`text-2xl font-bold ${
                state.backpressureStatus ? 
                  getThrottleLevelColor(state.backpressureStatus.throttle_level) : 
                  'text-gray-500'
              }`}>
                {state.backpressureStatus ? 
                  state.backpressureStatus.throttle_level.toUpperCase() : 
                  '未启用'
                }
              </p>
            </div>
            <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.084 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
          </div>
          {state.backpressureStatus && (
            <p className="text-xs text-gray-500 mt-2">
              缓冲区使用率: {(state.backpressureStatus.buffer_usage_ratio * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </div>

      {/* 详细指标 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 流量控制指标 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">流量控制指标</h3>
          
          {state.flowControlMetrics && (
            <div className="space-y-4">
              {/* 速率限制器 */}
              {state.flowControlMetrics.rate_limiter_stats && (
                <div className="border rounded p-3">
                  <h4 className="font-medium text-gray-700 mb-2">速率限制器</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>限制: {state.flowControlMetrics.rate_limiter_stats.rate}/秒</div>
                    <div>突发: {state.flowControlMetrics.rate_limiter_stats.burst}</div>
                    <div>当前额度: {state.flowControlMetrics.rate_limiter_stats.current_allowance.toFixed(1)}</div>
                    <div>拒绝率: {(state.flowControlMetrics.rate_limiter_stats.rejection_rate * 100).toFixed(1)}%</div>
                  </div>
                </div>
              )}

              {/* 熔断器 */}
              {state.flowControlMetrics.circuit_breaker_state && (
                <div className="border rounded p-3">
                  <h4 className="font-medium text-gray-700 mb-2">熔断器</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>状态: 
                      <span className={`ml-1 ${
                        state.flowControlMetrics.circuit_breaker_state.state === 'CLOSED' ? 'text-green-600' :
                        state.flowControlMetrics.circuit_breaker_state.state === 'OPEN' ? 'text-red-600' :
                        'text-yellow-600'
                      }`}>
                        {state.flowControlMetrics.circuit_breaker_state.state}
                      </span>
                    </div>
                    <div>失败次数: {state.flowControlMetrics.circuit_breaker_state.failure_count}/{state.flowControlMetrics.circuit_breaker_state.failure_threshold}</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 队列状态 */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">队列监控</h3>
          
          {state.queueStatus && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">总队列数:</span>
                  <span className="ml-2 font-medium">{state.queueStatus.system_summary.total_queues}</span>
                </div>
                <div>
                  <span className="text-gray-600">过载队列:</span>
                  <span className="ml-2 font-medium text-red-600">
                    {state.queueStatus.system_summary.overloaded_queues}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">系统利用率:</span>
                  <span className="ml-2 font-medium">
                    {(state.queueStatus.system_summary.system_utilization * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">运行状态:</span>
                  <span className={`ml-2 font-medium ${
                    state.queueStatus.system_summary.is_running ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {state.queueStatus.system_summary.is_running ? '运行中' : '已停止'}
                  </span>
                </div>
              </div>

              {/* 队列详情 */}
              {Object.entries(state.queueStatus.queue_metrics).map(([name, metrics]) => (
                <div key={name} className="border rounded p-3">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium text-gray-700">{metrics.name}</h4>
                    <span className={`text-sm ${metrics.is_overloaded ? 'text-red-600' : 'text-green-600'}`}>
                      {metrics.is_overloaded ? '过载' : '正常'}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>大小: {metrics.current_size}/{metrics.max_size}</div>
                    <div>利用率: {(metrics.utilization * 100).toFixed(1)}%</div>
                    <div>吞吐率: {metrics.throughput_ratio.toFixed(2)}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 活跃会话列表 */}
      {Object.keys(state.sessions).length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">活跃会话</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    会话ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    智能体
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    状态
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Token数
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    速率
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    持续时间
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {Object.entries(state.sessions).map(([sessionId, session]) => (
                  <tr key={sessionId}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                      {sessionId.substring(0, 8)}...
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {session.agent_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        session.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                        session.status === 'completed' ? 'bg-green-100 text-green-800' :
                        session.status === 'error' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {session.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {session.token_count.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {session.tokens_per_second.toFixed(1)} t/s
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {session.duration ? formatUptime(session.duration) : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};