/**
 * 统一监控页面
 * 
 * 集成流式处理、批处理和性能分析的综合监控界面
 */

import React, { useState, useEffect } from 'react';
import { StreamingDashboard } from '../components/streaming/StreamingDashboard';
import { StreamingSessionManager } from '../components/streaming/StreamingSessionManager';
import { BatchProcessingDashboard } from '../components/batch/BatchProcessingDashboard';
import { PerformanceAnalyzer } from '../components/streaming/PerformanceAnalyzer';
import FaultToleranceMonitor from '../components/streaming/FaultToleranceMonitor';
import CheckpointManager from '../components/batch/CheckpointManager';
import SchedulingMonitor from '../components/batch/SchedulingMonitor';
import { unifiedService } from '../services/unifiedService';
import { streamingService } from '../services/streamingService';

import { logger } from '../utils/logger'
type TabType = 'modules' | 'metrics' | 'streaming' | 'batch' | 'sessions' | 'performance' | 'fault-tolerance' | 'checkpoints' | 'scheduling';

const UnifiedMonitorPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('modules');
  const [unifiedMetrics, setUnifiedMetrics] = useState<any>(null);
  const [modulesStatus, setModulesStatus] = useState<any>(null);
  const [systemMetrics, setSystemMetrics] = useState<any>(null);
  const [monitoringSummary, setMonitoringSummary] = useState<any>(null);
  const [monitoringAlerts, setMonitoringAlerts] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadUnifiedData();
    const interval = setInterval(loadUnifiedData, 30000); // 每30秒刷新
    return () => clearInterval(interval);
  }, []);

  const loadUnifiedData = async () => {
    try {
      const [metrics, modules, sysMetrics, summary, alerts] = await Promise.all([
        unifiedService.getUnifiedMetrics().catch(err => {
          logger.warn('加载统一指标失败:', err);
          return null;
        }),
        unifiedService.getModulesStatus().catch(err => {
          logger.warn('加载模块状态失败:', err);
          return null;
        }),
        unifiedService.getSystemMetrics().catch(err => {
          logger.warn('加载系统指标失败:', err);
          return null;
        }),
        unifiedService.getMonitoringSummary().catch(err => {
          logger.warn('加载监控汇总失败:', err);
          return null;
        }),
        unifiedService.getMonitoringAlerts().catch(err => {
          logger.warn('加载监控告警失败:', err);
          return null;
        })
      ]);
      setUnifiedMetrics(metrics);
      setModulesStatus(modules);
      setSystemMetrics(sysMetrics);
      setMonitoringSummary(summary);
      setMonitoringAlerts(alerts);
    } catch (error) {
      logger.error('加载统一数据失败:', error);
    }
  };

  const tabs = [
    { 
      id: 'modules' as TabType, 
      name: '模块状态', 
      description: '系统各模块运行状态监控',
      icon: '🔧'
    },
    { 
      id: 'metrics' as TabType, 
      name: '系统监控', 
      description: '系统性能指标和告警监控',
      icon: '📊'
    },
    { 
      id: 'streaming' as TabType, 
      name: '流式处理', 
      description: '实时监控流式处理系统',
      icon: '📡'
    },
    { 
      id: 'batch' as TabType, 
      name: '批处理', 
      description: '批处理作业管理和进度跟踪',
      icon: '📦'
    },
    { 
      id: 'sessions' as TabType, 
      name: '会话管理', 
      description: '流式会话创建和管理',
      icon: '💬'
    },
    { 
      id: 'performance' as TabType, 
      name: '性能分析', 
      description: '系统性能分析和优化建议',
      icon: '📊'
    },
    { 
      id: 'fault-tolerance' as TabType, 
      name: '容错监控', 
      description: '连接状态和故障恢复监控',
      icon: '🛡️'
    },
    { 
      id: 'checkpoints' as TabType, 
      name: '检查点管理', 
      description: '批处理检查点和断点续传',
      icon: '💾'
    },
    { 
      id: 'scheduling' as TabType, 
      name: '智能调度', 
      description: '资源感知和SLA监控',
      icon: '⚡'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
        {/* 页面头部 */}
        <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-6">
            <div className="md:flex md:items-center md:justify-between">
              <div className="flex-1 min-w-0">
                <h1 className="text-3xl font-bold leading-tight text-gray-900">
                  统一监控中心
                </h1>
                <p className="mt-1 text-sm text-gray-500">
                  全面监控流式处理、批处理和系统性能
                </p>
              </div>
              <div className="mt-4 md:mt-0 md:ml-4">
                <div className="flex items-center space-x-2">
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    <span className="w-2 h-2 mr-1 bg-green-400 rounded-full animate-pulse"></span>
                    系统运行中
                  </span>
                  <span className="text-sm text-gray-500">
                    {new Date().toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 标签页导航 */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{tab.icon}</span>
                  <div className="flex flex-col items-start">
                    <span>{tab.name}</span>
                    <span className="text-xs text-gray-400 mt-1">{tab.description}</span>
                  </div>
                </div>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* 标签页内容 */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {activeTab === 'modules' && (
          <div className="mt-6">
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-900">系统模块状态</h2>
                <button 
                  onClick={loadUnifiedData}
                  className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                >
                  🔄 刷新
                </button>
              </div>
              
              {modulesStatus ? (
                modulesStatus.modules ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(modulesStatus.modules || {}).map(([key, module]: [string, any]) => (
                      <div key={key} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-lg font-medium text-gray-900">{module.name}</h3>
                          <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            module.health === 'healthy' 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {module.health === 'healthy' ? '✅ 健康' : '❌ 异常'}
                          </div>
                        </div>
                        <div className="space-y-2 text-sm text-gray-600">
                          <div>状态: <span className={`font-medium ${
                            module.status === 'active' ? 'text-green-600' : 'text-red-600'
                          }`}>{module.status === 'active' ? '运行中' : '停止'}</span></div>
                          <div>版本: <span className="font-medium text-gray-900">{module.version}</span></div>
                          <div>最后检查: <span className="font-medium text-gray-900">
                            {new Date(module.last_check).toLocaleString()}
                          </span></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : Array.isArray(modulesStatus.data?.loaded_modules) ? (
                  <div className="space-y-2 text-sm text-gray-700">
                    {modulesStatus.data.loaded_modules.map((m: string) => (
                      <div key={m} className="border rounded-lg px-3 py-2">
                        {m}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="text-gray-500">模块状态数据格式不支持</div>
                  </div>
                )
              ) : (
                <div className="text-center py-12">
                  <div className="text-gray-500">
                    {loading ? '加载中...' : '无法加载模块状态数据'}
                  </div>
                </div>
              )}
              
              {modulesStatus && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <div className="text-sm text-gray-500">
                    最后更新: {new Date(modulesStatus.timestamp || Date.now()).toLocaleString()}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="mt-6 space-y-6">
            {/* 监控摘要 */}
            {monitoringSummary && (
              <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">监控摘要</h2>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">{monitoringSummary.health_score}%</div>
                    <div className="text-sm text-green-700">健康度</div>
                  </div>
                  <div className="text-center p-4 bg-red-50 rounded-lg">
                    <div className="text-2xl font-bold text-red-600">{monitoringSummary.active_alerts}</div>
                    <div className="text-sm text-red-700">活跃告警</div>
                  </div>
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{monitoringSummary.performance_metrics?.avg_response_time}ms</div>
                    <div className="text-sm text-blue-700">平均响应时间</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">{monitoringSummary.performance_metrics?.success_rate_percent}%</div>
                    <div className="text-sm text-purple-700">成功率</div>
                  </div>
                </div>
                
                {/* 资源使用情况 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 border rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">CPU使用率</span>
                      <span className="text-sm text-gray-600">{monitoringSummary.resource_usage?.cpu_percent}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{width: `${monitoringSummary.resource_usage?.cpu_percent}%`}}></div>
                    </div>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">内存使用率</span>
                      <span className="text-sm text-gray-600">{monitoringSummary.resource_usage?.memory_percent}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-green-600 h-2 rounded-full" style={{width: `${monitoringSummary.resource_usage?.memory_percent}%`}}></div>
                    </div>
                  </div>
                  <div className="p-4 border rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">磁盘使用率</span>
                      <span className="text-sm text-gray-600">{monitoringSummary.resource_usage?.disk_percent}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-yellow-600 h-2 rounded-full" style={{width: `${monitoringSummary.resource_usage?.disk_percent}%`}}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 告警信息 */}
            {monitoringAlerts && monitoringAlerts.length > 0 && (
              <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">活跃告警</h2>
                <div className="space-y-3">
                  {monitoringAlerts.map((alert: any) => (
                    <div key={alert.id} className={`border-l-4 p-4 rounded-md ${
                      alert.severity === 'critical' ? 'border-red-500 bg-red-50' :
                      alert.severity === 'warning' ? 'border-yellow-500 bg-yellow-50' :
                      'border-blue-500 bg-blue-50'
                    }`}>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              alert.severity === 'critical' ? 'bg-red-100 text-red-800' :
                              alert.severity === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-blue-100 text-blue-800'
                            }`}>
                              {alert.severity === 'critical' ? '🔴 严重' :
                               alert.severity === 'warning' ? '🟡 警告' : '🔵 信息'}
                            </span>
                            <span className="ml-2 font-medium text-gray-900">{alert.title}</span>
                          </div>
                          <p className="mt-1 text-sm text-gray-600">{alert.message}</p>
                          <div className="mt-2 text-xs text-gray-500">
                            来源: {alert.source} | {new Date(alert.timestamp).toLocaleString()}
                          </div>
                        </div>
                        {alert.acknowledged && (
                          <span className="ml-4 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            ✅ 已确认
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 系统指标 */}
            {systemMetrics && (
              <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">系统指标详情</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* 系统资源 */}
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-3">系统资源</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span>CPU使用率:</span>
                        <span className="font-medium">{systemMetrics.system?.cpu_usage}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>内存使用率:</span>
                        <span className="font-medium">{systemMetrics.system?.memory_usage}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>磁盘使用率:</span>
                        <span className="font-medium">{systemMetrics.system?.disk_usage}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>系统运行时间:</span>
                        <span className="font-medium">{Math.floor(systemMetrics.system?.uptime / 3600)}小时</span>
                      </div>
                    </div>
                  </div>

                  {/* 应用指标 */}
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-3">应用指标</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span>活跃请求:</span>
                        <span className="font-medium">{systemMetrics.application?.active_requests}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>总请求数:</span>
                        <span className="font-medium">{systemMetrics.application?.total_requests?.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>错误率:</span>
                        <span className="font-medium">{(systemMetrics.application?.error_rate * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>平均响应时间:</span>
                        <span className="font-medium">{systemMetrics.application?.average_response_time}ms</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 服务状态 */}
                <div className="mt-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-3">服务状态</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Object.entries(systemMetrics.services || {}).map(([serviceName, service]: [string, any]) => (
                      <div key={serviceName} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{serviceName.toUpperCase()}</span>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            service.status === 'healthy' ? 'bg-green-100 text-green-800' :
                            service.status === 'degraded' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {service.status === 'healthy' ? '✅ 健康' :
                             service.status === 'degraded' ? '⚠️ 降级' : '❌ 异常'}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          {service.connections && (
                            <div>连接数: {service.connections}</div>
                          )}
                          {service.query_time && (
                            <div>查询时间: {service.query_time}ms</div>
                          )}
                          {service.memory_usage && (
                            <div>内存: {service.memory_usage}MB</div>
                          )}
                          {service.hit_rate && (
                            <div>命中率: {(service.hit_rate * 100).toFixed(1)}%</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'streaming' && (
          <div>
            <StreamingDashboard />
          </div>
        )}

        {activeTab === 'batch' && (
          <div>
            <BatchProcessingDashboard />
          </div>
        )}

        {activeTab === 'sessions' && (
          <div className="mt-6">
            <StreamingSessionManager />
          </div>
        )}

        {activeTab === 'performance' && (
          <div>
            <PerformanceAnalyzer />
          </div>
        )}

        {activeTab === 'fault-tolerance' && (
          <div className="mt-6">
            <FaultToleranceMonitor />
          </div>
        )}

        {activeTab === 'checkpoints' && (
          <div className="mt-6">
            <CheckpointManager />
          </div>
        )}

        {activeTab === 'scheduling' && (
          <div className="mt-6">
            <SchedulingMonitor />
          </div>
        )}
      </div>

      {/* 页面底部信息 */}
      <div className="mt-12 bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-gray-600">
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">流式处理特性</h4>
              <ul className="space-y-1">
                <li>• SSE/WebSocket实时流</li>
                <li>• Token级别流式输出</li>
                <li>• 背压和流量控制</li>
                <li>• 容错和断线重连</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">批处理特性</h4>
              <ul className="space-y-1">
                <li>• 智能任务调度</li>
                <li>• 检查点和断点续传</li>
                <li>• SLA监控和保证</li>
                <li>• 资源感知分配</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">高级功能</h4>
              <ul className="space-y-1">
                <li>• 预测性资源调度</li>
                <li>• 数据一致性保证</li>
                <li>• 熔断器和重试机制</li>
                <li>• 实时性能分析</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UnifiedMonitorPage;
