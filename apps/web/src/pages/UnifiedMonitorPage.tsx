/**
 * 统一监控页面
 * 
 * 集成流式处理、批处理和性能分析的综合监控界面
 */

import React, { useState } from 'react';
import { StreamingDashboard } from '../components/streaming/StreamingDashboard';
import { StreamingSessionManager } from '../components/streaming/StreamingSessionManager';
import { BatchProcessingDashboard } from '../components/batch/BatchProcessingDashboard';
import { PerformanceAnalyzer } from '../components/streaming/PerformanceAnalyzer';
import FaultToleranceMonitor from '../components/streaming/FaultToleranceMonitor';
import CheckpointManager from '../components/batch/CheckpointManager';
import SchedulingMonitor from '../components/batch/SchedulingMonitor';

type TabType = 'streaming' | 'batch' | 'sessions' | 'performance' | 'fault-tolerance' | 'checkpoints' | 'scheduling';

const UnifiedMonitorPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('streaming');

  const tabs = [
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