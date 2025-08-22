/**
 * 流式处理监控页面
 * 
 * 集成流式处理监控面板和会话管理功能
 */

import React, { useState } from 'react';
import { StreamingDashboard } from '../components/streaming/StreamingDashboard';
import { StreamingSessionManager } from '../components/streaming/StreamingSessionManager';

type TabType = 'dashboard' | 'sessions';

const StreamingMonitorPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');

  const tabs = [
    { id: 'dashboard' as TabType, name: '监控面板', description: '系统指标和背压控制状态' },
    { id: 'sessions' as TabType, name: '会话管理', description: '创建和管理流式处理会话' }
  ];

  const handleSessionCreated = (sessionId: string) => {
    console.log('新会话已创建:', sessionId);
    // 可以在这里添加通知或其他逻辑
  };

  const handleSessionStopped = (sessionId: string) => {
    console.log('会话已停止:', sessionId);
    // 可以在这里添加通知或其他逻辑
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 页面头部 */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-6">
            <div className="md:flex md:items-center md:justify-between">
              <div className="flex-1 min-w-0">
                <h1 className="text-3xl font-bold leading-tight text-gray-900">
                  流式处理监控
                </h1>
                <p className="mt-1 text-sm text-gray-500">
                  监控和管理实时AI响应流、背压控制和会话状态
                </p>
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
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="flex flex-col items-start">
                  <span>{tab.name}</span>
                  <span className="text-xs text-gray-400 mt-1">{tab.description}</span>
                </div>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* 标签页内容 */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {activeTab === 'dashboard' && (
          <div>
            <StreamingDashboard />
          </div>
        )}

        {activeTab === 'sessions' && (
          <div className="mt-6">
            <StreamingSessionManager
              onSessionCreated={handleSessionCreated}
              onSessionStopped={handleSessionStopped}
            />
          </div>
        )}
      </div>

      {/* 页面底部信息 */}
      <div className="mt-12 bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-500">
            <p>实时监控流式处理系统性能和状态</p>
            <p className="mt-1">
              支持Server-Sent Events (SSE)和WebSocket流式连接，提供背压控制和队列监控
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StreamingMonitorPage;