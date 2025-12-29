import React, { useState, useEffect } from 'react';
import apiClient from '../services/apiClient';

interface StructuredError {
  id: string;
  code: string;
  message: string;
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  context: {
    node_id?: string;
    session_id?: string;
    user_id?: string;
    agent_id?: string;
    operation?: string;
    component?: string;
  };
  details: Record<string, any>;
  suggestions: string[];
  related_errors: string[];
  stacktrace?: string;
  resolved: boolean;
}

interface ErrorStats {
  total_errors: number;
  by_severity: Record<string, number>;
  by_category: Record<string, number>;
  recent_count: number;
  resolution_rate: number;
}

const StructuredErrorPage: React.FC = () => {
  const [errors, setErrors] = useState<StructuredError[]>([]);
  const [stats, setStats] = useState<ErrorStats | null>(null);
  const [selectedError, setSelectedError] = useState<StructuredError | null>(null);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  // 错误代码映射
  const errorCodeCategories = {
    'SYS-': '系统错误',
    'CFG-': '配置错误', 
    'VAL-': '验证错误',
    'AUTH-': '认证错误',
    'AUTHZ-': '授权错误',
    'RES-': '资源错误',
    'NET-': '网络错误',
    'TMO-': '超时错误',
    'RATE-': '限流错误',
    'BIZ-': '业务逻辑错误'
  };

  useEffect(() => {
    const load = async () => {
      try {
        const [errRes, statsRes] = await Promise.all([
          apiClient.get('/structured-errors'),
          apiClient.get('/structured-errors/stats')
        ]);
        const list: StructuredError[] = errRes.data?.errors || errRes.data || [];
        setErrors(list);
        setStats(statsRes.data || null);
        if (list.length) setSelectedError(list[0]);
      } catch (e) {
        setErrors([]);
        setStats(null);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'bg-blue-100 text-blue-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'critical': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'system': return 'bg-red-100 text-red-800';
      case 'validation': return 'bg-green-100 text-green-800';
      case 'business_logic': return 'bg-purple-100 text-purple-800';
      case 'rate_limit': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const filteredErrors = errors.filter(error => {
    const matchesSeverity = filterSeverity === 'all' || error.severity === filterSeverity;
    const matchesCategory = filterCategory === 'all' || error.category === filterCategory;
    return matchesSeverity && matchesCategory;
  });

  if (loading) {
    return <div className="p-6">加载结构化错误处理系统...</div>;
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">结构化错误处理系统</h1>
        <p className="text-gray-600 mb-4">
          统一错误代码、详细错误上下文和本地化错误消息 - 展示structured error handling的技术实现
        </p>

        {/* 错误统计面板 */}
        {stats && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <h2 className="text-lg font-semibold mb-3">错误统计</h2>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
              <div>
                <span className="block text-sm text-gray-500">错误总数</span>
                <span className="text-2xl font-bold">{stats.total_errors}</span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">最近1小时</span>
                <span className="text-2xl font-bold text-red-600">{stats.recent_count}</span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">解决率</span>
                <span className="text-2xl font-bold text-green-600">
                  {stats.resolution_rate.toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">严重/紧急</span>
                <span className="text-2xl font-bold text-red-600">
                  {stats.by_severity.high + stats.by_severity.critical}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">中等</span>
                <span className="text-2xl font-bold text-yellow-600">
                  {stats.by_severity.medium}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">低级</span>
                <span className="text-2xl font-bold text-blue-600">
                  {stats.by_severity.low}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* 过滤器 */}
        <div className="bg-white rounded-lg shadow p-4 mb-6">
          <div className="flex gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">严重程度</label>
              <select
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
                className="border border-gray-300 rounded px-3 py-2"
              >
                <option value="all">所有级别</option>
                <option value="critical">紧急</option>
                <option value="high">严重</option>
                <option value="medium">中等</option>
                <option value="low">低级</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">错误分类</label>
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                className="border border-gray-300 rounded px-3 py-2"
              >
                <option value="all">所有分类</option>
                <option value="system">系统错误</option>
                <option value="validation">验证错误</option>
                <option value="business_logic">业务逻辑错误</option>
                <option value="rate_limit">限流错误</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 错误列表 */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow">
            <div className="px-4 py-3 border-b">
              <h2 className="text-lg font-semibold">错误记录 ({filteredErrors.length})</h2>
            </div>
            <div className="divide-y max-h-96 overflow-y-auto">
              {filteredErrors.map((error) => (
                <div 
                  key={error.id} 
                  className={`p-4 cursor-pointer hover:bg-gray-50 ${selectedError?.id === error.id ? 'bg-blue-50' : ''}`}
                  onClick={() => setSelectedError(error)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono">
                          {error.code}
                        </code>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(error.severity)}`}>
                          {error.severity.toUpperCase()}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(error.category)}`}>
                          {error.category.replace('_', ' ')}
                        </span>
                      </div>
                      <p className="text-sm text-gray-900 font-medium">{error.message}</p>
                    </div>
                    <div className="text-right">
                      <div className={`text-xs font-medium ${error.resolved ? 'text-green-600' : 'text-red-600'}`}>
                        {error.resolved ? '已解决' : '未解决'}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {new Date(error.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                  <div className="text-xs text-gray-500">
                    {error.context.component && `组件: ${error.context.component}`}
                    {error.context.operation && ` | 操作: ${error.context.operation}`}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 错误详情 */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-4 py-3 border-b">
            <h2 className="text-lg font-semibold">错误详情</h2>
          </div>
          <div className="p-4">
            {selectedError ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-gray-900 mb-2">基本信息</h3>
                  <div className="bg-gray-50 rounded p-3 text-sm space-y-1">
                    <div><strong>错误代码:</strong> {selectedError.code}</div>
                    <div><strong>分类:</strong> {selectedError.category}</div>
                    <div><strong>严重程度:</strong> {selectedError.severity}</div>
                    <div><strong>时间:</strong> {new Date(selectedError.timestamp).toLocaleString()}</div>
                  </div>
                </div>

                <div>
                  <h3 className="font-medium text-gray-900 mb-2">上下文信息</h3>
                  <div className="bg-gray-50 rounded p-3 text-sm space-y-1">
                    {Object.entries(selectedError.context).map(([key, value]) => (
                      value && <div key={key}><strong>{key}:</strong> {value}</div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="font-medium text-gray-900 mb-2">详细信息</h3>
                  <div className="bg-gray-50 rounded p-3 text-sm">
                    <pre className="whitespace-pre-wrap">
                      {JSON.stringify(selectedError.details, null, 2)}
                    </pre>
                  </div>
                </div>

                {selectedError.suggestions.length > 0 && (
                  <div>
                    <h3 className="font-medium text-gray-900 mb-2">解决建议</h3>
                    <ul className="bg-blue-50 rounded p-3 text-sm space-y-1">
                      {selectedError.suggestions.map((suggestion, index) => (
                        <li key={index} className="flex items-start">
                          <span className="mr-2">•</span>
                          <span>{suggestion}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {selectedError.stacktrace && (
                  <div>
                    <h3 className="font-medium text-gray-900 mb-2">堆栈跟踪</h3>
                    <div className="bg-gray-900 text-green-400 rounded p-3 text-xs font-mono overflow-x-auto">
                      <pre>{selectedError.stacktrace}</pre>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                选择一个错误记录查看详情
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 错误代码说明 */}
      <div className="mt-8 bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3">错误代码体系</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
          {Object.entries(errorCodeCategories).map(([prefix, description]) => (
            <div key={prefix} className="bg-white rounded p-2">
              <code className="bg-gray-100 px-2 py-1 rounded text-xs">{prefix}xxxx</code>
              <div className="mt-1 text-gray-600">{description}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm text-gray-700 space-y-2">
          <p><strong>结构化错误:</strong> 通过structured_errors.py实现统一错误处理</p>
          <p><strong>错误构建器:</strong> ErrorBuilder支持链式调用构建复杂错误信息</p>
          <p><strong>分类管理:</strong> 按错误类型和严重程度进行分类管理</p>
          <p><strong>上下文追踪:</strong> 记录完整的错误上下文信息便于调试</p>
        </div>
      </div>
    </div>
  );
};

export default StructuredErrorPage;
