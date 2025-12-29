/**
 * 性能分析工具
 * 
 * 提供流式处理和批处理系统的性能分析和优化建议
 */

import React, { useState, useEffect } from 'react';
import { streamingService } from '../../services/streamingService';
import { batchService } from '../../services/batchService';

import { logger } from '../../utils/logger'
interface PerformanceMetrics {
  streaming: {
    latency_p50: number;
    latency_p95: number;
    latency_p99: number;
    throughput: number;
    error_rate: number;
    concurrent_connections: number;
    buffer_utilization: number;
    cpu_usage: number;
    memory_usage: number;
  };
  batch: {
    tasks_per_second: number;
    avg_task_duration: number;
    queue_depth: number;
    worker_utilization: number;
    success_rate: number;
    retry_rate: number;
  };
}

interface PerformanceAnalysis {
  bottlenecks: string[];
  recommendations: string[];
  score: number;
  status: 'excellent' | 'good' | 'warning' | 'critical';
}

export const PerformanceAnalyzer: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [analysis, setAnalysis] = useState<PerformanceAnalysis | null>(null);
  const [historicalData, setHistoricalData] = useState<Array<{
    timestamp: Date;
    metrics: PerformanceMetrics;
  }>>([]);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');

  // 获取性能指标
  const fetchPerformanceMetrics = async () => {
    try {
      const [streamingMetrics, flowControl, batchMetrics] = await Promise.all([
        streamingService.getSystemMetrics(),
        streamingService.getFlowControlMetrics(),
        batchService.getMetrics()
      ]);

      const perfMetrics: PerformanceMetrics = {
        streaming: {
          latency_p50: 50, // 需要从实际API获取
          latency_p95: 100, // 需要从实际API获取
          latency_p99: 200, // 需要从实际API获取
          throughput: streamingMetrics.system_metrics?.total_tokens_processed || 0,
          error_rate: 0, // 需要从实际API获取
          concurrent_connections: streamingMetrics.system_metrics?.active_sessions || 0,
          buffer_utilization: streamingMetrics.system_metrics?.total_buffer_usage || 0,
          cpu_usage: 0, // 需要从实际API获取
          memory_usage: 0 // 需要从实际API获取
        },
        batch: {
          tasks_per_second: batchMetrics.tasks_per_second,
          avg_task_duration: batchMetrics.avg_task_duration,
          queue_depth: batchMetrics.queue_depth,
          worker_utilization: (batchMetrics.active_workers / batchMetrics.max_workers) * 100,
          success_rate: batchMetrics.success_rate,
          retry_rate: (batchMetrics.failed_tasks / batchMetrics.total_tasks) * 100
        }
      };

      setMetrics(perfMetrics);
      
      // 添加到历史数据
      setHistoricalData(prev => [
        ...prev.slice(-99), // 保留最近100个数据点
        { timestamp: new Date(), metrics: perfMetrics }
      ]);

      // 分析性能
      analyzePerformance(perfMetrics);
    } catch (error) {
      logger.error('获取性能指标失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 分析性能并生成建议
  const analyzePerformance = (metrics: PerformanceMetrics) => {
    const bottlenecks: string[] = [];
    const recommendations: string[] = [];
    let score = 100;

    // 分析流式处理性能
    if (metrics.streaming.latency_p99 > 1000) {
      bottlenecks.push('流式处理延迟过高');
      recommendations.push('考虑增加缓冲区大小或优化处理逻辑');
      score -= 15;
    }

    if (metrics.streaming.error_rate > 5) {
      bottlenecks.push('流式处理错误率高');
      recommendations.push('检查错误日志，优化错误处理机制');
      score -= 20;
    }

    if (metrics.streaming.buffer_utilization > 80) {
      bottlenecks.push('缓冲区使用率过高');
      recommendations.push('启用背压控制或增加缓冲区容量');
      score -= 10;
    }

    if (metrics.streaming.cpu_usage > 80) {
      bottlenecks.push('CPU使用率过高');
      recommendations.push('优化算法或考虑横向扩展');
      score -= 15;
    }

    // 分析批处理性能
    if (metrics.batch.queue_depth > 1000) {
      bottlenecks.push('批处理队列积压严重');
      recommendations.push('增加工作线程数或优化任务处理速度');
      score -= 15;
    }

    if (metrics.batch.worker_utilization < 50) {
      bottlenecks.push('工作线程利用率低');
      recommendations.push('减少工作线程数以节省资源');
      score -= 5;
    }

    if (metrics.batch.success_rate < 90) {
      bottlenecks.push('批处理成功率低');
      recommendations.push('分析失败原因，增加重试机制');
      score -= 20;
    }

    // 确定状态
    let status: PerformanceAnalysis['status'];
    if (score >= 90) status = 'excellent';
    else if (score >= 70) status = 'good';
    else if (score >= 50) status = 'warning';
    else status = 'critical';

    // 如果没有瓶颈，添加正面反馈
    if (bottlenecks.length === 0) {
      recommendations.push('系统运行状况良好，继续保持');
    }

    setAnalysis({
      bottlenecks,
      recommendations,
      score,
      status
    });
  };

  useEffect(() => {
    fetchPerformanceMetrics();
    const interval = setInterval(fetchPerformanceMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-600 bg-green-100';
      case 'good': return 'text-blue-600 bg-blue-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatLatency = (ms: number) => {
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  if (loading) {
    return (
      <div className="text-center py-8">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <p className="mt-2 text-gray-600">分析性能中...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 mt-6">
      {/* 性能评分 */}
      {analysis && (
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">性能分析</h3>
            <div className="flex items-center space-x-4">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as any)}
                className="border border-gray-300 rounded px-3 py-1 text-sm"
              >
                <option value="1h">最近1小时</option>
                <option value="6h">最近6小时</option>
                <option value="24h">最近24小时</option>
                <option value="7d">最近7天</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 性能评分 */}
            <div className="text-center">
              <div className="relative inline-block">
                <svg className="w-32 h-32">
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke="#e5e7eb"
                    strokeWidth="8"
                    fill="none"
                  />
                  <circle
                    cx="64"
                    cy="64"
                    r="56"
                    stroke={
                      analysis.status === 'excellent' ? '#10b981' :
                      analysis.status === 'good' ? '#3b82f6' :
                      analysis.status === 'warning' ? '#f59e0b' :
                      '#ef4444'
                    }
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${(analysis.score / 100) * 352} 352`}
                    strokeLinecap="round"
                    transform="rotate(-90 64 64)"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div>
                    <div className="text-3xl font-bold">{analysis.score}</div>
                    <div className="text-sm text-gray-600">性能评分</div>
                  </div>
                </div>
              </div>
              <span className={`inline-flex mt-4 px-3 py-1 text-sm font-semibold rounded-full ${getStatusColor(analysis.status)}`}>
                {analysis.status === 'excellent' ? '优秀' :
                 analysis.status === 'good' ? '良好' :
                 analysis.status === 'warning' ? '警告' :
                 '严重'}
              </span>
            </div>

            {/* 瓶颈分析 */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">性能瓶颈</h4>
              {analysis.bottlenecks.length === 0 ? (
                <p className="text-green-600">✓ 未发现性能瓶颈</p>
              ) : (
                <ul className="space-y-2">
                  {analysis.bottlenecks.map((bottleneck, index) => (
                    <li key={index} className="flex items-start">
                      <span className="text-red-500 mr-2">•</span>
                      <span className="text-sm text-gray-700">{bottleneck}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* 优化建议 */}
            <div>
              <h4 className="font-medium text-gray-900 mb-3">优化建议</h4>
              <ul className="space-y-2">
                {analysis.recommendations.map((recommendation, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-blue-500 mr-2">→</span>
                    <span className="text-sm text-gray-700">{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* 详细指标 */}
      {metrics && (
        <>
          {/* 流式处理指标 */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">流式处理性能</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">P50延迟</div>
                <div className="text-xl font-semibold text-gray-900">
                  {formatLatency(metrics.streaming.latency_p50)}
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">P95延迟</div>
                <div className="text-xl font-semibold text-gray-900">
                  {formatLatency(metrics.streaming.latency_p95)}
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">P99延迟</div>
                <div className="text-xl font-semibold text-gray-900">
                  {formatLatency(metrics.streaming.latency_p99)}
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">吞吐量</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.streaming.throughput.toFixed(1)} token/s
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">错误率</div>
                <div className="text-xl font-semibold text-red-600">
                  {metrics.streaming.error_rate.toFixed(2)}%
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">并发连接</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.streaming.concurrent_connections}
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">CPU使用率</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.streaming.cpu_usage.toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">内存使用率</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.streaming.memory_usage.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* 批处理指标 */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">批处理性能</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">处理速率</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.batch.tasks_per_second.toFixed(1)} 任务/秒
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">平均任务时长</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.batch.avg_task_duration.toFixed(2)}s
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">队列深度</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.batch.queue_depth}
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">工作线程利用率</div>
                <div className="text-xl font-semibold text-gray-900">
                  {metrics.batch.worker_utilization.toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">成功率</div>
                <div className="text-xl font-semibold text-green-600">
                  {metrics.batch.success_rate.toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded">
                <div className="text-sm text-gray-600">重试率</div>
                <div className="text-xl font-semibold text-yellow-600">
                  {metrics.batch.retry_rate.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* 性能趋势图 */}
          {historicalData.length > 1 && (
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">性能趋势</h3>
              <div className="h-64 flex items-end space-x-1">
                {historicalData.slice(-30).map((data, index) => {
                  const height = (data.metrics.streaming.throughput / 100) * 100;
                  return (
                    <div
                      key={index}
                      className="flex-1 bg-blue-500 hover:bg-blue-600 transition-colors"
                      style={{ height: `${Math.min(height, 100)}%` }}
                      title={`吞吐量: ${data.metrics.streaming.throughput.toFixed(1)} token/s`}
                    />
                  );
                })}
              </div>
              <div className="mt-2 text-sm text-gray-600 text-center">
                吞吐量趋势（最近30个数据点）
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};