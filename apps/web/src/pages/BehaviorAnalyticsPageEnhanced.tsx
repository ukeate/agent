import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Input } from '../components/ui/input';
import { Textarea } from '../components/ui/textarea';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Progress } from '../components/ui/progress';
import { Separator } from '../components/ui/separator';
import { Calendar, Download, Users, TrendingUp, AlertTriangle, BarChart3, Brain, Target, Settings, Activity } from 'lucide-react';
import { analyticsServiceEnhanced } from '../services/analyticsServiceEnhanced';

import { logger } from '../utils/logger'
const BehaviorAnalyticsPageEnhanced: React.FC = () => {
  const [activeTab, setActiveTab] = useState('realtime');
  const [realtimeStats, setRealtimeStats] = useState<any>(null);
  const [reportStatus, setReportStatus] = useState<any>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [anomalies, setAnomalies] = useState<any>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // 实时数据流管理
  useEffect(() => {
    if (activeTab === 'realtime') {
      initializeRealtimeStream();
    }
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [activeTab]);

  const initializeRealtimeStream = async () => {
    try {
      const stream = await analyticsServiceEnhanced.getRealtimeEventStream();
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      eventSourceRef.current = stream;
      
      stream.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setRealtimeStats(prev => ({ ...prev, ...data }));
        } catch (error) {
          logger.error('解析实时数据失败:', error);
        }
      };
      
      stream.onerror = (error) => {
        logger.error('实时数据流错误:', error);
        setError('实时数据流连接失败');
        stream.close();
        if (eventSourceRef.current === stream) {
          eventSourceRef.current = null;
        }
      };
    } catch (err) {
      setError('无法建立实时数据流连接');
    }
  };

  const loadWebSocketStats = async () => {
    try {
      setLoading(true);
      const stats = await analyticsServiceEnhanced.getWebSocketStats();
      setRealtimeStats(stats);
    } catch (err) {
      setError('获取WebSocket统计失败');
    } finally {
      setLoading(false);
    }
  };

  const generateAdvancedReport = async () => {
    try {
      setLoading(true);
      const report = await analyticsServiceEnhanced.generateAdvancedReport({
        report_type: 'comprehensive',
        format: 'json',
        include_visualizations: true
      });
      setReportStatus(report);
    } catch (err) {
      setError('生成报告失败');
    } finally {
      setLoading(false);
    }
  };

  const performDeepUserAnalysis = async () => {
    try {
      setLoading(true);
      const analysis = await analyticsServiceEnhanced.performDeepUserAnalysis('sample-user', {
        include_behavior_timeline: true,
        include_conversion_funnel: true,
        include_retention_analysis: true,
        include_segmentation: true,
        time_window_days: 30
      });
      setAnalysisResults(analysis);
    } catch (err) {
      setError('深度用户分析失败');
    } finally {
      setLoading(false);
    }
  };

  const performCohortAnalysis = async () => {
    try {
      setLoading(true);
      const cohortAnalysis = await analyticsServiceEnhanced.performCohortAnalysis({
        cohort_type: 'behavioral',
        time_period: 'weekly',
        start_date: '2024-01-01',
        end_date: '2024-12-31'
      });
      setAnalysisResults(prev => ({ ...prev, cohort: cohortAnalysis }));
    } catch (err) {
      setError('队列分析失败');
    } finally {
      setLoading(false);
    }
  };

  const performAttributionAnalysis = async () => {
    try {
      setLoading(true);
      const attribution = await analyticsServiceEnhanced.performAttributionAnalysis({
        conversion_event: 'purchase',
        attribution_model: 'linear',
        lookback_window_days: 30
      });
      setAnalysisResults(prev => ({ ...prev, attribution }));
    } catch (err) {
      setError('归因分析失败');
    } finally {
      setLoading(false);
    }
  };

  const performPredictiveAnalysis = async () => {
    try {
      setLoading(true);
      const predictions = await analyticsServiceEnhanced.performPredictiveAnalysis({
        prediction_type: 'user_churn',
        prediction_horizon_days: 30,
        model_type: 'classification'
      });
      setAnalysisResults(prev => ({ ...prev, predictions }));
    } catch (err) {
      setError('预测分析失败');
    } finally {
      setLoading(false);
    }
  };

  const performAdvancedAnomalyDetection = async () => {
    try {
      setLoading(true);
      const anomalyResults = await analyticsServiceEnhanced.performAdvancedAnomalyDetection({
        detection_algorithms: ['isolation_forest', 'local_outlier_factor'],
        sensitivity: 'medium',
        time_window_hours: 24,
        include_user_behavior: true,
        include_system_metrics: true
      });
      setAnomalies(anomalyResults);
    } catch (err) {
      setError('异常检测失败');
    } finally {
      setLoading(false);
    }
  };

  const exportEventData = async (format: 'csv' | 'json' | 'xlsx') => {
    try {
      setLoading(true);
      const blob = await analyticsServiceEnhanced.exportEventData({
        format,
        start_time: '2024-01-01T00:00:00Z',
        end_time: '2024-12-31T23:59:59Z',
        limit: 10000
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `events_export.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('导出数据失败');
    } finally {
      setLoading(false);
    }
  };

  const loadSystemHealth = async () => {
    try {
      setLoading(true);
      const health = await analyticsServiceEnhanced.getSystemHealthDashboard();
      setSystemHealth(health);
    } catch (err) {
      setError('获取系统健康状态失败');
    } finally {
      setLoading(false);
    }
  };

  const performUXAnalysis = async () => {
    try {
      setLoading(true);
      const uxAnalysis = await analyticsServiceEnhanced.performUserExperienceAnalysis({
        analysis_type: 'journey_mapping',
        time_period_days: 7,
        include_heatmaps: true
      });
      setAnalysisResults(prev => ({ ...prev, ux: uxAnalysis }));
    } catch (err) {
      setError('用户体验分析失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">高级行为分析</h1>
          <p className="text-gray-600">实时监控、预测分析、深度洞察</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => exportEventData('csv')} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            导出CSV
          </Button>
          <Button onClick={() => exportEventData('xlsx')} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            导出Excel
          </Button>
        </div>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="text-red-800">{error}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="realtime" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            实时监控
          </TabsTrigger>
          <TabsTrigger value="analysis" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            深度分析
          </TabsTrigger>
          <TabsTrigger value="predictions" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            预测分析
          </TabsTrigger>
          <TabsTrigger value="anomalies" className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            异常检测
          </TabsTrigger>
          <TabsTrigger value="reports" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            报告中心
          </TabsTrigger>
          <TabsTrigger value="system" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            系统监控
          </TabsTrigger>
        </TabsList>

        <TabsContent value="realtime" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">WebSocket连接</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {realtimeStats?.active_connections || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  活跃连接数
                </p>
                <Button onClick={loadWebSocketStats} className="mt-2" size="sm">
                  刷新统计
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">响应时间</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {realtimeStats?.average_response_time_ms || 0}ms
                </div>
                <p className="text-xs text-muted-foreground">
                  平均响应时间
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">系统状态</CardTitle>
                <Badge variant={realtimeStats?.status === 'healthy' ? 'default' : 'destructive'}>
                  {realtimeStats?.status || 'unknown'}
                </Badge>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {Math.round(realtimeStats?.uptime_seconds / 3600) || 0}h
                </div>
                <p className="text-xs text-muted-foreground">
                  系统运行时间
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>实时事件流</CardTitle>
              <CardDescription>监控系统实时事件和消息</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span>发送消息数: {realtimeStats?.messages_sent || 0}</span>
                  <span>接收消息数: {realtimeStats?.messages_received || 0}</span>
                </div>
                <Progress value={realtimeStats?.uptime_seconds ? 100 : 0} className="w-full" />
                <div className="text-sm text-gray-500">
                  连接状态: {eventSource ? '已连接' : '未连接'}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>深度用户分析</CardTitle>
                <CardDescription>行为时间线、转化漏斗、留存分析</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={performDeepUserAnalysis} disabled={loading} className="w-full">
                  {loading ? '分析中...' : '开始深度分析'}
                </Button>
                {analysisResults?.user_id && (
                  <div className="space-y-2">
                    <div className="text-sm">
                      <strong>用户ID:</strong> {analysisResults.user_id}
                    </div>
                    {analysisResults.retention_analysis && (
                      <div className="space-y-1">
                        <div>1日留存: {(analysisResults.retention_analysis.day_1_retention * 100).toFixed(1)}%</div>
                        <div>7日留存: {(analysisResults.retention_analysis.day_7_retention * 100).toFixed(1)}%</div>
                        <div>30日留存: {(analysisResults.retention_analysis.day_30_retention * 100).toFixed(1)}%</div>
                        <div>流失风险评分: {analysisResults.retention_analysis.churn_risk_score}</div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>队列分析</CardTitle>
                <CardDescription>用户行为队列和留存分析</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={performCohortAnalysis} disabled={loading} className="w-full">
                  {loading ? '分析中...' : '队列分析'}
                </Button>
                {analysisResults?.cohort && (
                  <div className="space-y-2">
                    <div>队列类型: {analysisResults.cohort.cohort_type}</div>
                    <div>总队列数: {analysisResults.cohort.summary_statistics?.total_cohorts}</div>
                    <div>平均留存率: {(analysisResults.cohort.summary_statistics?.avg_retention_rate * 100).toFixed(1)}%</div>
                    <div>趋势: {analysisResults.cohort.summary_statistics?.trend_direction}</div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>归因分析</CardTitle>
                <CardDescription>转化路径和触点归因</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={performAttributionAnalysis} disabled={loading} className="w-full">
                  {loading ? '分析中...' : '归因分析'}
                </Button>
                {analysisResults?.attribution && (
                  <div className="space-y-2">
                    <div>归因模型: {analysisResults.attribution.attribution_model}</div>
                    <div>转化事件: {analysisResults.attribution.conversion_event}</div>
                    <div className="text-sm">
                      <strong>归因结果:</strong>
                      {analysisResults.attribution.attribution_results?.slice(0, 3).map((result: any, idx: number) => (
                        <div key={idx} className="ml-2">
                          {result.touchpoint_type}: {(result.attribution_credit * 100).toFixed(1)}%
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>用户体验分析</CardTitle>
                <CardDescription>用户旅程和摩擦点分析</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={performUXAnalysis} disabled={loading} className="w-full">
                  {loading ? '分析中...' : 'UX分析'}
                </Button>
                {analysisResults?.ux && (
                  <div className="space-y-2">
                    <div>分析类型: {analysisResults.ux.analysis_type}</div>
                    <div>满意度评分: {analysisResults.ux.ux_insights?.satisfaction_metrics?.overall_satisfaction_score}</div>
                    <div className="text-sm">
                      <strong>关键摩擦点:</strong>
                      {analysisResults.ux.ux_insights?.critical_friction_points?.slice(0, 2).map((point: any, idx: number) => (
                        <div key={idx} className="ml-2">
                          {point.page_or_feature}: 评分 {point.friction_score}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>预测分析</CardTitle>
              <CardDescription>用户流失、LTV预测和转化概率分析</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">预测类型</label>
                  <select className="w-full p-2 border border-gray-300 rounded" defaultValue="user_churn">
                    <option value="user_churn">用户流失</option>
                    <option value="ltv">生命周期价值</option>
                    <option value="next_action">下一步行为</option>
                    <option value="conversion_probability">转化概率</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">预测期限(天)</label>
                  <Input type="number" defaultValue="30" />
                </div>
              </div>
              
              <Button onClick={performPredictiveAnalysis} disabled={loading} className="w-full">
                {loading ? '预测中...' : '开始预测分析'}
              </Button>

              {analysisResults?.predictions && (
                <div className="space-y-4">
                  <Separator />
                  <div>
                    <h4 className="font-medium mb-2">预测结果</h4>
                    <div className="space-y-2">
                      <div>预测类型: {analysisResults.predictions.prediction_type}</div>
                      <div>模型准确率: {(analysisResults.predictions.model_info?.accuracy_score * 100).toFixed(1)}%</div>
                      <div>训练数据规模: {analysisResults.predictions.model_info?.training_data_size}</div>
                      <div>平均预测值: {analysisResults.predictions.global_insights?.avg_prediction?.toFixed(3)}</div>
                    </div>
                  </div>
                  
                  {analysisResults.predictions.predictions?.slice(0, 3).map((pred: any, idx: number) => (
                    <div key={idx} className="border rounded p-3 space-y-1">
                      <div className="font-medium">预测 #{idx + 1}</div>
                      <div>预测值: {pred.prediction_value?.toFixed(3)}</div>
                      <div>置信区间: [{pred.confidence_interval?.[0]?.toFixed(3)}, {pred.confidence_interval?.[1]?.toFixed(3)}]</div>
                      <div className="text-sm text-gray-600">建议: {pred.recommendation}</div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="anomalies" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>高级异常检测</CardTitle>
              <CardDescription>多算法异常检测和调查分析</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">敏感度</label>
                  <select className="w-full p-2 border border-gray-300 rounded" defaultValue="medium">
                    <option value="low">低</option>
                    <option value="medium">中</option>
                    <option value="high">高</option>
                    <option value="adaptive">自适应</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">时间窗口(小时)</label>
                  <Input type="number" defaultValue="24" />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">最小异常分数</label>
                  <Input type="number" step="0.1" defaultValue="0.5" />
                </div>
              </div>
              
              <Button onClick={performAdvancedAnomalyDetection} disabled={loading} className="w-full">
                {loading ? '检测中...' : '开始异常检测'}
              </Button>

              {anomalies && (
                <div className="space-y-4">
                  <Separator />
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{anomalies.detection_summary?.total_anomalies_detected || 0}</div>
                        <div className="text-sm text-gray-600">检测到异常</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{anomalies.detection_summary?.high_severity_count || 0}</div>
                        <div className="text-sm text-gray-600">高严重性异常</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-2xl font-bold">{(anomalies.detection_summary?.detection_accuracy_estimate * 100).toFixed(1)}%</div>
                        <div className="text-sm text-gray-600">检测准确率</div>
                      </CardContent>
                    </Card>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">异常详情</h4>
                    <div className="space-y-2">
                      {anomalies.anomalies?.slice(0, 5).map((anomaly: any, idx: number) => (
                        <div key={idx} className="border rounded p-3 space-y-2">
                          <div className="flex justify-between items-center">
                            <Badge variant={anomaly.severity === 'high' || anomaly.severity === 'critical' ? 'destructive' : 'secondary'}>
                              {anomaly.severity}
                            </Badge>
                            <span className="text-sm text-gray-500">{anomaly.detected_at}</span>
                          </div>
                          <div className="font-medium">{anomaly.description}</div>
                          <div className="text-sm">异常分数: {anomaly.anomaly_score?.toFixed(3)}</div>
                          <div className="text-sm">检测方法: {anomaly.detection_method}</div>
                          <div className="text-sm">误报概率: {(anomaly.false_positive_probability * 100).toFixed(1)}%</div>
                          {anomaly.recommended_actions?.length > 0 && (
                            <div className="text-sm">
                              <strong>建议操作:</strong>
                              <ul className="list-disc list-inside ml-2">
                                {anomaly.recommended_actions.slice(0, 2).map((action: string, actionIdx: number) => (
                                  <li key={actionIdx}>{action}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>高级报告生成</CardTitle>
              <CardDescription>生成综合分析报告并支持多格式导出</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">报告类型</label>
                  <select className="w-full p-2 border border-gray-300 rounded" defaultValue="comprehensive">
                    <option value="comprehensive">综合报告</option>
                    <option value="summary">摘要报告</option>
                    <option value="custom">自定义报告</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">输出格式</label>
                  <select className="w-full p-2 border border-gray-300 rounded" defaultValue="json">
                    <option value="json">JSON</option>
                    <option value="html">HTML</option>
                    <option value="pdf">PDF</option>
                  </select>
                </div>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium">报告筛选条件</label>
                <Textarea placeholder="输入JSON格式的筛选条件..." />
              </div>

              <Button onClick={generateAdvancedReport} disabled={loading} className="w-full">
                {loading ? '生成中...' : '生成高级报告'}
              </Button>

              {reportStatus && (
                <div className="space-y-4">
                  <Separator />
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">报告状态</span>
                      <Badge variant={reportStatus.status === 'completed' ? 'default' : 'secondary'}>
                        {reportStatus.status}
                      </Badge>
                    </div>
                    <div>报告ID: {reportStatus.report_id}</div>
                    <div>消息: {reportStatus.message}</div>
                    {reportStatus.estimated_completion_time && (
                      <div>预计完成时间: {reportStatus.estimated_completion_time}</div>
                    )}
                    {reportStatus.progress && (
                      <div className="space-y-1">
                        <div>完成进度: {reportStatus.progress}%</div>
                        <Progress value={reportStatus.progress} className="w-full" />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>系统健康监控</CardTitle>
              <CardDescription>监控分析系统各组件状态和性能指标</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={loadSystemHealth} disabled={loading} className="w-full">
                {loading ? '加载中...' : '刷新系统状态'}
              </Button>

              {systemHealth && (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">整体健康状态</span>
                    <Badge variant={systemHealth.overall_health === 'healthy' ? 'default' : 'destructive'}>
                      {systemHealth.overall_health}
                    </Badge>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">组件状态</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {Object.entries(systemHealth.component_status || {}).map(([component, status]) => (
                        <div key={component} className="flex justify-between items-center p-2 border rounded">
                          <span className="text-sm">{component}</span>
                          <Badge variant={status === 'operational' ? 'default' : 'destructive'}>
                            {status as string}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">性能指标</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div>每小时处理事件: {systemHealth.performance_metrics?.events_processed_last_hour || 0}</div>
                        <div>平均处理延迟: {systemHealth.performance_metrics?.avg_processing_latency_ms || 0}ms</div>
                        <div>内存使用率: {systemHealth.performance_metrics?.memory_usage_percentage || 0}%</div>
                      </div>
                      <div className="space-y-2">
                        <div>CPU使用率: {systemHealth.performance_metrics?.cpu_usage_percentage || 0}%</div>
                        <div>磁盘使用率: {systemHealth.performance_metrics?.disk_usage_percentage || 0}%</div>
                        <div>系统运行时间: {systemHealth.system_statistics?.uptime_hours || 0}小时</div>
                      </div>
                    </div>
                  </div>

                  {systemHealth.active_alerts?.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-2">活跃告警</h4>
                      <div className="space-y-2">
                        {systemHealth.active_alerts.slice(0, 5).map((alert: any, idx: number) => (
                          <div key={idx} className="border rounded p-3 space-y-1">
                            <div className="flex justify-between items-center">
                              <Badge variant={alert.severity === 'critical' || alert.severity === 'error' ? 'destructive' : 'secondary'}>
                                {alert.severity}
                              </Badge>
                              <span className="text-sm text-gray-500">{alert.created_at}</span>
                            </div>
                            <div className="text-sm font-medium">{alert.component}</div>
                            <div className="text-sm">{alert.message}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BehaviorAnalyticsPageEnhanced;
