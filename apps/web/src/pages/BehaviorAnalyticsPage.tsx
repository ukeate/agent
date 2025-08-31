import React, { useState, useEffect, useCallback } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert } from '../components/ui/alert';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/Tabs';
import { Progress } from '../components/ui/progress';
import { Badge } from '../components/ui/badge';
import { BehaviorOverview } from '../components/analytics/BehaviorOverview';
import { PatternAnalysis } from '../components/analytics/PatternAnalysis';
import { AnomalyDetection } from '../components/analytics/AnomalyDetection';
import { TrendAnalysis } from '../components/analytics/TrendAnalysis';
import { RealTimeMonitor } from '../components/analytics/RealTimeMonitor';
import { PerformanceMetrics } from '../components/analytics/PerformanceMetrics';
import { behaviorAnalyticsService } from '../services/behaviorAnalyticsService';

interface AnalyticsData {
  overview: any;
  patterns: any[];
  anomalies: any[];
  trends: any;
  realtime: any;
  performance: any;
}

const BehaviorAnalyticsPage: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshing, setRefreshing] = useState(false);
  const [websocketConnected, setWebsocketConnected] = useState(false);

  // WebSocket连接状态（使用模拟数据）
  useEffect(() => {
    // 模拟WebSocket连接
    setWebsocketConnected(true);
    
    // 模拟实时数据更新
    const interval = setInterval(() => {
      if (data) {
        setData(prevData => ({
          ...prevData!,
          realtime: {
            active_users: Math.floor(Math.random() * 50) + 100,
            events_per_second: Math.floor(Math.random() * 20) + 10,
            response_time_ms: Math.floor(Math.random() * 50) + 20,
            error_rate: Math.random() * 0.05
          }
        }));
      }
    }, 3000);

    return () => {
      clearInterval(interval);
    };
  }, [data]);

  // 模拟消息处理
  const handleMockMessage = (event: any) => {
    const message = JSON.parse(event.data);
    if (message.type === 'stats') {
      setData(prev => prev ? {
        ...prev,
        realtime: message.data
      } : null);
    }
  };

  // 加载分析数据（使用真实API和模拟数据混合）
  const loadAnalyticsData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // 调用真实的异常检测API
      let realAnomalies = [];
      try {
        const anomaliesResponse = await behaviorAnalyticsService.getAnomalies({ 
          limit: 20,
          use_real_detection: true 
        });
        if (anomaliesResponse.success && anomaliesResponse.anomalies) {
          realAnomalies = anomaliesResponse.anomalies.map((anomaly: any) => ({
            anomaly_id: anomaly.anomaly_id,
            user_id: anomaly.user_id,
            event_type: anomaly.event_type,
            timestamp: anomaly.timestamp,
            severity: anomaly.severity,
            confidence: anomaly.confidence,
            description: anomaly.description,
            anomaly_type: anomaly.anomaly_type,
            detected_by: ["IntelligentAnomalyDetector"],
            context: { score: anomaly.score },
            resolved: false
          }));
        }
      } catch (apiError) {
        console.warn('真实异常检测API调用失败，使用模拟数据:', apiError);
      }

      // 如果没有获取到真实数据，使用模拟数据
      const fallbackAnomalies = [
        {
          anomaly_id: "anomaly_1",
          user_id: "user_123",
          event_type: "click",
          timestamp: new Date().toISOString(),
          severity: "medium",
          confidence: 0.85,
          description: "用户在短时间内异常高频点击同一元素",
          anomaly_type: "behavioral",
          detected_by: ["statistical_analysis", "pattern_detection"],
          context: { click_count: 45, duration_seconds: 30, element: "product_button" },
          resolved: false
        },
        {
          anomaly_id: "anomaly_2", 
          user_id: "user_456",
          event_type: "navigation",
          timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
          severity: "high",
          confidence: 0.92,
          description: "异常的页面跳转模式，疑似机器人行为",
          anomaly_type: "navigation",
          detected_by: ["sequence_analysis"],
          context: { pages_per_second: 3.2, bounce_rate: 0.05 },
          resolved: false
        },
        {
          anomaly_id: "anomaly_3",
          user_id: "user_789", 
          event_type: "purchase",
          timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
          severity: "critical",
          confidence: 0.94,
          description: "异常大额订单，与用户历史行为不符",
          anomaly_type: "transaction",
          detected_by: ["value_analysis", "user_profile"],
          context: { order_amount: 5000, avg_order_amount: 150, deviation: 33.33 },
          resolved: true
        }
      ];

      // 模拟数据
      const mockData = {
        overview: {
          total_events: 25430,
          unique_users: 3420,
          unique_sessions: 1250,
          events_per_minute: 42.5,
          event_type_distribution: {
            'page_view': 12340,
            'click': 8920,
            'purchase': 340,
            'search': 2890,
            'login': 1940
          },
          hourly_distribution: {
            '0': 120, '1': 145, '2': 189, '3': 203, '4': 187, '5': 165, '6': 142,
            '7': 198, '8': 256, '9': 289, '10': 245, '11': 223, '12': 267,
            '13': 234, '14': 198, '15': 176, '16': 145, '17': 198, '18': 234,
            '19': 245, '20': 198, '21': 167, '22': 134, '23': 98
          },
          most_active_hour: 9
        },
        patterns: [
          {
            pattern_id: "pattern_1",
            sequence: ["login", "browse_products", "add_to_cart"],
            description: "用户登录 → 浏览商品 → 加入购物车序列",
            support: 0.85,
            confidence: 0.72,
            frequency: 156,
            user_count: 342
          },
          {
            pattern_id: "pattern_2", 
            sequence: ["click_category", "filter_products", "view_details"],
            description: "高频点击特定商品类别",
            support: 0.73,
            confidence: 0.68,
            frequency: 128,
            user_count: 298
          },
          {
            pattern_id: "pattern_3",
            sequence: ["search", "view_results", "click_product", "purchase"],
            description: "搜索 → 查看结果 → 点击商品 → 购买流程",
            support: 0.56,
            confidence: 0.84,
            frequency: 89,
            user_count: 156
          }
        ],
        anomalies: realAnomalies.length > 0 ? realAnomalies : fallbackAnomalies,
        trends: {
          correlation_analysis: [
            {
              dimension_pairs: ["page_views", "user_engagement"],
              correlation_type: "pearson",
              correlation_value: 0.85,
              p_value: 0.0021,
              significance_level: "high"
            },
            {
              dimension_pairs: ["session_duration", "bounce_rate"],
              correlation_type: "pearson", 
              correlation_value: -0.72,
              p_value: 0.0089,
              significance_level: "high"
            },
            {
              dimension_pairs: ["click_frequency", "conversion_rate"],
              correlation_type: "spearman",
              correlation_value: 0.68,
              p_value: 0.0156,
              significance_level: "medium"
            }
          ],
          cluster_analysis: {
            cluster_labels: [0, 1, 0, 2, 1, 0, 2, 1],
            cluster_centers: {
              0: [2.5, 1.8, 3.2, 0.9],
              1: [1.2, 3.5, 2.1, 1.7],
              2: [3.8, 0.8, 1.5, 2.4]
            },
            pca_coordinates: [[1.2, 0.8], [2.1, -0.5], [0.9, 1.3]],
            explained_variance: [0.65, 0.23, 0.08, 0.04],
            feature_importance: {
              "session_duration": 0.89,
              "page_views": 0.76,
              "click_frequency": 0.68,
              "bounce_rate": 0.55,
              "time_on_page": 0.42,
              "scroll_depth": 0.38,
              "conversion_events": 0.31
            }
          },
          behavioral_patterns: {
            user_patterns: {
              "pattern_1": { frequency: 156, users: 342 },
              "pattern_2": { frequency: 128, users: 298 }
            },
            pattern_similarities: {
              "user_123-user_456": 0.87,
              "user_789-user_123": 0.72,
              "user_456-user_789": 0.65,
              "user_234-user_567": 0.58,
              "user_890-user_345": 0.51
            },
            dominant_patterns: {
              most_common_events: [
                ["login", 342],
                ["browse_products", 298],
                ["add_to_cart", 186],
                ["search", 145],
                ["checkout", 89],
                ["view_profile", 78],
                ["share_content", 56]
              ],
              user_count: 1250,
              unique_event_types: 15
            }
          },
          actionable_recommendations: [
            {
              type: "optimization",
              priority: "high",
              recommendation: "优化页面加载速度，减少首屏加载时间至2秒以内",
              impact: "预计可提升用户留存率15-20%，减少跳出率8-12%"
            },
            {
              type: "engagement",
              priority: "medium", 
              recommendation: "在用户浏览商品页面时添加相关推荐功能",
              impact: "预计可增加用户互动时长25%，提升转化率5-8%"
            },
            {
              type: "personalization",
              priority: "low",
              recommendation: "基于用户行为模式个性化首页内容布局",
              impact: "预计可提升用户满意度10-15%，增加回访率3-5%"
            }
          ]
        },
        realtime: {
          event_count: 1847,
          active_users: 127,
          events_per_minute: 15.2,
          window_duration_seconds: 300,
          unique_sessions: 89,
          event_type_distribution: {
            'click': 450,
            'page_view': 380,
            'scroll': 320,
            'form_submit': 285,
            'hover': 412
          },
          response_time_ms: 35,
          error_rate: 0.012
        },
        performance: {
          api_response_time: 45,
          database_queries: 234,
          cache_hit_rate: 0.85,
          throughput: 1250
        }
      };

      setData(mockData);
    } catch (err) {
      setError(err instanceof Error ? err.message : '数据加载失败');
    } finally {
      setLoading(false);
    }
  }, []);

  // 刷新数据
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadAnalyticsData();
    setRefreshing(false);
  };

  // 导出数据
  const handleExport = async (format: string) => {
    try {
      const blob = await behaviorAnalyticsService.exportEvents(format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `behavior_analytics.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('导出失败: ' + (err instanceof Error ? err.message : '未知错误'));
    }
  };

  // 初始加载数据
  useEffect(() => {
    loadAnalyticsData();
  }, [loadAnalyticsData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-500"></div>
          <p className="mt-4 text-gray-600">加载行为分析数据...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <Alert variant="destructive" className="mb-4">
          <h3 className="font-bold">加载失败</h3>
          <p>{error}</p>
        </Alert>
        <Button onClick={loadAnalyticsData}>重试</Button>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="p-8">
        <Alert variant="warning">
          <p>暂无分析数据</p>
        </Alert>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: '概览', icon: '📊' },
    { id: 'patterns', label: '行为模式', icon: '🔍' },
    { id: 'anomalies', label: '异常检测', icon: '⚠️' },
    { id: 'trends', label: '趋势分析', icon: '📈' },
    { id: 'realtime', label: '实时监控', icon: '⚡' },
    { id: 'performance', label: '性能指标', icon: '🚀' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 页头 */}
      <div className="bg-white shadow-sm border-b">
        <div className="px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                用户行为分析仪表板
              </h1>
              <p className="text-gray-600 mt-1">
                实时监控用户行为，智能识别模式和异常
              </p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* 连接状态 */}
              <Badge 
                variant={websocketConnected ? 'success' : 'danger'}
                className="flex items-center space-x-1"
              >
                <div className={`w-2 h-2 rounded-full ${
                  websocketConnected ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <span>{websocketConnected ? '实时连接' : '连接断开'}</span>
              </Badge>

              {/* 操作按钮 */}
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefresh}
                  disabled={refreshing}
                  className="flex items-center space-x-2"
                >
                  <span className={refreshing ? 'animate-spin' : ''}>🔄</span>
                  <span>刷新</span>
                </Button>
                
                <div className="relative group">
                  <Button variant="outline" size="sm">
                    📥 导出
                  </Button>
                  <div className="absolute right-0 mt-1 w-32 bg-white rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                    <div className="py-1">
                      <button
                        onClick={() => handleExport('json')}
                        className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                      >
                        JSON格式
                      </button>
                      <button
                        onClick={() => handleExport('csv')}
                        className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                      >
                        CSV格式
                      </button>
                      <button
                        onClick={() => handleExport('xlsx')}
                        className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 w-full text-left"
                      >
                        Excel格式
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 主内容 */}
      <div className="px-8 py-6">
        {/* 标签页 */}
        <div className="mb-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              {tabs.map(tab => (
                <TabsTrigger key={tab.id} value={tab.id}>
                  {tab.icon} {tab.label}
                </TabsTrigger>
              ))}
            </TabsList>
            
            <TabsContent value="overview">
              <BehaviorOverview 
                data={data.overview}
                realtime={data.realtime}
              />
            </TabsContent>
            
            <TabsContent value="patterns">
              <PatternAnalysis patterns={data.patterns} />
            </TabsContent>
            
            <TabsContent value="anomalies">
              <AnomalyDetection anomalies={data.anomalies} />
            </TabsContent>
            
            <TabsContent value="trends">
              <TrendAnalysis trends={data.trends} />
            </TabsContent>
            
            <TabsContent value="realtime">
              <RealTimeMonitor 
                data={data.realtime}
                connected={websocketConnected}
              />
            </TabsContent>
            
            <TabsContent value="performance">
              <PerformanceMetrics metrics={data.performance} />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default BehaviorAnalyticsPage;