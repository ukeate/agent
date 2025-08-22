import React, { useState } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Progress } from '../ui/Progress';

interface TrendAnalysisProps {
  trends: {
    dimension_metrics?: Record<string, any>;
    correlation_analysis?: Array<{
      dimension_pairs: [string, string];
      correlation_type: string;
      correlation_value: number;
      p_value: number;
      significance_level: string;
    }>;
    cluster_analysis?: {
      cluster_labels: number[];
      cluster_centers: Record<number, number[]>;
      pca_coordinates: number[][];
      explained_variance: number[];
      feature_importance: Record<string, number>;
    };
    behavioral_patterns?: {
      user_patterns: Record<string, any>;
      pattern_similarities: Record<string, number>;
      dominant_patterns: {
        most_common_events: Array<[string, number]>;
        user_count: number;
        unique_event_types: number;
      };
    };
    actionable_recommendations?: Array<{
      type: string;
      priority: string;
      recommendation: string;
      impact: string;
    }>;
  };
}

export const TrendAnalysis: React.FC<TrendAnalysisProps> = ({ trends }) => {
  const [activeTab, setActiveTab] = useState<'correlations' | 'clusters' | 'patterns' | 'recommendations'>('correlations');

  // 获取显著性水平的颜色
  const getSignificanceColor = (level: string) => {
    const colors = {
      'high': 'text-green-600 bg-green-100',
      'medium': 'text-yellow-600 bg-yellow-100',
      'low': 'text-orange-600 bg-orange-100',
      'not_significant': 'text-gray-600 bg-gray-100'
    };
    return colors[level as keyof typeof colors] || colors.not_significant;
  };

  // 获取优先级颜色
  const getPriorityColor = (priority: string) => {
    const colors = {
      'high': 'text-red-600 bg-red-100',
      'medium': 'text-yellow-600 bg-yellow-100',
      'low': 'text-blue-600 bg-blue-100'
    };
    return colors[priority as keyof typeof colors] || colors.low;
  };

  // 渲染相关性分析
  const renderCorrelationAnalysis = () => {
    if (!trends.correlation_analysis || trends.correlation_analysis.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-4 block">📊</span>
          <p>暂无相关性分析数据</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {trends.correlation_analysis.map((correlation, index) => (
          <Card key={index} className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <h5 className="font-medium text-gray-900 mb-2">
                  {correlation.dimension_pairs[0]} ↔ {correlation.dimension_pairs[1]}
                </h5>
                <div className="flex items-center space-x-2 mb-2">
                  <Badge className="text-xs bg-blue-100 text-blue-800">
                    {correlation.correlation_type}
                  </Badge>
                  <Badge className={`text-xs ${getSignificanceColor(correlation.significance_level)}`}>
                    {correlation.significance_level}
                  </Badge>
                </div>
                <p className="text-sm text-gray-600">
                  P值: {correlation.p_value.toFixed(4)}
                </p>
              </div>
              
              <div className="text-right ml-4">
                <p className="text-2xl font-bold text-gray-900">
                  {correlation.correlation_value.toFixed(3)}
                </p>
                <div className="w-24 mt-2">
                  <Progress
                    value={Math.abs(correlation.correlation_value) * 100}
                    max={100}
                    className="h-2"
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {correlation.correlation_value > 0 ? '正相关' : '负相关'}
                </p>
              </div>
            </div>
          </Card>
        ))}
      </div>
    );
  };

  // 渲染聚类分析
  const renderClusterAnalysis = () => {
    if (!trends.cluster_analysis) {
      return (
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-4 block">🎯</span>
          <p>暂无聚类分析数据</p>
        </div>
      );
    }

    const { cluster_centers, explained_variance, feature_importance } = trends.cluster_analysis;

    return (
      <div className="space-y-6">
        {/* 聚类概览 */}
        <Card className="p-6">
          <h5 className="font-medium text-gray-900 mb-4">聚类概览</h5>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-md">
              <p className="text-2xl font-bold text-blue-600">
                {Object.keys(cluster_centers).length}
              </p>
              <p className="text-sm text-blue-800">聚类数量</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-md">
              <p className="text-2xl font-bold text-green-600">
                {(explained_variance[0] * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-green-800">第一主成分解释方差</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-md">
              <p className="text-2xl font-bold text-purple-600">
                {((explained_variance[0] + explained_variance[1]) * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-purple-800">前两个主成分解释方差</p>
            </div>
          </div>
        </Card>

        {/* 特征重要性 */}
        <Card className="p-6">
          <h5 className="font-medium text-gray-900 mb-4">特征重要性</h5>
          <div className="space-y-3">
            {Object.entries(feature_importance)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 10)
              .map(([feature, importance]) => (
                <div key={feature} className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">{feature}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32">
                      <Progress
                        value={importance * 100}
                        max={100}
                        className="h-2"
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-12">
                      {(importance * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </Card>

        {/* 聚类中心 */}
        <Card className="p-6">
          <h5 className="font-medium text-gray-900 mb-4">聚类中心</h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(cluster_centers).map(([clusterId, center]) => (
              <div key={clusterId} className="bg-gray-50 p-4 rounded-md">
                <h6 className="font-medium text-gray-900 mb-2">
                  聚类 {clusterId}
                </h6>
                <div className="space-y-2">
                  {center.slice(0, 5).map((value, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">维度 {index + 1}</span>
                      <span className="text-sm font-medium text-gray-900">
                        {value.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    );
  };

  // 渲染行为模式分析
  const renderPatternAnalysis = () => {
    if (!trends.behavioral_patterns) {
      return (
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-4 block">🔄</span>
          <p>暂无行为模式数据</p>
        </div>
      );
    }

    const { dominant_patterns, pattern_similarities } = trends.behavioral_patterns;

    return (
      <div className="space-y-6">
        {/* 主导模式统计 */}
        <Card className="p-6">
          <h5 className="font-medium text-gray-900 mb-4">主导模式统计</h5>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded-md">
              <p className="text-2xl font-bold text-blue-600">
                {dominant_patterns.user_count}
              </p>
              <p className="text-sm text-blue-800">分析用户数</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-md">
              <p className="text-2xl font-bold text-green-600">
                {dominant_patterns.unique_event_types}
              </p>
              <p className="text-sm text-green-800">事件类型数</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-md">
              <p className="text-2xl font-bold text-purple-600">
                {dominant_patterns.most_common_events.length}
              </p>
              <p className="text-sm text-purple-800">主要行为模式</p>
            </div>
          </div>

          {/* 最常见事件 */}
          <div>
            <h6 className="font-medium text-gray-700 mb-3">最常见事件</h6>
            <div className="space-y-2">
              {dominant_patterns.most_common_events.slice(0, 10).map(([event, count], index) => (
                <div key={event} className="flex items-center justify-between p-3 bg-gray-50 rounded-md">
                  <div className="flex items-center space-x-3">
                    <Badge variant={index < 3 ? 'default' : 'secondary'}>
                      #{index + 1}
                    </Badge>
                    <span className="text-sm font-medium text-gray-900">{event}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-bold text-gray-900">{count}</span>
                    <span className="text-xs text-gray-500 ml-1">用户</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* 用户模式相似性 */}
        <Card className="p-6">
          <h5 className="font-medium text-gray-900 mb-4">用户模式相似性</h5>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {Object.entries(pattern_similarities)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 20)
              .map(([userPair, similarity]) => (
                <div key={userPair} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <span className="text-sm text-gray-700">{userPair}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20">
                      <Progress
                        value={similarity * 100}
                        max={100}
                        className="h-2"
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-12">
                      {(similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </Card>
      </div>
    );
  };

  // 渲染建议
  const renderRecommendations = () => {
    if (!trends.actionable_recommendations || trends.actionable_recommendations.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">
          <span className="text-4xl mb-4 block">💡</span>
          <p>暂无优化建议</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {trends.actionable_recommendations.map((recommendation, index) => (
          <Card key={index} className="p-6">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getPriorityColor(recommendation.priority)}`}>
                  {recommendation.priority}
                </div>
              </div>
              
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  <Badge className="text-xs bg-blue-100 text-blue-800">
                    {recommendation.type}
                  </Badge>
                </div>
                
                <h5 className="font-medium text-gray-900 mb-2">
                  {recommendation.recommendation}
                </h5>
                
                <p className="text-sm text-gray-600">
                  <span className="font-medium">预期影响：</span>
                  {recommendation.impact}
                </p>
              </div>
              
              <div className="flex-shrink-0">
                <span className="text-2xl">
                  {recommendation.priority === 'high' ? '🔥' : 
                   recommendation.priority === 'medium' ? '⚡' : '💡'}
                </span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    );
  };

  const tabs = [
    { id: 'correlations', label: '相关性分析', icon: '📊' },
    { id: 'clusters', label: '聚类分析', icon: '🎯' },
    { id: 'patterns', label: '模式分析', icon: '🔄' },
    { id: 'recommendations', label: '优化建议', icon: '💡' }
  ];

  return (
    <div className="space-y-6">
      {/* 标签导航 */}
      <Card className="p-4">
        <div className="flex space-x-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </Card>

      {/* 内容区域 */}
      <div>
        {activeTab === 'correlations' && renderCorrelationAnalysis()}
        {activeTab === 'clusters' && renderClusterAnalysis()}
        {activeTab === 'patterns' && renderPatternAnalysis()}
        {activeTab === 'recommendations' && renderRecommendations()}
      </div>
    </div>
  );
};