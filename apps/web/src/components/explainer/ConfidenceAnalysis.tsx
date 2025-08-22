import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/Progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle,
  Target,
  Gauge,
  Info,
  Zap,
  Shield
} from 'lucide-react';

interface ConfidenceSource {
  source: string;
  score: number;
  weight: number;
  description: string;
  reliability: number;
}

interface ConfidenceMetrics {
  overall_confidence: number;
  prediction_confidence?: number;
  evidence_confidence?: number;
  model_confidence?: number;
  uncertainty_score: number;
  variance?: number;
  confidence_interval_lower?: number;
  confidence_interval_upper?: number;
  confidence_sources: ConfidenceSource[];
  calibration_score?: number;
}

interface UncertaintyFactors {
  data_quality: number;
  model_complexity: number;
  feature_reliability: number;
  temporal_distance: number;
  context_similarity: number;
  sample_size: number;
}

interface ConfidenceAnalysisProps {
  metrics: ConfidenceMetrics;
  uncertaintyFactors?: UncertaintyFactors;
  className?: string;
}

const ConfidenceAnalysis: React.FC<ConfidenceAnalysisProps> = ({ 
  metrics, 
  uncertaintyFactors,
  className = '' 
}) => {
  const [selectedView, setSelectedView] = useState<'overview' | 'breakdown' | 'uncertainty' | 'calibration'>('overview');

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.9) return { level: '极高', color: 'text-emerald-600', bgColor: 'bg-emerald-100' };
    if (confidence >= 0.8) return { level: '高', color: 'text-green-600', bgColor: 'bg-green-100' };
    if (confidence >= 0.7) return { level: '中等偏高', color: 'text-blue-600', bgColor: 'bg-blue-100' };
    if (confidence >= 0.6) return { level: '中等', color: 'text-yellow-600', bgColor: 'bg-yellow-100' };
    if (confidence >= 0.5) return { level: '中等偏低', color: 'text-orange-600', bgColor: 'bg-orange-100' };
    return { level: '低', color: 'text-red-600', bgColor: 'bg-red-100' };
  };

  const getUncertaintyLevel = (uncertainty: number) => {
    if (uncertainty <= 0.1) return { level: '极低', color: 'text-emerald-600' };
    if (uncertainty <= 0.2) return { level: '低', color: 'text-green-600' };
    if (uncertainty <= 0.3) return { level: '中等', color: 'text-yellow-600' };
    if (uncertainty <= 0.4) return { level: '中等偏高', color: 'text-orange-600' };
    return { level: '高', color: 'text-red-600' };
  };

  const confidenceLevel = getConfidenceLevel(metrics.overall_confidence);
  const uncertaintyLevel = getUncertaintyLevel(metrics.uncertainty_score);

  // 模拟不确定性因子数据
  const defaultUncertaintyFactors: UncertaintyFactors = uncertaintyFactors || {
    data_quality: 0.15,
    model_complexity: 0.20,
    feature_reliability: 0.12,
    temporal_distance: 0.08,
    context_similarity: 0.10,
    sample_size: 0.18
  };

  const factorLabels = {
    data_quality: '数据质量',
    model_complexity: '模型复杂度',
    feature_reliability: '特征可靠性',
    temporal_distance: '时间距离',
    context_similarity: '上下文相似性',
    sample_size: '样本大小'
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* 置信度概览 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Gauge className="h-5 w-5 text-blue-600" />
            <span>置信度分析</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* 总体置信度 */}
            <div className="text-center">
              <div className="relative w-24 h-24 mx-auto mb-3">
                <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 36 36">
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#E5E7EB"
                    strokeWidth="2"
                  />
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke={confidenceLevel.color.replace('text-', '#')}
                    strokeWidth="2"
                    strokeDasharray={`${metrics.overall_confidence * 100}, 100`}
                    className="transition-all duration-500"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className={`text-xl font-bold ${confidenceLevel.color}`}>
                    {(metrics.overall_confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600">总体置信度</div>
                <Badge className={`${confidenceLevel.bgColor} ${confidenceLevel.color} mt-1`}>
                  {confidenceLevel.level}
                </Badge>
              </div>
            </div>

            {/* 不确定性 */}
            <div className="text-center">
              <div className="relative w-24 h-24 mx-auto mb-3">
                <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 36 36">
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#E5E7EB"
                    strokeWidth="2"
                  />
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke={uncertaintyLevel.color.replace('text-', '#')}
                    strokeWidth="2"
                    strokeDasharray={`${metrics.uncertainty_score * 100}, 100`}
                    className="transition-all duration-500"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className={`text-xl font-bold ${uncertaintyLevel.color}`}>
                    {(metrics.uncertainty_score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600">不确定性</div>
                <Badge className={`${uncertaintyLevel.color} mt-1`} variant="outline">
                  {uncertaintyLevel.level}
                </Badge>
              </div>
            </div>

            {/* 置信区间 */}
            <div className="text-center">
              <div className="mb-3">
                <div className="text-2xl font-bold text-gray-700">
                  [{((metrics.confidence_interval_lower || 0) * 100).toFixed(0)}%, {((metrics.confidence_interval_upper || 1) * 100).toFixed(0)}%]
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-600">95%置信区间</div>
                <div className="mt-2">
                  <Progress 
                    value={metrics.overall_confidence * 100} 
                    className="h-2"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>{((metrics.confidence_interval_lower || 0) * 100).toFixed(1)}%</span>
                    <span>{((metrics.confidence_interval_upper || 1) * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 详细分析 */}
      <Tabs value={selectedView} onValueChange={(value: any) => setSelectedView(value)}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" className="flex items-center">
            <Target className="h-4 w-4 mr-2" />
            概览
          </TabsTrigger>
          <TabsTrigger value="breakdown" className="flex items-center">
            <BarChart3 className="h-4 w-4 mr-2" />
            置信度分解
          </TabsTrigger>
          <TabsTrigger value="uncertainty" className="flex items-center">
            <AlertTriangle className="h-4 w-4 mr-2" />
            不确定性分析
          </TabsTrigger>
          <TabsTrigger value="calibration" className="flex items-center">
            <Zap className="h-4 w-4 mr-2" />
            校准分析
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>置信度解读</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className={`p-4 rounded-lg ${confidenceLevel.bgColor}`}>
                  <div className="flex items-center space-x-2 mb-2">
                    {metrics.overall_confidence >= 0.7 ? (
                      <CheckCircle className={`h-5 w-5 ${confidenceLevel.color}`} />
                    ) : (
                      <AlertTriangle className={`h-5 w-5 ${confidenceLevel.color}`} />
                    )}
                    <span className={`font-semibold ${confidenceLevel.color}`}>
                      {confidenceLevel.level}置信度 ({(metrics.overall_confidence * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <p className="text-sm text-gray-700">
                    {metrics.overall_confidence >= 0.9 && "决策具有极高的可信度，可以安全采用。"}
                    {metrics.overall_confidence >= 0.8 && metrics.overall_confidence < 0.9 && "决策具有高可信度，建议采用并保持监控。"}
                    {metrics.overall_confidence >= 0.7 && metrics.overall_confidence < 0.8 && "决策可信度良好，建议谨慎采用并收集更多证据。"}
                    {metrics.overall_confidence >= 0.6 && metrics.overall_confidence < 0.7 && "决策可信度中等，建议进一步验证或获取专家意见。"}
                    {metrics.overall_confidence < 0.6 && "决策可信度较低，建议重新评估或收集更多高质量数据。"}
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">关键指标</h4>
                    <div className="space-y-2 text-sm">
                      {metrics.prediction_confidence && (
                        <div className="flex justify-between">
                          <span>预测置信度:</span>
                          <span className="font-medium">{(metrics.prediction_confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {metrics.evidence_confidence && (
                        <div className="flex justify-between">
                          <span>证据置信度:</span>
                          <span className="font-medium">{(metrics.evidence_confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {metrics.model_confidence && (
                        <div className="flex justify-between">
                          <span>模型置信度:</span>
                          <span className="font-medium">{(metrics.model_confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {metrics.variance && (
                        <div className="flex justify-between">
                          <span>方差:</span>
                          <span className="font-medium">{metrics.variance.toFixed(4)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">风险评估</h4>
                    <div className="space-y-2">
                      {metrics.uncertainty_score <= 0.2 && (
                        <div className="flex items-center space-x-2 text-green-700">
                          <Shield className="h-4 w-4" />
                          <span className="text-sm">低风险决策</span>
                        </div>
                      )}
                      {metrics.uncertainty_score > 0.2 && metrics.uncertainty_score <= 0.4 && (
                        <div className="flex items-center space-x-2 text-yellow-700">
                          <Info className="h-4 w-4" />
                          <span className="text-sm">中等风险，需要监控</span>
                        </div>
                      )}
                      {metrics.uncertainty_score > 0.4 && (
                        <div className="flex items-center space-x-2 text-red-700">
                          <AlertTriangle className="h-4 w-4" />
                          <span className="text-sm">高风险，需要谨慎处理</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="breakdown" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>置信度来源分解</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {metrics.confidence_sources.map((source, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <h4 className="font-semibold">{source.source}</h4>
                        <p className="text-sm text-gray-600">{source.description}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-semibold">{(source.score * 100).toFixed(1)}%</div>
                        <div className="text-xs text-gray-500">权重: {source.weight}</div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Progress value={source.score * 100} className="h-2" />
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>可靠性: {(source.reliability * 100).toFixed(0)}%</span>
                        <span>加权贡献: {(source.score * source.weight * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="uncertainty" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>不确定性因子分析</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(defaultUncertaintyFactors).map(([factor, value]) => (
                  <div key={factor} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold">{factorLabels[factor as keyof typeof factorLabels]}</h4>
                      <span className={`font-semibold ${getUncertaintyLevel(value).color}`}>
                        {(value * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={value * 100} className="h-2 mb-2" />
                    <div className="text-xs text-gray-600">
                      {factor === 'data_quality' && '输入数据的质量和完整性影响'}
                      {factor === 'model_complexity' && '模型复杂度带来的不确定性'}
                      {factor === 'feature_reliability' && '特征变量的可靠性评估'}
                      {factor === 'temporal_distance' && '时间距离对预测的影响'}
                      {factor === 'context_similarity' && '上下文匹配度的影响'}
                      {factor === 'sample_size' && '样本数量对结果的影响'}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="calibration" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>置信度校准分析</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {metrics.calibration_score && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium mb-2">校准分数</h4>
                      <div className={`text-2xl font-bold ${getConfidenceLevel(metrics.calibration_score).color}`}>
                        {(metrics.calibration_score * 100).toFixed(1)}%
                      </div>
                      <Progress value={metrics.calibration_score * 100} className="h-2 mt-2" />
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">校准质量</h4>
                      <div className="space-y-2 text-sm">
                        {metrics.calibration_score > 0.9 && (
                          <div className="flex items-center space-x-2 text-green-700">
                            <CheckCircle className="h-4 w-4" />
                            <span>校准质量优秀</span>
                          </div>
                        )}
                        {metrics.calibration_score > 0.7 && metrics.calibration_score <= 0.9 && (
                          <div className="flex items-center space-x-2 text-blue-700">
                            <Info className="h-4 w-4" />
                            <span>校准质量良好</span>
                          </div>
                        )}
                        {metrics.calibration_score <= 0.7 && (
                          <div className="flex items-center space-x-2 text-orange-700">
                            <AlertTriangle className="h-4 w-4" />
                            <span>需要改进校准</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium mb-2 text-blue-800">校准建议</h4>
                  <ul className="text-sm text-blue-700 space-y-1">
                    <li>• 收集更多历史决策数据以改进校准</li>
                    <li>• 定期评估预测准确性与置信度的匹配程度</li>
                    <li>• 考虑不同场景下的校准表现差异</li>
                    <li>• 建立置信度校准的反馈机制</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ConfidenceAnalysis;