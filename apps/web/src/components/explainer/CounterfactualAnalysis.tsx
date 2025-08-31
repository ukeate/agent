import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { 
  HelpCircle, 
  ArrowRight, 
  ArrowDown, 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Zap,
  Eye,
  Settings,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';

interface CounterfactualScenario {
  scenario_name: string;
  scenario_id: string;
  changed_factors: Record<string, any>;
  predicted_outcome: string;
  probability: number;
  impact_difference: number;
  confidence_change: number;
  explanation: string;
  risk_level: 'low' | 'medium' | 'high';
  metadata: {
    computation_time?: number;
    model_confidence?: number;
    sensitivity_score?: number;
    plausibility?: number;
  };
}

interface FactorSensitivity {
  factor_name: string;
  original_value: any;
  sensitivity_range: [number, number];
  impact_curve: Array<{value: number, impact: number, probability: number}>;
  critical_threshold?: number;
  elasticity: number;
}

interface CounterfactualAnalysisProps {
  scenarios: CounterfactualScenario[];
  factorSensitivities?: FactorSensitivity[];
  baselineOutcome: string;
  baselineConfidence: number;
  className?: string;
}

const CounterfactualAnalysis: React.FC<CounterfactualAnalysisProps> = ({
  scenarios,
  factorSensitivities = [],
  baselineOutcome,
  baselineConfidence,
  className = ''
}) => {
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [animationEnabled, setAnimationEnabled] = useState(true);
  const [selectedView, setSelectedView] = useState<'scenarios' | 'sensitivity' | 'comparison' | 'simulation'>('scenarios');

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return 'text-green-600 bg-green-100 border-green-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 border-yellow-200';
      case 'high':
        return 'text-red-600 bg-red-100 border-red-200';
      default:
        return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  const getImpactIcon = (impact: number) => {
    if (impact > 0.1) return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (impact < -0.1) return <TrendingDown className="h-4 w-4 text-red-600" />;
    return <ArrowRight className="h-4 w-4 text-gray-600" />;
  };

  const getImpactColor = (impact: number) => {
    if (impact > 0.1) return 'text-green-600';
    if (impact < -0.1) return 'text-red-600';
    return 'text-gray-600';
  };

  const formatFactorValue = (value: any) => {
    if (typeof value === 'number') {
      return value.toFixed(2);
    }
    if (typeof value === 'boolean') {
      return value ? '是' : '否';
    }
    return String(value);
  };

  // 模拟敏感性数据
  const defaultSensitivities: FactorSensitivity[] = factorSensitivities.length > 0 
    ? factorSensitivities 
    : [
        {
          factor_name: '信用评分',
          original_value: 750,
          sensitivity_range: [600, 850],
          impact_curve: [
            {value: 600, impact: -0.5, probability: 0.3},
            {value: 650, impact: -0.3, probability: 0.5},
            {value: 700, impact: -0.1, probability: 0.7},
            {value: 750, impact: 0, probability: 0.85},
            {value: 800, impact: 0.1, probability: 0.9},
            {value: 850, impact: 0.15, probability: 0.95}
          ],
          critical_threshold: 650,
          elasticity: 0.8
        },
        {
          factor_name: '月收入',
          original_value: 15000,
          sensitivity_range: [5000, 25000],
          impact_curve: [
            {value: 5000, impact: -0.6, probability: 0.2},
            {value: 8000, impact: -0.4, probability: 0.4},
            {value: 12000, impact: -0.2, probability: 0.6},
            {value: 15000, impact: 0, probability: 0.85},
            {value: 20000, impact: 0.1, probability: 0.9},
            {value: 25000, impact: 0.15, probability: 0.92}
          ],
          critical_threshold: 8000,
          elasticity: 0.6
        }
      ];

  return (
    <div className={`space-y-4 ${className}`}>
      {/* 反事实分析标题 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <HelpCircle className="h-5 w-5 text-orange-600" />
              <span>反事实分析 - "如果...会怎样？"</span>
              <Badge variant="outline" className="ml-2">
                {scenarios.length} 场景
              </Badge>
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button
                onClick={() => setAnimationEnabled(!animationEnabled)}
                variant="outline"
                size="sm"
              >
                {animationEnabled ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {animationEnabled ? '暂停' : '开启'}动画
              </Button>
              <Button
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                variant="outline"
                size="sm"
              >
                <Settings className="h-4 w-4 mr-2" />
                {showTechnicalDetails ? '隐藏' : '显示'}技术细节
              </Button>
            </div>
          </div>
          <p className="text-sm text-gray-600">
            通过改变关键因素，分析对决策结果的潜在影响，帮助理解决策的敏感性和稳健性
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-sm text-gray-600">基线结果</div>
              <div className="text-lg font-semibold text-blue-600">{baselineOutcome}</div>
              <div className="text-sm text-gray-500">置信度: {(baselineConfidence * 100).toFixed(1)}%</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-600">已分析场景</div>
              <div className="text-lg font-semibold text-green-600">{scenarios.length}</div>
              <div className="text-sm text-gray-500">覆盖主要风险因素</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-600">最大影响差异</div>
              <div className={`text-lg font-semibold ${getImpactColor(Math.max(...scenarios.map(s => Math.abs(s.impact_difference))))}`}>
                {(Math.max(...scenarios.map(s => Math.abs(s.impact_difference))) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">敏感性评估</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 分析视图切换 */}
      <Tabs value={selectedView} onValueChange={(value: any) => setSelectedView(value)}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="scenarios" >
            <Eye className="h-4 w-4 mr-2" />
            场景分析
          </TabsTrigger>
          <TabsTrigger value="sensitivity" >
            <BarChart3 className="h-4 w-4 mr-2" />
            敏感性分析
          </TabsTrigger>
          <TabsTrigger value="comparison" >
            <TrendingUp className="h-4 w-4 mr-2" />
            对比分析
          </TabsTrigger>
          <TabsTrigger value="simulation" >
            <Zap className="h-4 w-4 mr-2" />
            模拟实验
          </TabsTrigger>
        </TabsList>

        <TabsContent value="scenarios" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {scenarios.map((scenario, index) => {
              const isSelected = selectedScenario === scenario.scenario_id;
              return (
                <Card 
                  key={scenario.scenario_id} 
                  className={`cursor-pointer transition-all duration-200 ${
                    isSelected ? 'ring-2 ring-orange-500 border-orange-200' : 'hover:shadow-md'
                  } ${animationEnabled ? 'transform hover:scale-105' : ''}`}
                  onClick={() => setSelectedScenario(isSelected ? null : scenario.scenario_id)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                          <span className="text-orange-600 font-semibold text-sm">{index + 1}</span>
                        </div>
                        <div>
                          <h4 className="font-semibold">{scenario.scenario_name}</h4>
                          <div className="flex items-center space-x-2 mt-1">
                            <Badge className={getRiskColor(scenario.risk_level)} variant="outline">
                              {scenario.risk_level === 'low' ? '低风险' : 
                               scenario.risk_level === 'medium' ? '中风险' : '高风险'}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              概率: {(scenario.probability * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center space-x-1">
                          {getImpactIcon(scenario.impact_difference)}
                          <span className={`font-semibold ${getImpactColor(scenario.impact_difference)}`}>
                            {scenario.impact_difference > 0 ? '+' : ''}{(scenario.impact_difference * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-500">影响差异</div>
                      </div>
                    </div>
                  </CardHeader>
                  
                  {isSelected && (
                    <CardContent className="pt-0">
                      <div className="space-y-4">
                        {/* 预测结果 */}
                        <div className="bg-orange-50 p-3 rounded-lg">
                          <h5 className="font-medium text-orange-800 mb-1">预测结果</h5>
                          <p className="text-orange-700 text-sm">{scenario.predicted_outcome}</p>
                        </div>

                        {/* 改变的因素 */}
                        <div>
                          <h5 className="font-medium mb-2">改变的因素</h5>
                          <div className="space-y-2">
                            {Object.entries(scenario.changed_factors).map(([factor, value]) => (
                              <div key={factor} className="flex items-center justify-between text-sm bg-gray-50 p-2 rounded">
                                <span className="font-medium">{factor}:</span>
                                <div className="flex items-center space-x-2">
                                  <span className="text-gray-600">→</span>
                                  <span className="font-medium text-blue-600">{formatFactorValue(value)}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* 影响分析 */}
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-sm text-gray-600">置信度变化</div>
                            <div className={`text-lg font-semibold ${getImpactColor(scenario.confidence_change)}`}>
                              {scenario.confidence_change > 0 ? '+' : ''}{(scenario.confidence_change * 100).toFixed(1)}%
                            </div>
                            <Progress 
                              value={Math.abs(scenario.confidence_change) * 100} 
                              className="h-2 mt-1" 
                            />
                          </div>
                          <div>
                            <div className="text-sm text-gray-600">发生概率</div>
                            <div className="text-lg font-semibold text-purple-600">
                              {(scenario.probability * 100).toFixed(1)}%
                            </div>
                            <Progress 
                              value={scenario.probability * 100} 
                              className="h-2 mt-1" 
                            />
                          </div>
                        </div>

                        {/* 解释说明 */}
                        <div className="bg-blue-50 p-3 rounded-lg">
                          <h5 className="font-medium text-blue-800 mb-1">解释说明</h5>
                          <p className="text-blue-700 text-sm">{scenario.explanation}</p>
                        </div>

                        {/* 技术细节 */}
                        {showTechnicalDetails && scenario.metadata && (
                          <div className="bg-gray-50 p-3 rounded-lg">
                            <h5 className="font-medium text-gray-800 mb-2">技术指标</h5>
                            <div className="grid grid-cols-2 gap-4 text-xs">
                              {scenario.metadata.computation_time && (
                                <div>
                                  <span className="text-gray-600">计算时间:</span>
                                  <div className="font-medium">{scenario.metadata.computation_time}ms</div>
                                </div>
                              )}
                              {scenario.metadata.model_confidence && (
                                <div>
                                  <span className="text-gray-600">模型置信度:</span>
                                  <div className="font-medium">{(scenario.metadata.model_confidence * 100).toFixed(1)}%</div>
                                </div>
                              )}
                              {scenario.metadata.sensitivity_score && (
                                <div>
                                  <span className="text-gray-600">敏感性分数:</span>
                                  <div className="font-medium">{scenario.metadata.sensitivity_score.toFixed(3)}</div>
                                </div>
                              )}
                              {scenario.metadata.plausibility && (
                                <div>
                                  <span className="text-gray-600">可信度:</span>
                                  <div className="font-medium">{(scenario.metadata.plausibility * 100).toFixed(1)}%</div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  )}
                </Card>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="sensitivity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>因素敏感性分析</CardTitle>
              <p className="text-sm text-gray-600">
                分析各个因素对决策结果的敏感程度，识别关键影响因子
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {defaultSensitivities.map((sensitivity, index) => (
                  <div key={sensitivity.factor_name} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold">{sensitivity.factor_name}</h4>
                      <div className="flex items-center space-x-4">
                        <div className="text-sm">
                          <span className="text-gray-600">当前值: </span>
                          <span className="font-medium">{formatFactorValue(sensitivity.original_value)}</span>
                        </div>
                        <div className="text-sm">
                          <span className="text-gray-600">弹性系数: </span>
                          <span className="font-medium">{sensitivity.elasticity.toFixed(2)}</span>
                        </div>
                      </div>
                    </div>

                    {/* 敏感性曲线可视化 */}
                    <div className="bg-gray-50 p-4 rounded mb-3">
                      <div className="flex justify-between items-end h-32 space-x-1">
                        {sensitivity.impact_curve.map((point, i) => {
                          const height = ((point.impact + 1) / 2) * 100; // 归一化到0-100%
                          const isOriginal = point.value === sensitivity.original_value;
                          const isCritical = sensitivity.critical_threshold && 
                                           point.value <= sensitivity.critical_threshold;
                          
                          return (
                            <div key={i} className="flex-1 flex flex-col items-center">
                              <div 
                                className={`w-full rounded-t transition-all duration-500 ${
                                  isOriginal ? 'bg-blue-500' : 
                                  isCritical ? 'bg-red-400' : 'bg-gray-400'
                                } ${animationEnabled ? 'hover:bg-opacity-80' : ''}`}
                                style={{ height: `${height}%` }}
                                title={`值: ${point.value}, 影响: ${(point.impact * 100).toFixed(1)}%`}
                              />
                              <div className="text-xs text-gray-600 mt-1 transform rotate-45">
                                {point.value}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="flex justify-between text-xs text-gray-500 mt-2">
                        <span>负影响</span>
                        <span>中性</span>
                        <span>正影响</span>
                      </div>
                    </div>

                    {/* 关键阈值提醒 */}
                    {sensitivity.critical_threshold && (
                      <div className="flex items-center space-x-2 text-sm">
                        <AlertTriangle className="h-4 w-4 text-red-500" />
                        <span className="text-red-700">
                          关键阈值: {sensitivity.critical_threshold} 
                          (低于此值将显著影响决策结果)
                        </span>
                      </div>
                    )}

                    {/* 敏感性统计 */}
                    <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
                      <div className="text-center p-2 bg-blue-50 rounded">
                        <div className="text-blue-600 font-medium">
                          {sensitivity.sensitivity_range[0]} - {sensitivity.sensitivity_range[1]}
                        </div>
                        <div className="text-blue-700">取值范围</div>
                      </div>
                      <div className="text-center p-2 bg-green-50 rounded">
                        <div className="text-green-600 font-medium">
                          {Math.max(...sensitivity.impact_curve.map(p => Math.abs(p.impact))).toFixed(2)}
                        </div>
                        <div className="text-green-700">最大影响</div>
                      </div>
                      <div className="text-center p-2 bg-purple-50 rounded">
                        <div className="text-purple-600 font-medium">
                          {sensitivity.elasticity > 0.7 ? '高' : sensitivity.elasticity > 0.4 ? '中' : '低'}
                        </div>
                        <div className="text-purple-700">敏感性等级</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>场景对比分析</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse border border-gray-200">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="border border-gray-200 p-3 text-left">场景名称</th>
                      <th className="border border-gray-200 p-3 text-center">风险等级</th>
                      <th className="border border-gray-200 p-3 text-center">发生概率</th>
                      <th className="border border-gray-200 p-3 text-center">影响程度</th>
                      <th className="border border-gray-200 p-3 text-center">置信度变化</th>
                      <th className="border border-gray-200 p-3 text-center">综合评分</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scenarios
                      .sort((a, b) => Math.abs(b.impact_difference) - Math.abs(a.impact_difference))
                      .map((scenario, index) => {
                        const compositeScore = (
                          scenario.probability * 0.3 + 
                          Math.abs(scenario.impact_difference) * 0.4 +
                          (scenario.risk_level === 'high' ? 1 : scenario.risk_level === 'medium' ? 0.6 : 0.3) * 0.3
                        );
                        
                        return (
                          <tr key={scenario.scenario_id} className={index % 2 === 0 ? 'bg-gray-25' : ''}>
                            <td className="border border-gray-200 p-3 font-medium">
                              {scenario.scenario_name}
                            </td>
                            <td className="border border-gray-200 p-3 text-center">
                              <Badge className={getRiskColor(scenario.risk_level)} variant="outline">
                                {scenario.risk_level === 'low' ? '低' : 
                                 scenario.risk_level === 'medium' ? '中' : '高'}
                              </Badge>
                            </td>
                            <td className="border border-gray-200 p-3 text-center">
                              {(scenario.probability * 100).toFixed(1)}%
                            </td>
                            <td className="border border-gray-200 p-3 text-center">
                              <div className="flex items-center justify-center space-x-1">
                                {getImpactIcon(scenario.impact_difference)}
                                <span className={getImpactColor(scenario.impact_difference)}>
                                  {(Math.abs(scenario.impact_difference) * 100).toFixed(1)}%
                                </span>
                              </div>
                            </td>
                            <td className="border border-gray-200 p-3 text-center">
                              <span className={getImpactColor(scenario.confidence_change)}>
                                {scenario.confidence_change > 0 ? '+' : ''}{(scenario.confidence_change * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="border border-gray-200 p-3 text-center">
                              <div className="flex items-center justify-center">
                                <div className={`w-3 h-3 rounded-full mr-2 ${
                                  compositeScore > 0.7 ? 'bg-red-500' : 
                                  compositeScore > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                                }`} />
                                <span className="font-medium">{compositeScore.toFixed(2)}</span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="simulation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle >
                <Zap className="h-5 w-5 mr-2" />
                交互式模拟实验
              </CardTitle>
              <p className="text-sm text-gray-600">
                调整关键参数，实时观察对决策结果的影响
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500">
                  <div className="flex items-center space-x-2 mb-2">
                    <CheckCircle className="h-5 w-5 text-blue-600" />
                    <span className="font-medium text-blue-800">交互式模拟功能</span>
                  </div>
                  <p className="text-blue-700 text-sm">
                    此功能展示了如何通过交互式界面让用户实时调整参数并观察影响。
                    在实际实现中，可以连接到后端API进行实时计算。
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-3">参数调整器</h4>
                    <div className="space-y-4">
                      {defaultSensitivities.slice(0, 2).map((sensitivity) => (
                        <div key={sensitivity.factor_name} className="border rounded p-3">
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-medium">{sensitivity.factor_name}</span>
                            <span className="text-sm text-gray-600">
                              当前: {formatFactorValue(sensitivity.original_value)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min={sensitivity.sensitivity_range[0]}
                            max={sensitivity.sensitivity_range[1]}
                            defaultValue={sensitivity.original_value}
                            className="w-full"
                            disabled
                          />
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>{sensitivity.sensitivity_range[0]}</span>
                            <span>{sensitivity.sensitivity_range[1]}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-3">实时结果预览</h4>
                    <div className="border rounded p-4 bg-gray-50">
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span>预测结果:</span>
                          <span className="font-medium text-blue-600">{baselineOutcome}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>置信度:</span>
                          <span className="font-medium">{(baselineConfidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>风险等级:</span>
                          <Badge className="text-green-600 bg-green-100">低风险</Badge>
                        </div>
                        <Progress value={baselineConfidence * 100} className="h-2" />
                      </div>
                    </div>

                    <div className="mt-4">
                      <Button className="w-full" disabled>
                        <RotateCcw className="h-4 w-4 mr-2" />
                        重置参数
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CounterfactualAnalysis;