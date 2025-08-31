import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Alert } from '../components/ui/alert';
import { Progress } from '../components/ui/progress';

interface AnalysisReport {
  id: string;
  name: string;
  type: 'experiment_summary' | 'algorithm_comparison' | 'parameter_analysis' | 'performance_report';
  status: 'generating' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  experiments: string[];
  insights: ReportInsight[];
  recommendations: string[];
  export_formats: string[];
}

interface ReportInsight {
  type: 'finding' | 'warning' | 'optimization';
  title: string;
  description: string;
  impact_level: 'low' | 'medium' | 'high';
  data?: Record<string, any>;
}

interface ExperimentSummary {
  experiment_id: string;
  experiment_name: string;
  algorithm: string;
  status: string;
  total_trials: number;
  best_value: number;
  best_params: Record<string, any>;
  success_rate: number;
  average_duration: number;
  resource_efficiency: number;
}

interface AlgorithmComparison {
  algorithm: string;
  experiments_count: number;
  average_best_value: number;
  average_trials_to_best: number;
  success_rate: number;
  average_duration: number;
  convergence_speed: number;
  resource_usage: number;
}

interface ParameterInsight {
  parameter_name: string;
  importance_score: number;
  optimal_range: [number, number];
  correlation_with_objective: number;
  interaction_effects: Array<{
    with_parameter: string;
    interaction_strength: number;
  }>;
}

const HyperparameterReportsPage: React.FC = () => {
  const [reports, setReports] = useState<AnalysisReport[]>([]);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [algorithmComparisons, setAlgorithmComparisons] = useState<AlgorithmComparison[]>([]);
  const [parameterInsights, setParameterInsights] = useState<ParameterInsight[]>([]);
  const [selectedReport, setSelectedReport] = useState<AnalysisReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'insights' | 'comparisons' | 'parameters'>('overview');

  const API_BASE = '/api/v1/hyperparameter-optimization';

  // 加载报告数据
  const loadReports = async () => {
    try {
      setLoading(true);
      
      // 模拟实验摘要数据
      const mockExperiments: ExperimentSummary[] = [
        {
          experiment_id: 'exp-1',
          experiment_name: '神经网络超参数优化',
          algorithm: 'tpe',
          status: 'completed',
          total_trials: 150,
          best_value: 0.9234,
          best_params: { learning_rate: 0.001, batch_size: 32, hidden_units: 128 },
          success_rate: 87.3,
          average_duration: 45.6,
          resource_efficiency: 92.1
        },
        {
          experiment_id: 'exp-2',
          experiment_name: '深度学习模型调优',
          algorithm: 'cmaes',
          status: 'completed',
          total_trials: 200,
          best_value: 0.8967,
          best_params: { learning_rate: 0.0015, dropout: 0.3, l2_reg: 0.001 },
          success_rate: 79.5,
          average_duration: 62.3,
          resource_efficiency: 88.7
        },
        {
          experiment_id: 'exp-3',
          experiment_name: '随机森林参数搜索',
          algorithm: 'random',
          status: 'completed',
          total_trials: 100,
          best_value: 0.8756,
          best_params: { n_estimators: 200, max_depth: 15, min_samples_split: 5 },
          success_rate: 65.0,
          average_duration: 12.4,
          resource_efficiency: 73.2
        }
      ];

      // 模拟算法比较数据
      const mockAlgorithmComparisons: AlgorithmComparison[] = [
        {
          algorithm: 'TPE',
          experiments_count: 5,
          average_best_value: 0.9156,
          average_trials_to_best: 34.2,
          success_rate: 84.6,
          average_duration: 48.7,
          convergence_speed: 8.7,
          resource_usage: 78.3
        },
        {
          algorithm: 'CMA-ES',
          experiments_count: 3,
          average_best_value: 0.8934,
          average_trials_to_best: 56.8,
          success_rate: 76.4,
          average_duration: 67.2,
          convergence_speed: 6.4,
          resource_usage: 89.1
        },
        {
          algorithm: '随机搜索',
          experiments_count: 4,
          average_best_value: 0.8456,
          average_trials_to_best: 78.5,
          success_rate: 62.8,
          average_duration: 23.1,
          convergence_speed: 3.2,
          resource_usage: 45.7
        }
      ];

      // 模拟参数洞察数据
      const mockParameterInsights: ParameterInsight[] = [
        {
          parameter_name: 'learning_rate',
          importance_score: 0.89,
          optimal_range: [0.0001, 0.01],
          correlation_with_objective: 0.76,
          interaction_effects: [
            { with_parameter: 'batch_size', interaction_strength: 0.45 },
            { with_parameter: 'hidden_units', interaction_strength: 0.32 }
          ]
        },
        {
          parameter_name: 'batch_size',
          importance_score: 0.67,
          optimal_range: [16, 128],
          correlation_with_objective: 0.52,
          interaction_effects: [
            { with_parameter: 'learning_rate', interaction_strength: 0.45 },
            { with_parameter: 'dropout', interaction_strength: 0.28 }
          ]
        },
        {
          parameter_name: 'dropout',
          importance_score: 0.54,
          optimal_range: [0.1, 0.5],
          correlation_with_objective: -0.34,
          interaction_effects: [
            { with_parameter: 'batch_size', interaction_strength: 0.28 },
            { with_parameter: 'l2_reg', interaction_strength: 0.41 }
          ]
        }
      ];

      // 模拟报告数据
      const mockReports: AnalysisReport[] = [
        {
          id: 'report-1',
          name: '本月实验总结报告',
          type: 'experiment_summary',
          status: 'completed',
          created_at: new Date(Date.now() - 86400000).toISOString(),
          completed_at: new Date(Date.now() - 86400000 + 3600000).toISOString(),
          experiments: ['exp-1', 'exp-2', 'exp-3'],
          insights: [
            {
              type: 'finding',
              title: 'TPE算法表现最佳',
              description: 'TPE算法在所有测试场景中都获得了最高的平均最佳值，建议优先使用',
              impact_level: 'high'
            },
            {
              type: 'optimization',
              title: '学习率参数影响最大',
              description: 'learning_rate是影响模型性能最重要的参数，重要性评分达到0.89',
              impact_level: 'high'
            },
            {
              type: 'warning',
              title: '资源利用率待优化',
              description: 'CMA-ES算法的资源使用率较高，建议在资源受限环境中谨慎使用',
              impact_level: 'medium'
            }
          ],
          recommendations: [
            '推荐使用TPE算法进行超参数优化，特别是对于神经网络模型',
            '重点关注learning_rate参数的调优，其对模型性能影响最大',
            '考虑增加试验的并行度以提高整体优化效率',
            '在资源受限的情况下，可以先使用随机搜索进行初步探索'
          ],
          export_formats: ['pdf', 'excel', 'json']
        },
        {
          id: 'report-2',
          name: '算法性能对比分析',
          type: 'algorithm_comparison',
          status: 'completed',
          created_at: new Date(Date.now() - 172800000).toISOString(),
          completed_at: new Date(Date.now() - 172800000 + 7200000).toISOString(),
          experiments: ['exp-1', 'exp-2', 'exp-3', 'exp-4', 'exp-5'],
          insights: [
            {
              type: 'finding',
              title: 'TPE在收敛速度上领先',
              description: 'TPE算法平均只需要34.2次试验就能找到最佳值，比其他算法快40%以上',
              impact_level: 'high'
            }
          ],
          recommendations: [
            '对于需要快速获得结果的场景，推荐使用TPE算法',
            'CMA-ES适合对精度要求很高的连续参数优化',
            '随机搜索可以作为基准和快速验证的方法'
          ],
          export_formats: ['pdf', 'excel']
        },
        {
          id: 'report-3',
          name: '参数重要性分析',
          type: 'parameter_analysis',
          status: 'generating',
          created_at: new Date(Date.now() - 3600000).toISOString(),
          experiments: ['exp-1', 'exp-2'],
          insights: [],
          recommendations: [],
          export_formats: []
        }
      ];

      setExperiments(mockExperiments);
      setAlgorithmComparisons(mockAlgorithmComparisons);
      setParameterInsights(mockParameterInsights);
      setReports(mockReports);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReports();
  }, []);

  // 生成新报告
  const generateReport = async (type: AnalysisReport['type']) => {
    try {
      const newReport: AnalysisReport = {
        id: `report-${Date.now()}`,
        name: `${getReportTypeName(type)} - ${new Date().toLocaleDateString('zh-CN')}`,
        type,
        status: 'generating',
        created_at: new Date().toISOString(),
        experiments: experiments.map(e => e.experiment_id),
        insights: [],
        recommendations: [],
        export_formats: []
      };

      setReports(prev => [newReport, ...prev]);

      // 模拟异步生成过程
      setTimeout(() => {
        setReports(prev => prev.map(report => 
          report.id === newReport.id 
            ? { ...report, status: 'completed', completed_at: new Date().toISOString() }
            : report
        ));
      }, 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate report');
    }
  };

  // 导出报告
  const exportReport = async (reportId: string, format: string) => {
    try {
      // 模拟导出过程
      const link = document.createElement('a');
      link.href = '#';
      link.download = `report-${reportId}.${format}`;
      link.click();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export report');
    }
  };

  // 获取报告类型名称
  const getReportTypeName = (type: AnalysisReport['type']) => {
    switch (type) {
      case 'experiment_summary':
        return '实验摘要报告';
      case 'algorithm_comparison':
        return '算法对比分析';
      case 'parameter_analysis':
        return '参数重要性分析';
      case 'performance_report':
        return '性能分析报告';
      default:
        return '分析报告';
    }
  };

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500 text-white';
      case 'generating':
        return 'bg-blue-500 text-white';
      case 'failed':
        return 'bg-red-500 text-white';
      default:
        return 'bg-gray-500 text-white';
    }
  };

  // 获取洞察类型颜色
  const getInsightTypeColor = (type: string) => {
    switch (type) {
      case 'finding':
        return 'bg-blue-100 text-blue-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      case 'optimization':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // 渲染报告概览
  const renderOverview = () => (
    <div className="space-y-6">
      {/* 报告列表 */}
      <div className="space-y-4">
        {reports.map((report) => (
          <Card key={report.id} className="p-6">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">{report.name}</h3>
                  <Badge className={getStatusColor(report.status)}>
                    {report.status}
                  </Badge>
                  <Badge className="bg-purple-100 text-purple-800">
                    {getReportTypeName(report.type)}
                  </Badge>
                </div>
                
                <div className="text-sm text-gray-600 mb-3">
                  创建时间: {new Date(report.created_at).toLocaleString('zh-CN')}
                  {report.completed_at && (
                    <span className="ml-4">
                      完成时间: {new Date(report.completed_at).toLocaleString('zh-CN')}
                    </span>
                  )}
                </div>
                
                <div className="text-sm text-gray-600">
                  包含实验: {report.experiments.length} 个
                  {report.insights.length > 0 && (
                    <span className="ml-4">洞察: {report.insights.length} 条</span>
                  )}
                  {report.recommendations.length > 0 && (
                    <span className="ml-4">建议: {report.recommendations.length} 条</span>
                  )}
                </div>
              </div>
              
              <div className="flex space-x-2 ml-4">
                {report.status === 'completed' && (
                  <>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => setSelectedReport(report)}
                    >
                      查看详情
                    </Button>
                    {report.export_formats.map((format) => (
                      <Button
                        key={format}
                        size="sm"
                        variant="outline"
                        onClick={() => exportReport(report.id, format)}
                      >
                        导出{format.toUpperCase()}
                      </Button>
                    ))}
                  </>
                )}
                {report.status === 'generating' && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">生成中...</span>
                  </div>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* 生成新报告 */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">生成新报告</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button onClick={() => generateReport('experiment_summary')}>
            实验摘要报告
          </Button>
          <Button onClick={() => generateReport('algorithm_comparison')}>
            算法对比分析
          </Button>
          <Button onClick={() => generateReport('parameter_analysis')}>
            参数重要性分析
          </Button>
          <Button onClick={() => generateReport('performance_report')}>
            性能分析报告
          </Button>
        </div>
      </Card>
    </div>
  );

  // 渲染洞察分析
  const renderInsights = () => (
    <div className="space-y-6">
      {selectedReport ? (
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold">{selectedReport.name} - 分析洞察</h2>
            <Button variant="outline" onClick={() => setSelectedReport(null)}>
              返回列表
            </Button>
          </div>

          {selectedReport.insights.length === 0 ? (
            <Card className="p-8 text-center">
              <div className="text-gray-500">该报告暂无分析洞察</div>
            </Card>
          ) : (
            <div className="space-y-4">
              {selectedReport.insights.map((insight, index) => (
                <Card key={index} className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-lg font-semibold text-gray-900">{insight.title}</h3>
                        <Badge className={getInsightTypeColor(insight.type)}>
                          {insight.type}
                        </Badge>
                        <Badge className={`${
                          insight.impact_level === 'high' ? 'bg-red-100 text-red-800' :
                          insight.impact_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {insight.impact_level} 影响
                        </Badge>
                      </div>
                      <p className="text-gray-600">{insight.description}</p>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {/* 推荐建议 */}
          {selectedReport.recommendations.length > 0 && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">推荐建议</h3>
              <div className="space-y-3">
                {selectedReport.recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start">
                    <span className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center text-xs font-medium mr-3 mt-0.5">
                      {index + 1}
                    </span>
                    <p className="text-gray-700">{recommendation}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      ) : (
        <Card className="p-8 text-center">
          <div className="text-gray-500 mb-4">请先选择一个报告查看详细洞察</div>
          <Button onClick={() => setActiveTab('overview')}>
            返回报告列表
          </Button>
        </Card>
      )}
    </div>
  );

  // 渲染算法比较
  const renderAlgorithmComparisons = () => (
    <div className="space-y-6">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {['算法', '实验数量', '平均最佳值', '收敛试验数', '成功率', '平均耗时', '收敛速度', '资源使用'].map((header) => (
                <th key={header} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {algorithmComparisons.map((comparison, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <Badge className="bg-blue-100 text-blue-800">
                    {comparison.algorithm}
                  </Badge>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {comparison.experiments_count}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-blue-600">
                  {comparison.average_best_value.toFixed(4)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {comparison.average_trials_to_best.toFixed(1)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {comparison.success_rate.toFixed(1)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {comparison.average_duration.toFixed(1)}分钟
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {comparison.convergence_speed.toFixed(1)}/10
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {comparison.resource_usage.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  // 渲染参数分析
  const renderParameterAnalysis = () => (
    <div className="space-y-6">
      {parameterInsights.map((insight, index) => (
        <Card key={index} className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">{insight.parameter_name}</h3>
              <div className="flex items-center space-x-4 mt-2">
                <div className="text-sm text-gray-600">
                  重要性评分: <span className="font-semibold text-blue-600">{insight.importance_score.toFixed(2)}</span>
                </div>
                <div className="text-sm text-gray-600">
                  与目标相关性: <span className="font-semibold">{insight.correlation_with_objective.toFixed(2)}</span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">最优范围</div>
              <div className="font-mono text-gray-900">
                [{insight.optimal_range[0]}, {insight.optimal_range[1]}]
              </div>
            </div>
          </div>

          {/* 重要性进度条 */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>参数重要性</span>
              <span>{(insight.importance_score * 100).toFixed(0)}%</span>
            </div>
            <Progress value={insight.importance_score * 100} className="h-3" />
          </div>

          {/* 交互效应 */}
          {insight.interaction_effects.length > 0 && (
            <div>
              <h4 className="font-medium text-gray-900 mb-3">与其他参数的交互效应</h4>
              <div className="space-y-2">
                {insight.interaction_effects.map((effect, effectIndex) => (
                  <div key={effectIndex} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">{effect.with_parameter}</span>
                    <div className="flex items-center space-x-2">
                      <Progress value={effect.interaction_strength * 100} className="h-2 w-24" />
                      <span className="text-xs text-gray-500 w-12">
                        {(effect.interaction_strength * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      ))}
    </div>
  );

  const tabsData = [
    { key: 'overview', label: '报告概览', count: reports.length },
    { key: 'insights', label: '分析洞察' },
    { key: 'comparisons', label: '算法对比', count: algorithmComparisons.length },
    { key: 'parameters', label: '参数分析', count: parameterInsights.length }
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">分析报告</h1>
            <p className="mt-2 text-gray-600">
              生成和查看超参数优化的深度分析报告
            </p>
          </div>
          <Button onClick={loadReports} disabled={loading}>
            {loading ? '刷新中...' : '刷新'}
          </Button>
        </div>

        {error && (
          <Alert variant="destructive">
            {error}
          </Alert>
        )}

        {/* 标签页导航 */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabsData.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
                {tab.count !== undefined && (
                  <span className="ml-2 bg-gray-100 text-gray-900 py-0.5 px-2 rounded-full text-xs">
                    {tab.count}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* 内容区域 */}
        <div>
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'insights' && renderInsights()}
          {activeTab === 'comparisons' && renderAlgorithmComparisons()}
          {activeTab === 'parameters' && renderParameterAnalysis()}
        </div>
      </div>
    </div>
  );
};

export default HyperparameterReportsPage;