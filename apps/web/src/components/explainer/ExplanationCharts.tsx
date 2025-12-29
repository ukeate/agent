import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart3, 
  TrendingUp, 
  GitBranch,
  Target,
  Network,
  Settings,
  Info,
} from 'lucide-react';

interface DecisionTreeNode {
  id: string;
  label: string;
  condition?: string;
  outcome?: string;
  confidence: number;
  samples: number;
  children?: DecisionTreeNode[];
  isLeaf: boolean;
  depth: number;
}

interface FactorImportanceData {
  factor_name: string;
  importance_score: number;
  shap_value: number;
  contribution: number;
  p_value?: number;
  feature_type: 'numerical' | 'categorical' | 'binary';
}

interface ConfidenceDistribution {
  bin_start: number;
  bin_end: number;
  count: number;
  accuracy: number;
  calibration_score: number;
}

interface ExplanationChartsProps {
  factorImportance?: FactorImportanceData[];
  decisionTree?: DecisionTreeNode;
  confidenceDistribution?: ConfidenceDistribution[];
  timeSeriesData?: { timestamp: string; confidence: number; accuracy: number }[];
  correlationMatrix?: { factor1: string; factor2: string; correlation: number }[];
  className?: string;
}

const ExplanationCharts: React.FC<ExplanationChartsProps> = ({
  factorImportance = [],
  decisionTree,
  confidenceDistribution = [],
  timeSeriesData = [],
  correlationMatrix = [],
  className = ''
}) => {
  const [selectedChart, setSelectedChart] = useState<'factors' | 'tree' | 'confidence' | 'timeseries' | 'correlation'>('factors');
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const collectNodes = (node: DecisionTreeNode, nodes: DecisionTreeNode[]) => {
    nodes.push(node);
    node.children?.forEach(child => collectNodes(child, nodes));
  };
  const treeNodes: DecisionTreeNode[] = [];
  if (decisionTree) collectNodes(decisionTree, treeNodes);
  const treeDepth = treeNodes.length ? Math.max(...treeNodes.map(n => n.depth)) + 1 : 0;
  const treeLeaves = treeNodes.filter(n => n.isLeaf).length;
  const treeAvgConfidence = treeNodes.length
    ? treeNodes.reduce((sum, node) => sum + node.confidence, 0) / treeNodes.length
    : 0;
  const confidenceDenom = Math.max(confidenceDistribution.length - 1, 1);
  const confidenceAvgCalibration = confidenceDistribution.length
    ? confidenceDistribution.reduce((sum, point) => sum + point.calibration_score, 0) / confidenceDistribution.length
    : 0;
  const confidenceTotalSamples = confidenceDistribution.reduce((sum, point) => sum + point.count, 0);
  const confidenceMaxError = confidenceDistribution.length
    ? Math.max(...confidenceDistribution.map(p => Math.abs(p.accuracy - (p.bin_start + p.bin_end) / 2)))
    : 0;

  const renderDecisionTreeNode = (node: DecisionTreeNode, x: number, y: number, width: number, isRoot = false) => {
    const nodeWidth = 120;
    const nodeHeight = 60;
    const childSpacing = width / (node.children?.length || 1);
    
    return (
      <g key={node.id}>
        {/* 节点 */}
        <rect
          x={x - nodeWidth / 2}
          y={y - nodeHeight / 2}
          width={nodeWidth}
          height={nodeHeight}
          rx={8}
          className={`${
            node.isLeaf 
              ? 'fill-blue-100 stroke-blue-500' 
              : 'fill-gray-100 stroke-gray-500'
          } stroke-2`}
        />
        
        {/* 节点文字 */}
        <text
          x={x}
          y={y - 10}
          textAnchor="middle"
          className="fill-gray-800 text-xs font-medium"
        >
          {node.label}
        </text>
        
        {node.condition && (
          <text
            x={x}
            y={y + 5}
            textAnchor="middle"
            className="fill-gray-600 text-xs"
          >
            {node.condition}
          </text>
        )}
        
        <text
          x={x}
          y={y + 20}
          textAnchor="middle"
          className="fill-gray-700 text-xs"
        >
          {(node.confidence * 100).toFixed(0)}% ({node.samples})
        </text>
        
        {/* 连接线和子节点 */}
        {node.children?.map((child, index) => {
          const childX = x - width / 2 + (index + 0.5) * childSpacing;
          const childY = y + 120;
          
          return (
            <g key={child.id}>
              {/* 连接线 */}
              <line
                x1={x}
                y1={y + nodeHeight / 2}
                x2={childX}
                y2={childY - nodeHeight / 2}
                className="stroke-gray-400 stroke-2"
              />
              
              {/* 递归渲染子节点 */}
              {renderDecisionTreeNode(child, childX, childY, childSpacing * 0.8)}
            </g>
          );
        })}
      </g>
    );
  };

  const getFeatureTypeColor = (type: string) => {
    switch (type) {
      case 'numerical':
        return 'bg-blue-100 text-blue-700';
      case 'categorical':
        return 'bg-green-100 text-green-700';
      case 'binary':
        return 'bg-purple-100 text-purple-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* 图表类型选择 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              <span>解释性可视化图表</span>
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                variant="outline"
                size="sm"
              >
                <Settings className="h-4 w-4 mr-2" />
                {showTechnicalDetails ? '隐藏' : '显示'}技术参数
              </Button>
            </div>
          </div>
          <p className="text-sm text-gray-600">
            通过多种图表形式展示AI决策的内部机制和统计特性
          </p>
        </CardHeader>
      </Card>

      <Tabs value={selectedChart} onValueChange={(value: any) => setSelectedChart(value)}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="factors" >
            <BarChart3 className="h-4 w-4 mr-2" />
            因素重要性
          </TabsTrigger>
          <TabsTrigger value="tree" >
            <GitBranch className="h-4 w-4 mr-2" />
            决策树
          </TabsTrigger>
          <TabsTrigger value="confidence" >
            <Target className="h-4 w-4 mr-2" />
            置信度校准
          </TabsTrigger>
          <TabsTrigger value="timeseries" >
            <TrendingUp className="h-4 w-4 mr-2" />
            时间序列
          </TabsTrigger>
          <TabsTrigger value="correlation" >
            <Network className="h-4 w-4 mr-2" />
            因子相关性
          </TabsTrigger>
        </TabsList>

        <TabsContent value="factors" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>因素重要性分析</CardTitle>
              <p className="text-sm text-gray-600">
                基于SHAP值和特征重要性分析各因素对决策结果的影响程度
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {factorImportance.length > 0 ? factorImportance
                  .slice()
                  .sort((a, b) => b.importance_score - a.importance_score)
                  .map((factor, index) => (
                    <div key={factor.factor_name} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center text-white text-xs font-bold">
                            {index + 1}
                          </div>
                          <div>
                            <span className="font-medium">{factor.factor_name}</span>
                            <div className="flex items-center space-x-2 mt-1">
                              <Badge className={getFeatureTypeColor(factor.feature_type)} variant="outline">
                                {factor.feature_type}
                              </Badge>
                              {factor.p_value && factor.p_value < 0.05 && (
                                <Badge className="bg-green-100 text-green-700" variant="outline">
                                  显著
                                </Badge>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-semibold text-lg">
                            {(factor.importance_score * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500">重要性分数</div>
                        </div>
                      </div>
                      
                      {/* 重要性条形图 */}
                      <div className="relative">
                        <Progress value={factor.importance_score * 100} className="h-3" />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>0%</span>
                          <span>{(factor.importance_score * 100).toFixed(1)}%</span>
                          <span>100%</span>
                        </div>
                      </div>

                      {/* 技术细节 */}
                      {showTechnicalDetails && (
                        <div className="bg-gray-50 p-3 rounded text-xs grid grid-cols-3 gap-4">
                          <div>
                            <span className="text-gray-600">SHAP值:</span>
                            <div className="font-medium">{factor.shap_value.toFixed(3)}</div>
                          </div>
                          <div>
                            <span className="text-gray-600">贡献度:</span>
                            <div className="font-medium">{(factor.contribution * 100).toFixed(1)}%</div>
                          </div>
                          {factor.p_value && (
                            <div>
                              <span className="text-gray-600">P值:</span>
                              <div className="font-medium">{factor.p_value.toFixed(4)}</div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )) : (
                    <div className="h-40 flex items-center justify-center text-sm text-gray-500">
                      暂无数据
                    </div>
                  )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tree" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>决策树可视化</CardTitle>
              <p className="text-sm text-gray-600">
                展示决策过程的树形结构，每个节点代表一个决策条件
              </p>
            </CardHeader>
            <CardContent>
              {decisionTree ? (
                <div className="w-full h-96 bg-gray-50 rounded-lg p-4 overflow-auto">
                  <svg width="100%" height="100%" viewBox="0 0 600 400">
                    {renderDecisionTreeNode(decisionTree, 300, 60, 500, true)}
                  </svg>
                </div>
              ) : (
                <div className="h-40 flex items-center justify-center text-sm text-gray-500">
                  暂无数据
                </div>
              )}
              
              {showTechnicalDetails && decisionTree && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium text-blue-800 mb-2">决策树统计</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-blue-600">树深度:</span>
                      <div className="font-medium">{treeDepth} 层</div>
                    </div>
                    <div>
                      <span className="text-blue-600">叶子节点:</span>
                      <div className="font-medium">{treeLeaves} 个</div>
                    </div>
                    <div>
                      <span className="text-blue-600">平均置信度:</span>
                      <div className="font-medium">{(treeAvgConfidence * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <span className="text-blue-600">样本覆盖:</span>
                      <div className="font-medium">{decisionTree.samples} 条</div>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="confidence" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>置信度校准图</CardTitle>
              <p className="text-sm text-gray-600">
                评估模型预测置信度与实际准确率的一致性
              </p>
            </CardHeader>
            <CardContent>
              {confidenceDistribution.length > 0 ? (
              <div className="space-y-4">
                {/* 校准曲线 */}
                <div className="h-64 bg-gray-50 rounded-lg p-4">
                  <div className="relative h-full">
                    <svg width="100%" height="100%" viewBox="0 0 400 200">
                      {/* 网格线 */}
                      {[0, 1, 2, 3, 4, 5].map(i => (
                        <g key={i}>
                          <line
                            x1={50 + i * 60}
                            y1={20}
                            x2={50 + i * 60}
                            y2={180}
                            className="stroke-gray-300 stroke-1"
                          />
                          <line
                            x1={50}
                            y1={20 + i * 32}
                            x2={350}
                            y2={20 + i * 32}
                            className="stroke-gray-300 stroke-1"
                          />
                        </g>
                      ))}
                      
                      {/* 理想校准线 */}
                      <line
                        x1={50}
                        y1={180}
                        x2={350}
                        y2={20}
                        className="stroke-gray-400 stroke-2 stroke-dasharray-4"
                      />
                      
                      {/* 实际校准线 */}
                      <polyline
                        points={confidenceDistribution.map((point, i) => {
                          const x = 50 + (i / confidenceDenom) * 300;
                          const y = 180 - (point.accuracy * 160);
                          return `${x},${y}`;
                        }).join(' ')}
                        className="fill-none stroke-blue-500 stroke-3"
                      />
                      
                      {/* 数据点 */}
                      {confidenceDistribution.map((point, i) => {
                        const x = 50 + (i / confidenceDenom) * 300;
                        const y = 180 - (point.accuracy * 160);
                        return (
                          <circle
                            key={i}
                            cx={x}
                            cy={y}
                            r="4"
                            className="fill-blue-500"
                          />
                        );
                      })}
                      
                      {/* 坐标轴标签 */}
                      <text x="200" y="200" textAnchor="middle" className="fill-gray-600 text-xs">
                        预测置信度
                      </text>
                      <text x="20" y="100" textAnchor="middle" className="fill-gray-600 text-xs" transform="rotate(-90 20 100)">
                        实际准确率
                      </text>
                    </svg>
                  </div>
                </div>

                {/* 校准统计 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {confidenceAvgCalibration.toFixed(3)}
                    </div>
                    <div className="text-sm text-blue-700">平均校准分数</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {confidenceTotalSamples}
                    </div>
                    <div className="text-sm text-green-700">总样本数</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">
                      {(confidenceMaxError * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-purple-700">最大校准误差</div>
                  </div>
                </div>

                {/* 分布详情 */}
                {showTechnicalDetails && (
                  <div className="space-y-2">
                    <h4 className="font-medium">置信度区间分布</h4>
                    {confidenceDistribution.map((bin, i) => (
                      <div key={i} className="flex items-center justify-between text-sm p-2 bg-gray-50 rounded">
                        <div className="flex items-center space-x-4">
                          <span className="font-mono">
                            [{(bin.bin_start * 100).toFixed(0)}%, {(bin.bin_end * 100).toFixed(0)}%]
                          </span>
                          <span>样本数: {bin.count}</span>
                        </div>
                        <div className="flex items-center space-x-4">
                          <span>准确率: {(bin.accuracy * 100).toFixed(1)}%</span>
                          <span>校准分数: {bin.calibration_score.toFixed(3)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              ) : (
                <div className="h-40 flex items-center justify-center text-sm text-gray-500">
                  暂无数据
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeseries" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>模型性能时间序列</CardTitle>
              <p className="text-sm text-gray-600">
                展示模型置信度和准确率随时间的变化趋势
              </p>
            </CardHeader>
            <CardContent>
              {timeSeriesData.length > 0 ? (
                <div className="space-y-2">
                  {timeSeriesData.slice(-20).map((item, index) => (
                    <div key={`${item.timestamp}-${index}`} className="flex items-center justify-between text-sm p-2 bg-gray-50 rounded">
                      <span className="text-gray-700">{item.timestamp}</span>
                      <span className="text-gray-700">
                        置信度 {(item.confidence * 100).toFixed(1)}% · 准确率 {(item.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="h-40 flex items-center justify-center text-sm text-gray-500">
                  暂无数据
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="correlation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>因子相关性矩阵</CardTitle>
              <p className="text-sm text-gray-600">
                显示各个特征之间的相关性强度和方向
              </p>
            </CardHeader>
            <CardContent>
              {correlationMatrix.length > 0 ? (
                <div className="space-y-2">
                  {correlationMatrix.slice(0, 50).map((item, index) => (
                    <div key={`${item.factor1}-${item.factor2}-${index}`} className="flex items-center justify-between text-sm p-2 bg-gray-50 rounded">
                      <span className="text-gray-700">{item.factor1} ↔ {item.factor2}</span>
                      <span className="font-mono text-gray-700">{item.correlation.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="h-40 flex items-center justify-center text-sm text-gray-500">
                  暂无数据
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 图表说明 */}
      <Card className="border-l-4 border-l-blue-500">
        <CardContent className="pt-6">
          <div className="flex items-start space-x-3">
            <Info className="h-5 w-5 text-blue-600 mt-0.5" />
            <div>
              <h3 className="font-medium text-blue-800 mb-1">图表使用说明</h3>
              <div className="text-sm text-blue-700 space-y-1">
                <p>• <strong>因素重要性</strong>：基于SHAP值量化各特征对预测结果的贡献</p>
                <p>• <strong>决策树</strong>：可视化模型的决策路径和分支逻辑</p>
                <p>• <strong>置信度校准</strong>：评估预测置信度与实际准确率的匹配程度</p>
                <p>• <strong>时间序列</strong>：监控模型性能指标的时间变化趋势</p>
                <p>• <strong>相关性矩阵</strong>：分析特征间的线性和非线性关系</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ExplanationCharts;
