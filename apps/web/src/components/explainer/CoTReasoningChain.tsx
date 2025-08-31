import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  ChevronDown, 
  ChevronRight, 
  Brain, 
  ArrowDown, 
  CheckCircle, 
  AlertCircle,
  Target,
  Lightbulb,
  Search,
  Zap
} from 'lucide-react';

interface ReasoningStep {
  step_id: string;
  step_type: 'observation' | 'thought' | 'action' | 'reflection' | 'conclusion';
  content: string;
  confidence: number;
  dependencies: string[];
  evidence: string[];
  duration_ms: number;
  metadata: {
    reasoning_strategy?: string;
    evidence_quality?: number;
    logical_validity?: number;
  };
}

interface ReasoningChain {
  chain_id: string;
  reasoning_mode: 'analytical' | 'deductive' | 'inductive' | 'abductive';
  steps: ReasoningStep[];
  overall_confidence: number;
  logical_consistency: number;
  evidence_quality: number;
  created_at: string;
}

interface CoTReasoningChainProps {
  chain: ReasoningChain;
  className?: string;
}

const CoTReasoningChain: React.FC<CoTReasoningChainProps> = ({ chain, className = '' }) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [showMetadata, setShowMetadata] = useState(false);

  const toggleStep = (stepId: string) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    setExpandedSteps(newExpanded);
  };

  const getStepIcon = (stepType: string) => {
    switch (stepType) {
      case 'observation':
        return <Search className="h-4 w-4" />;
      case 'thought':
        return <Brain className="h-4 w-4" />;
      case 'action':
        return <Zap className="h-4 w-4" />;
      case 'reflection':
        return <Lightbulb className="h-4 w-4" />;
      case 'conclusion':
        return <Target className="h-4 w-4" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  const getStepColor = (stepType: string) => {
    switch (stepType) {
      case 'observation':
        return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'thought':
        return 'bg-purple-100 text-purple-700 border-purple-200';
      case 'action':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'reflection':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'conclusion':
        return 'bg-red-100 text-red-700 border-red-200';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getModeDescription = (mode: string) => {
    switch (mode) {
      case 'analytical':
        return '分析性推理 - 逐步分解复杂问题';
      case 'deductive':
        return '演绎推理 - 从一般原理推导具体结论';
      case 'inductive':
        return '归纳推理 - 从具体观察得出一般规律';
      case 'abductive':
        return '溯因推理 - 寻找最佳解释';
      default:
        return '未知推理模式';
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* 推理链概览 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-purple-600" />
              <span>Chain-of-Thought 推理链</span>
              <Badge variant="outline" className="ml-2">
                {chain.reasoning_mode}
              </Badge>
            </CardTitle>
            <Button
              onClick={() => setShowMetadata(!showMetadata)}
              variant="outline"
              size="sm"
            >
              {showMetadata ? '隐藏' : '显示'}元数据
            </Button>
          </div>
          <p className="text-sm text-gray-600">{getModeDescription(chain.reasoning_mode)}</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-600">整体置信度</div>
              <div className={`text-lg font-semibold ${getConfidenceColor(chain.overall_confidence)}`}>
                {(chain.overall_confidence * 100).toFixed(1)}%
              </div>
              <Progress value={chain.overall_confidence * 100} className="h-2 mt-1" />
            </div>
            <div>
              <div className="text-sm text-gray-600">逻辑一致性</div>
              <div className={`text-lg font-semibold ${getConfidenceColor(chain.logical_consistency)}`}>
                {(chain.logical_consistency * 100).toFixed(1)}%
              </div>
              <Progress value={chain.logical_consistency * 100} className="h-2 mt-1" />
            </div>
            <div>
              <div className="text-sm text-gray-600">证据质量</div>
              <div className={`text-lg font-semibold ${getConfidenceColor(chain.evidence_quality)}`}>
                {(chain.evidence_quality * 100).toFixed(1)}%
              </div>
              <Progress value={chain.evidence_quality * 100} className="h-2 mt-1" />
            </div>
          </div>

          {showMetadata && (
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">推理链ID:</span> {chain.chain_id}
                </div>
                <div>
                  <span className="font-medium">步骤数量:</span> {chain.steps.length}
                </div>
                <div>
                  <span className="font-medium">创建时间:</span> {new Date(chain.created_at).toLocaleString()}
                </div>
                <div>
                  <span className="font-medium">总耗时:</span> {chain.steps.reduce((sum, step) => sum + step.duration_ms, 0)}ms
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 推理步骤 */}
      <div className="space-y-3">
        <h3 className="font-semibold text-lg">推理步骤</h3>
        {chain.steps.map((step, index) => {
          const isExpanded = expandedSteps.has(step.step_id);
          const isLast = index === chain.steps.length - 1;
          
          return (
            <div key={step.step_id} className="relative">
              {/* 连接线 */}
              {!isLast && (
                <div className="absolute left-6 top-12 w-0.5 h-6 bg-gray-300 z-0" />
              )}
              
              <Card className="relative z-10">
                <CardHeader 
                  className="cursor-pointer hover:bg-gray-50 transition-colors"
                  onClick={() => toggleStep(step.step_id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center ${getStepColor(step.step_type)}`}>
                        {getStepIcon(step.step_type)}
                      </div>
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">步骤 {index + 1}</span>
                          <Badge variant="outline" className="text-xs">
                            {step.step_type}
                          </Badge>
                          <span className={`text-sm font-medium ${getConfidenceColor(step.confidence)}`}>
                            {(step.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                          {step.content}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-500">
                        {step.duration_ms}ms
                      </span>
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4 text-gray-400" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-gray-400" />
                      )}
                    </div>
                  </div>
                </CardHeader>

                {isExpanded && (
                  <CardContent className="pt-0">
                    <div className="space-y-4">
                      {/* 完整内容 */}
                      <div>
                        <h4 className="font-medium text-sm mb-2">推理内容</h4>
                        <div className="bg-gray-50 p-3 rounded-lg">
                          <p className="text-sm text-gray-700">{step.content}</p>
                        </div>
                      </div>

                      {/* 证据 */}
                      {step.evidence.length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm mb-2">支持证据</h4>
                          <div className="space-y-2">
                            {step.evidence.map((evidence, evidenceIndex) => (
                              <div key={evidenceIndex} className="flex items-start space-x-2 text-sm">
                                <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                                <span className="text-gray-700">{evidence}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* 依赖关系 */}
                      {step.dependencies.length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm mb-2">依赖步骤</h4>
                          <div className="flex flex-wrap gap-2">
                            {step.dependencies.map((dep, depIndex) => (
                              <Badge key={depIndex} variant="outline" className="text-xs">
                                {dep}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* 元数据 */}
                      {step.metadata && Object.keys(step.metadata).length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm mb-2">技术指标</h4>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                            {step.metadata.reasoning_strategy && (
                              <div>
                                <span className="text-gray-600">推理策略:</span>
                                <div className="font-medium">{step.metadata.reasoning_strategy}</div>
                              </div>
                            )}
                            {step.metadata.evidence_quality && (
                              <div>
                                <span className="text-gray-600">证据质量:</span>
                                <div className="font-medium">{(step.metadata.evidence_quality * 100).toFixed(1)}%</div>
                              </div>
                            )}
                            {step.metadata.logical_validity && (
                              <div>
                                <span className="text-gray-600">逻辑有效性:</span>
                                <div className="font-medium">{(step.metadata.logical_validity * 100).toFixed(1)}%</div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                )}
              </Card>
            </div>
          );
        })}
      </div>

      {/* 推理链总结 */}
      <Card className="border-l-4 border-l-purple-500">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5 text-purple-600" />
            <span>推理链分析</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2">强项分析</h4>
              <ul className="space-y-1 text-sm">
                {chain.logical_consistency > 0.8 && (
                  <li className="flex items-center space-x-2 text-green-700">
                    <CheckCircle className="h-4 w-4" />
                    <span>逻辑推理一致性很高</span>
                  </li>
                )}
                {chain.evidence_quality > 0.7 && (
                  <li className="flex items-center space-x-2 text-green-700">
                    <CheckCircle className="h-4 w-4" />
                    <span>证据质量良好</span>
                  </li>
                )}
                {chain.steps.length >= 3 && (
                  <li className="flex items-center space-x-2 text-green-700">
                    <CheckCircle className="h-4 w-4" />
                    <span>推理步骤充分详细</span>
                  </li>
                )}
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">改进建议</h4>
              <ul className="space-y-1 text-sm">
                {chain.logical_consistency < 0.7 && (
                  <li className="flex items-center space-x-2 text-orange-700">
                    <AlertCircle className="h-4 w-4" />
                    <span>可以加强逻辑推理的一致性</span>
                  </li>
                )}
                {chain.evidence_quality < 0.6 && (
                  <li className="flex items-center space-x-2 text-orange-700">
                    <AlertCircle className="h-4 w-4" />
                    <span>建议提供更高质量的证据</span>
                  </li>
                )}
                {chain.steps.length < 3 && (
                  <li className="flex items-center space-x-2 text-orange-700">
                    <AlertCircle className="h-4 w-4" />
                    <span>推理步骤可以更加详细</span>
                  </li>
                )}
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CoTReasoningChain;