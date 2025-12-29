import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Brain, 
  Search, 
  Clock, 
  Tag, 
  Database,
  ArrowRight,
  Eye,
  BarChart3,
  Activity,
  Layers,
  Target,
  Zap,
  GitBranch,
  Hash,
  Calendar,
  Star,
  TrendingUp,
  Filter,
  Cpu
} from 'lucide-react';

interface RecalledMemory {
  memory_id: string;
  content_preview: string;
  full_content?: string;
  relevance_score: number;
  memory_type: string;
  importance: number;
  created_at: string;
  last_accessed: string;
  tags: string[];
  embedding_similarity?: number;
  temporal_boost?: number;
  entity_matches?: string[];
}

interface MemoryRecallProcess {
  query_context: string;
  total_memories_searched: number;
  recalled_memories: RecalledMemory[];
  processing_steps: {
    step_name: string;
    step_type: 'vectorization' | 'search' | 'temporal' | 'entity' | 'ranking';
    duration_ms: number;
    memories_processed: number;
    memories_retained: number;
    details: Record<string, any>;
  }[];
  performance_metrics: {
    total_duration: number;
    vector_search_time: number;
    temporal_processing_time: number;
    entity_matching_time: number;
    ranking_time: number;
    cache_hit_rate?: number;
  };
  search_strategy: {
    vector_weight: number;
    temporal_weight: number;
    entity_weight: number;
    importance_weight: number;
  };
}

interface MemoryRecallExplainerProps {
  recallProcess: MemoryRecallProcess;
  showTechnicalDetails?: boolean;
  className?: string;
}

const MemoryRecallExplainer: React.FC<MemoryRecallExplainerProps> = ({
  recallProcess,
  showTechnicalDetails = false,
  className = ''
}) => {
  const [expandedMemory, setExpandedMemory] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'overview' | 'process' | 'memories' | 'analysis'>('overview');
  const [showFullContent, setShowFullContent] = useState<Set<string>>(new Set());

  const getMemoryTypeColor = (type: string) => {
    const colors = {
      'conversation': 'bg-blue-100 text-blue-700 border-blue-200',
      'knowledge': 'bg-green-100 text-green-700 border-green-200',
      'task': 'bg-purple-100 text-purple-700 border-purple-200',
      'context': 'bg-yellow-100 text-yellow-700 border-yellow-200',
      'experience': 'bg-red-100 text-red-700 border-red-200'
    };
    return colors[type as keyof typeof colors] || 'bg-gray-100 text-gray-700 border-gray-200';
  };

  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getStepIcon = (stepType: string) => {
    switch (stepType) {
      case 'vectorization':
        return <Hash className="h-4 w-4" />;
      case 'search':
        return <Search className="h-4 w-4" />;
      case 'temporal':
        return <Clock className="h-4 w-4" />;
      case 'entity':
        return <Tag className="h-4 w-4" />;
      case 'ranking':
        return <TrendingUp className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const toggleFullContent = (memoryId: string) => {
    const newSet = new Set(showFullContent);
    if (newSet.has(memoryId)) {
      newSet.delete(memoryId);
    } else {
      newSet.add(memoryId);
    }
    setShowFullContent(newSet);
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* 记忆召回概览 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-blue-600" />
            <span>记忆召回过程解释</span>
            <Badge variant="outline" className="ml-2">
              {recallProcess.recalled_memories.length} 记忆召回
            </Badge>
          </CardTitle>
          <p className="text-sm text-gray-600">
            查询: "{recallProcess.query_context}"
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {recallProcess.total_memories_searched}
              </div>
              <div className="text-sm text-gray-600">总搜索量</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {recallProcess.recalled_memories.length}
              </div>
              <div className="text-sm text-gray-600">成功召回</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {formatDuration(recallProcess.performance_metrics.total_duration)}
              </div>
              <div className="text-sm text-gray-600">总耗时</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {recallProcess.recalled_memories.length > 0 ? 
                  (recallProcess.recalled_memories.reduce((sum, m) => sum + m.relevance_score, 0) / recallProcess.recalled_memories.length * 100).toFixed(0) : 0}%
              </div>
              <div className="text-sm text-gray-600">平均相关性</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 详细分析 */}
      <Tabs value={selectedView} onValueChange={(value: any) => setSelectedView(value)}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" >
            <Eye className="h-4 w-4 mr-2" />
            概览
          </TabsTrigger>
          <TabsTrigger value="process" >
            <GitBranch className="h-4 w-4 mr-2" />
            处理流程
          </TabsTrigger>
          <TabsTrigger value="memories" >
            <Database className="h-4 w-4 mr-2" />
            召回记忆
          </TabsTrigger>
          <TabsTrigger value="analysis" >
            <BarChart3 className="h-4 w-4 mr-2" />
            分析统计
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 搜索策略 */}
            <Card>
              <CardHeader>
                <CardTitle >
                  <Target className="h-5 w-5 mr-2" />
                  搜索策略配置
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">向量相似度权重</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={recallProcess.search_strategy.vector_weight * 100} 
                        className="w-20 h-2" 
                      />
                      <span className="font-medium">{(recallProcess.search_strategy.vector_weight * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">时间权重</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={recallProcess.search_strategy.temporal_weight * 100} 
                        className="w-20 h-2" 
                      />
                      <span className="font-medium">{(recallProcess.search_strategy.temporal_weight * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">实体匹配权重</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={recallProcess.search_strategy.entity_weight * 100} 
                        className="w-20 h-2" 
                      />
                      <span className="font-medium">{(recallProcess.search_strategy.entity_weight * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">重要性权重</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={recallProcess.search_strategy.importance_weight * 100} 
                        className="w-20 h-2" 
                      />
                      <span className="font-medium">{(recallProcess.search_strategy.importance_weight * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 性能指标 */}
            <Card>
              <CardHeader>
                <CardTitle >
                  <Cpu className="h-5 w-5 mr-2" />
                  性能指标
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">向量搜索耗时:</span>
                    <span className="font-medium">{formatDuration(recallProcess.performance_metrics.vector_search_time)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">时间处理耗时:</span>
                    <span className="font-medium">{formatDuration(recallProcess.performance_metrics.temporal_processing_time)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">实体匹配耗时:</span>
                    <span className="font-medium">{formatDuration(recallProcess.performance_metrics.entity_matching_time)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">排序耗时:</span>
                    <span className="font-medium">{formatDuration(recallProcess.performance_metrics.ranking_time)}</span>
                  </div>
                  {recallProcess.performance_metrics.cache_hit_rate && (
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">缓存命中率:</span>
                      <span className="font-medium">{(recallProcess.performance_metrics.cache_hit_rate * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="process" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>处理流程步骤</CardTitle>
              <p className="text-sm text-gray-600">
                记忆召回过程的详细处理步骤和性能统计
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recallProcess.processing_steps.map((step, index) => {
                  const efficiency = step.memories_retained / step.memories_processed;
                  const isLast = index === recallProcess.processing_steps.length - 1;
                  
                  return (
                    <div key={index} className="relative">
                      {/* 连接线 */}
                      {!isLast && (
                        <div className="absolute left-6 top-12 w-0.5 h-8 bg-gray-300 z-0" />
                      )}
                      
                      <div className="relative z-10 border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center ${
                              step.step_type === 'vectorization' ? 'bg-blue-100 text-blue-600 border-blue-200' :
                              step.step_type === 'search' ? 'bg-green-100 text-green-600 border-green-200' :
                              step.step_type === 'temporal' ? 'bg-yellow-100 text-yellow-600 border-yellow-200' :
                              step.step_type === 'entity' ? 'bg-purple-100 text-purple-600 border-purple-200' :
                              'bg-red-100 text-red-600 border-red-200'
                            }`}>
                              {getStepIcon(step.step_type)}
                            </div>
                            <div>
                              <h4 className="font-semibold">{step.step_name}</h4>
                              <div className="flex items-center space-x-2 text-sm text-gray-600">
                                <Badge variant="outline" className="text-xs">
                                  {step.step_type}
                                </Badge>
                                <span>耗时: {formatDuration(step.duration_ms)}</span>
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm text-gray-600">处理效率</div>
                            <div className="text-lg font-semibold">
                              {(efficiency * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">输入记忆数:</span>
                            <div className="font-medium">{step.memories_processed}</div>
                          </div>
                          <div>
                            <span className="text-gray-600">保留记忆数:</span>
                            <div className="font-medium">{step.memories_retained}</div>
                          </div>
                          <div>
                            <span className="text-gray-600">过滤率:</span>
                            <div className="font-medium">
                              {((1 - efficiency) * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* 处理详情 */}
                        {showTechnicalDetails && Object.keys(step.details).length > 0 && (
                          <div className="mt-3 p-3 bg-gray-50 rounded">
                            <h5 className="font-medium text-gray-800 mb-2">技术细节</h5>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                              {Object.entries(step.details).map(([key, value]) => (
                                <div key={key}>
                                  <span className="text-gray-600">{key}:</span>
                                  <span className="ml-1 font-medium">
                                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* 进度条 */}
                        <div className="mt-3">
                          <Progress value={efficiency * 100} className="h-2" />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="memories" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>召回的记忆详情</CardTitle>
              <p className="text-sm text-gray-600">
                按相关性排序的记忆列表，包含详细的匹配信息
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recallProcess.recalled_memories
                  .sort((a, b) => b.relevance_score - a.relevance_score)
                  .map((memory, index) => {
                    const isExpanded = expandedMemory === memory.memory_id;
                    const showFull = showFullContent.has(memory.memory_id);
                    
                    return (
                      <div 
                        key={memory.memory_id} 
                        className={`border rounded-lg p-4 transition-all ${
                          isExpanded ? 'border-blue-200 bg-blue-50' : 'hover:border-gray-300'
                        }`}
                      >
                        <div 
                          className="cursor-pointer"
                          onClick={() => setExpandedMemory(isExpanded ? null : memory.memory_id)}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-3">
                              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                                <span className="text-blue-600 font-semibold text-sm">{index + 1}</span>
                              </div>
                              <div>
                                <div className="flex items-center space-x-2">
                                  <Badge className={getMemoryTypeColor(memory.memory_type)} variant="outline">
                                    {memory.memory_type}
                                  </Badge>
                                  <span className="text-xs text-gray-500">
                                    ID: {memory.memory_id.slice(0, 8)}...
                                  </span>
                                </div>
                                <p className="text-sm text-gray-700 mt-1">
                                  {showFull && memory.full_content ? memory.full_content : memory.content_preview}
                                </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className={`text-lg font-semibold ${getRelevanceColor(memory.relevance_score)}`}>
                                {(memory.relevance_score * 100).toFixed(1)}%
                              </div>
                              <div className="text-xs text-gray-500">相关性</div>
                            </div>
                          </div>
                        </div>

                        {isExpanded && (
                          <div className="mt-4 space-y-4">
                            {/* 内容控制 */}
                            {memory.full_content && memory.full_content !== memory.content_preview && (
                              <Button
                                onClick={() => toggleFullContent(memory.memory_id)}
                                variant="outline"
                                size="sm"
                              >
                                {showFull ? '显示摘要' : '显示完整内容'}
                              </Button>
                            )}

                            {/* 详细指标 */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                              <div>
                                <div className="text-gray-600">重要性:</div>
                                <div className="flex items-center space-x-2">
                                  <Progress value={memory.importance * 100} className="w-16 h-2" />
                                  <span className="font-medium">{(memory.importance * 100).toFixed(0)}%</span>
                                </div>
                              </div>
                              {memory.embedding_similarity && (
                                <div>
                                  <div className="text-gray-600">向量相似度:</div>
                                  <div className="font-medium">{(memory.embedding_similarity * 100).toFixed(1)}%</div>
                                </div>
                              )}
                              {memory.temporal_boost && (
                                <div>
                                  <div className="text-gray-600">时间加权:</div>
                                  <div className="font-medium">+{(memory.temporal_boost * 100).toFixed(1)}%</div>
                                </div>
                              )}
                              <div>
                                <div className="text-gray-600">最后访问:</div>
                                <div className="font-medium text-xs">
                                  {new Date(memory.last_accessed).toLocaleDateString()}
                                </div>
                              </div>
                            </div>

                            {/* 标签和实体匹配 */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {memory.tags.length > 0 && (
                                <div>
                                  <h5 className="font-medium mb-2 flex items-center">
                                    <Tag className="h-4 w-4 mr-1" />
                                    标签
                                  </h5>
                                  <div className="flex flex-wrap gap-1">
                                    {memory.tags.map((tag, i) => (
                                      <Badge key={i} variant="outline" className="text-xs">
                                        {tag}
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {memory.entity_matches && memory.entity_matches.length > 0 && (
                                <div>
                                  <h5 className="font-medium mb-2 flex items-center">
                                    <Target className="h-4 w-4 mr-1" />
                                    实体匹配
                                  </h5>
                                  <div className="flex flex-wrap gap-1">
                                    {memory.entity_matches.map((entity, i) => (
                                      <Badge key={i} variant="outline" className="text-xs bg-green-50 text-green-700">
                                        {entity}
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* 时间信息 */}
                            <div className="bg-gray-50 p-3 rounded">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                                <div className="flex items-center space-x-2">
                                  <Calendar className="h-3 w-3 text-gray-500" />
                                  <span className="text-gray-600">创建时间:</span>
                                  <span className="font-medium">{new Date(memory.created_at).toLocaleString()}</span>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <Clock className="h-3 w-3 text-gray-500" />
                                  <span className="text-gray-600">最后访问:</span>
                                  <span className="font-medium">{new Date(memory.last_accessed).toLocaleString()}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 记忆类型分布 */}
            <Card>
              <CardHeader>
                <CardTitle >
                  <Layers className="h-5 w-5 mr-2" />
                  记忆类型分布
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(
                    recallProcess.recalled_memories.reduce((acc, memory) => {
                      acc[memory.memory_type] = (acc[memory.memory_type] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).map(([type, count]) => {
                    const percentage = (count / recallProcess.recalled_memories.length) * 100;
                    return (
                      <div key={type}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium">{type}</span>
                          <span>{count} ({percentage.toFixed(1)}%)</span>
                        </div>
                        <Progress value={percentage} className="h-2" />
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* 相关性分布 */}
            <Card>
              <CardHeader>
                <CardTitle >
                  <BarChart3 className="h-5 w-5 mr-2" />
                  相关性分布
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { label: '高相关性 (≥80%)', min: 0.8, color: 'text-green-600' },
                    { label: '中相关性 (60-79%)', min: 0.6, color: 'text-yellow-600' },
                    { label: '低相关性 (<60%)', min: 0, color: 'text-red-600' }
                  ].map(({ label, min, color }) => {
                    const count = recallProcess.recalled_memories.filter(m => 
                      min === 0.8 ? m.relevance_score >= min :
                      min === 0.6 ? m.relevance_score >= min && m.relevance_score < 0.8 :
                      m.relevance_score < 0.6
                    ).length;
                    const percentage = (count / recallProcess.recalled_memories.length) * 100;
                    
                    return (
                      <div key={label}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium">{label}</span>
                          <span className={color}>{count} ({percentage.toFixed(1)}%)</span>
                        </div>
                        <Progress value={percentage} className="h-2" />
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 召回质量分析 */}
          <Card>
            <CardHeader>
              <CardTitle >
                <Star className="h-5 w-5 mr-2" />
                召回质量分析
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {(recallProcess.recalled_memories.reduce((sum, m) => sum + m.relevance_score, 0) / recallProcess.recalled_memories.length * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-blue-700">平均相关性</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {(recallProcess.recalled_memories.reduce((sum, m) => sum + m.importance, 0) / recallProcess.recalled_memories.length * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-green-700">平均重要性</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {new Set(recallProcess.recalled_memories.flatMap(m => m.tags)).size}
                  </div>
                  <div className="text-sm text-purple-700">覆盖标签数</div>
                </div>
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">
                    {((recallProcess.recalled_memories.length / recallProcess.total_memories_searched) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-orange-700">召回率</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MemoryRecallExplainer;