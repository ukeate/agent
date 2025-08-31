import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Alert } from '../components/ui/alert';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/Tabs';
import { banditRecommendationService } from '../services/banditRecommendationService';
import RecommendationEngineMonitor from '../components/bandit/RecommendationEngineMonitor';
import AlgorithmVisualization from '../components/bandit/AlgorithmVisualization';
import AlgorithmTester from '../components/bandit/AlgorithmTester';

interface Recommendation {
  item_id: string;
  score: number;
  confidence: number;
}

interface RecommendationResponse {
  request_id: string;
  user_id: string;
  recommendations: Recommendation[];
  algorithm_used: string;
  confidence_score: number;
  cold_start_strategy?: string;
  explanations?: string[];
  processing_time_ms: number;
}

interface EngineStats {
  engine_stats: {
    total_requests: number;
    cache_hits: number;
    cold_start_requests: number;
    algorithm_usage: Record<string, number>;
    average_response_time_ms: number;
  };
  algorithm_stats: Record<string, any>;
  active_users: number;
  cache_size: number;
}

interface AlgorithmInfo {
  name: string;
  display_name: string;
  description: string;
  supports_context: boolean;
  supports_binary_feedback: boolean;
}

const BanditRecommendationPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('initialize');
  const [isInitialized, setIsInitialized] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // 初始化状态
  const [nItems, setNItems] = useState(100);
  const [enableColdStart, setEnableColdStart] = useState(true);
  const [enableEvaluation, setEnableEvaluation] = useState(true);
  
  // 推荐状态
  const [userId, setUserId] = useState('user_001');
  const [numRecommendations, setNumRecommendations] = useState(5);
  const [includeExplanations, setIncludeExplanations] = useState(true);
  const [context, setContext] = useState('{"age": 25, "location": "Beijing"}');
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  
  // 反馈状态
  const [feedbackUserId, setFeedbackUserId] = useState('user_001');
  const [feedbackItemId, setFeedbackItemId] = useState('');
  const [feedbackType, setFeedbackType] = useState('click');
  const [feedbackValue, setFeedbackValue] = useState(1.0);
  
  // 统计状态
  const [engineStats, setEngineStats] = useState<EngineStats | null>(null);
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  
  // 清除消息
  const clearMessages = () => {
    setError(null);
    setSuccess(null);
  };

  // 初始化引擎
  const handleInitialize = async () => {
    setLoading(true);
    clearMessages();
    
    try {
      const response = await banditRecommendationService.initialize(nItems, {
        enable_cold_start: enableColdStart,
        enable_evaluation: enableEvaluation
      });
      
      setIsInitialized(true);
      setSuccess(`推荐引擎初始化成功！支持 ${nItems} 个物品`);
      
      // 加载可用算法
      await loadAlgorithms();
      
    } catch (err: any) {
      setError(`初始化失败: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 获取推荐
  const handleGetRecommendations = async () => {
    if (!isInitialized) {
      setError('请先初始化推荐引擎');
      return;
    }
    
    setLoading(true);
    clearMessages();
    
    try {
      let parsedContext = null;
      if (context.trim()) {
        try {
          parsedContext = JSON.parse(context);
        } catch (e) {
          throw new Error('上下文格式错误，请输入有效的JSON');
        }
      }
      
      const response = await banditRecommendationService.getRecommendations({
        user_id: userId,
        num_recommendations: numRecommendations,
        context: parsedContext,
        include_explanations: includeExplanations
      });
      
      setRecommendations(response);
      setSuccess(`成功获取 ${response.recommendations.length} 个推荐`);
      
    } catch (err: any) {
      setError(`获取推荐失败: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 提交反馈
  const handleSubmitFeedback = async () => {
    if (!isInitialized) {
      setError('请先初始化推荐引擎');
      return;
    }
    
    setLoading(true);
    clearMessages();
    
    try {
      await banditRecommendationService.submitFeedback({
        user_id: feedbackUserId,
        item_id: feedbackItemId,
        feedback_type: feedbackType,
        feedback_value: feedbackValue
      });
      
      setSuccess('反馈提交成功！');
      
    } catch (err: any) {
      setError(`提交反馈失败: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 获取统计信息
  const handleGetStats = async () => {
    if (!isInitialized) {
      setError('请先初始化推荐引擎');
      return;
    }
    
    try {
      const response = await banditRecommendationService.getStatistics();
      setEngineStats(response.statistics);
      setSuccess('统计信息已更新');
    } catch (err: any) {
      setError(`获取统计信息失败: ${err.message}`);
    }
  };

  // 加载算法列表
  const loadAlgorithms = async () => {
    try {
      const response = await banditRecommendationService.getAlgorithms();
      setAlgorithms(response.algorithms);
    } catch (err: any) {
      console.error('加载算法列表失败:', err);
    }
  };

  // 页面加载时检查健康状态
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await banditRecommendationService.getHealth();
        if (health.is_initialized) {
          setIsInitialized(true);
          await loadAlgorithms();
        }
      } catch (err) {
        console.error('健康检查失败:', err);
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">多臂老虎机推荐引擎</h1>
        <p className="text-gray-600">
          学习和演示多臂老虎机算法在推荐系统中的应用
        </p>
      </div>

      {/* 错误和成功消息 */}
      {error && (
        <Alert className="border-red-200 bg-red-50">
          <div className="text-red-800">{error}</div>
        </Alert>
      )}
      
      {success && (
        <Alert className="border-green-200 bg-green-50">
          <div className="text-green-800">{success}</div>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="initialize">初始化引擎</TabsTrigger>
          <TabsTrigger value="recommend">获取推荐</TabsTrigger>
          <TabsTrigger value="feedback">提交反馈</TabsTrigger>
          <TabsTrigger value="stats">统计分析</TabsTrigger>
          <TabsTrigger value="algorithms">算法介绍</TabsTrigger>
        </TabsList>

        {/* 初始化引擎 */}
        <TabsContent value="initialize">
          <Card>
            <CardHeader>
              <CardTitle>引擎初始化</CardTitle>
              <CardDescription>
                初始化多臂老虎机推荐引擎，配置物品数量和功能选项
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">物品数量</label>
                  <Input
                    type="number"
                    value={nItems}
                    onChange={(e) => setNItems(Number(e.target.value))}
                    min="10"
                    max="10000"
                    className="w-full"
                  />
                  <p className="text-xs text-gray-500 mt-1">推荐系统中的物品总数</p>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="coldStart"
                    checked={enableColdStart}
                    onChange={(e) => setEnableColdStart(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="coldStart" className="text-sm">
                    启用冷启动策略
                  </label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="evaluation"
                    checked={enableEvaluation}
                    onChange={(e) => setEnableEvaluation(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="evaluation" className="text-sm">
                    启用评估功能
                  </label>
                </div>
              </div>
              
              <Button 
                onClick={handleInitialize}
                disabled={loading || isInitialized}
                className="w-full"
              >
                {loading ? '初始化中...' : isInitialized ? '已初始化' : '初始化引擎'}
              </Button>
              
              {isInitialized && (
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span className="text-green-800 font-medium">引擎已就绪</span>
                  </div>
                  <p className="text-green-700 text-sm mt-1">
                    支持 {nItems} 个物品，冷启动: {enableColdStart ? '已启用' : '已禁用'}，
                    评估: {enableEvaluation ? '已启用' : '已禁用'}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* 获取推荐 */}
        <TabsContent value="recommend">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 推荐请求 */}
            <Card>
              <CardHeader>
                <CardTitle>推荐请求</CardTitle>
                <CardDescription>
                  配置用户信息和上下文，获取个性化推荐
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">用户ID</label>
                  <Input
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    placeholder="输入用户ID"
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">推荐数量</label>
                  <Input
                    type="number"
                    value={numRecommendations}
                    onChange={(e) => setNumRecommendations(Number(e.target.value))}
                    min="1"
                    max="20"
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">
                    用户上下文 (JSON格式)
                  </label>
                  <textarea
                    value={context}
                    onChange={(e) => setContext(e.target.value)}
                    placeholder='{"age": 25, "location": "Beijing", "interests": ["tech", "sports"]}'
                    rows={3}
                    className="w-full p-2 border border-gray-300 rounded-md resize-none font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    用于上下文感知算法的特征信息
                  </p>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="explanations"
                    checked={includeExplanations}
                    onChange={(e) => setIncludeExplanations(e.target.checked)}
                    className="rounded"
                  />
                  <label htmlFor="explanations" className="text-sm">
                    包含推荐解释
                  </label>
                </div>
                
                <Button 
                  onClick={handleGetRecommendations}
                  disabled={loading || !isInitialized}
                  className="w-full"
                >
                  {loading ? '生成推荐中...' : '获取推荐'}
                </Button>
              </CardContent>
            </Card>

            {/* 推荐结果 */}
            <Card>
              <CardHeader>
                <CardTitle>推荐结果</CardTitle>
                <CardDescription>
                  展示推荐物品和算法决策过程
                </CardDescription>
              </CardHeader>
              <CardContent>
                {recommendations ? (
                  <div className="space-y-4">
                    {/* 元信息 */}
                    <div className="bg-gray-50 p-4 rounded-lg space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">算法:</span>
                        <Badge variant="outline">{recommendations.algorithm_used}</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">整体置信度:</span>
                        <span className="text-sm">{recommendations.confidence_score?.toFixed(3) || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">处理时间:</span>
                        <span className="text-sm">{recommendations.processing_time_ms?.toFixed(2) || 'N/A'}ms</span>
                      </div>
                      {recommendations.cold_start_strategy && (
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">冷启动策略:</span>
                          <Badge variant="secondary">{recommendations.cold_start_strategy}</Badge>
                        </div>
                      )}
                    </div>

                    {/* 推荐物品列表 */}
                    <div className="space-y-2">
                      <h4 className="font-medium">推荐物品</h4>
                      {recommendations.recommendations.map((rec, index) => (
                        <div key={rec.item_id} className="border border-gray-200 p-3 rounded-lg">
                          <div className="flex justify-between items-start">
                            <div>
                              <span className="font-medium">#{index + 1} 物品 {rec.item_id}</span>
                              <div className="text-sm text-gray-600 mt-1">
                                分数: {rec.score?.toFixed(3) || 'N/A'} | 置信度: {rec.confidence?.toFixed(3) || 'N/A'}
                              </div>
                            </div>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => {
                                setFeedbackUserId(userId);
                                setFeedbackItemId(rec.item_id);
                              }}
                            >
                              选择反馈
                            </Button>
                          </div>
                          
                          {recommendations.explanations && recommendations.explanations[index] && (
                            <div className="mt-2 p-2 bg-blue-50 rounded text-sm text-blue-800">
                              <strong>解释:</strong> {recommendations.explanations[index]}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-gray-500 text-center py-8">
                    暂无推荐结果，请先获取推荐
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* 提交反馈 */}
        <TabsContent value="feedback">
          <Card>
            <CardHeader>
              <CardTitle>用户反馈</CardTitle>
              <CardDescription>
                提交用户对推荐物品的反馈，帮助算法学习和优化
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">用户ID</label>
                  <Input
                    value={feedbackUserId}
                    onChange={(e) => setFeedbackUserId(e.target.value)}
                    placeholder="输入用户ID"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">物品ID</label>
                  <Input
                    value={feedbackItemId}
                    onChange={(e) => setFeedbackItemId(e.target.value)}
                    placeholder="输入物品ID"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">反馈类型</label>
                  <select
                    value={feedbackType}
                    onChange={(e) => setFeedbackType(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-md"
                  >
                    <option value="view">浏览</option>
                    <option value="click">点击</option>
                    <option value="like">喜欢</option>
                    <option value="share">分享</option>
                    <option value="purchase">购买</option>
                    <option value="rating">评分</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1">反馈值 (0-5)</label>
                  <Input
                    type="number"
                    value={feedbackValue}
                    onChange={(e) => setFeedbackValue(Number(e.target.value))}
                    min="0"
                    max="5"
                    step="0.1"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    数值越高表示用户越喜欢该物品
                  </p>
                </div>
              </div>
              
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-medium text-blue-800 mb-2">反馈类型说明</h4>
                <div className="text-sm text-blue-700 space-y-1">
                  <div>• <strong>浏览:</strong> 用户查看了物品 (奖励值: 0.1)</div>
                  <div>• <strong>点击:</strong> 用户点击了物品 (奖励值: 0.3)</div>
                  <div>• <strong>喜欢:</strong> 用户标记喜欢 (奖励值: 0.6)</div>
                  <div>• <strong>分享:</strong> 用户分享了物品 (奖励值: 0.8)</div>
                  <div>• <strong>购买:</strong> 用户购买了物品 (奖励值: 1.0)</div>
                  <div>• <strong>评分:</strong> 用户给出评分 (奖励值: 评分/5)</div>
                </div>
              </div>
              
              <Button 
                onClick={handleSubmitFeedback}
                disabled={loading || !isInitialized || !feedbackItemId}
                className="w-full"
              >
                {loading ? '提交中...' : '提交反馈'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 统计分析 */}
        <TabsContent value="stats">
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-bold">引擎统计信息</h2>
              <Button onClick={handleGetStats} disabled={loading || !isInitialized}>
                {loading ? '刷新中...' : '刷新数据'}
              </Button>
            </div>

            {/* 引擎状态监控 */}
            <RecommendationEngineMonitor refreshInterval={10000} />

            {engineStats ? (
              <div className="space-y-6">
                {/* 算法性能可视化 */}
                {engineStats.algorithm_stats && Object.keys(engineStats.algorithm_stats).length > 0 && (
                  <AlgorithmVisualization
                    data={Object.entries(engineStats.algorithm_stats).map(([algorithm, stats]: [string, any]) => ({
                      algorithm,
                      average_reward: stats.average_reward || 0,
                      total_pulls: stats.total_pulls || 0,
                      regret: stats.regret || 0,
                      usage_count: engineStats.engine_stats.algorithm_usage[algorithm] || 0
                    }))}
                    title="算法性能实时对比"
                    width={800}
                    height={400}
                  />
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* 基础统计 */}
                  <Card>
                    <CardHeader>
                      <CardTitle>基础指标</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span>总请求数:</span>
                          <Badge variant="secondary">{engineStats.engine_stats.total_requests}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>缓存命中数:</span>
                          <Badge variant="secondary">{engineStats.engine_stats.cache_hits}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>冷启动请求:</span>
                          <Badge variant="secondary">{engineStats.engine_stats.cold_start_requests}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>活跃用户数:</span>
                          <Badge variant="secondary">{engineStats.active_users}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>缓存大小:</span>
                          <Badge variant="secondary">{engineStats.cache_size}</Badge>
                        </div>
                        <div className="flex justify-between">
                          <span>平均响应时间:</span>
                          <Badge variant="secondary">
                            {engineStats.engine_stats.average_response_time_ms?.toFixed(2) || 'N/A'}ms
                          </Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* 算法使用统计 */}
                  <Card>
                    <CardHeader>
                      <CardTitle>算法使用统计</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {Object.entries(engineStats.engine_stats.algorithm_usage).map(([algorithm, count]) => (
                          <div key={algorithm} className="flex justify-between items-center">
                            <span className="capitalize">{algorithm.replace('_', ' ')}:</span>
                            <div className="flex items-center space-x-2">
                              <Badge variant="outline">{count}</Badge>
                              <div className="w-20 bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-blue-500 h-2 rounded-full"
                                  style={{
                                    width: `${(count / engineStats.engine_stats.total_requests * 100)}%`
                                  }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* 算法性能 */}
                  <Card className="md:col-span-2">
                    <CardHeader>
                      <CardTitle>算法性能指标</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {Object.entries(engineStats.algorithm_stats).map(([algorithm, stats]: [string, any]) => (
                          <div key={algorithm} className="border border-gray-200 p-4 rounded-lg">
                            <h4 className="font-medium capitalize mb-3">
                              {algorithm.replace('_', ' ')}
                            </h4>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span>平均奖励:</span>
                                <span>{stats.average_reward?.toFixed(3) || 'N/A'}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>累积奖励:</span>
                                <span>{stats.total_reward?.toFixed(1) || 'N/A'}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>选择次数:</span>
                                <span>{stats.total_pulls || 'N/A'}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>累积遗憾:</span>
                                <span>{stats.regret?.toFixed(3) || 'N/A'}</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* 算法测试工具 */}
                <AlgorithmTester 
                  onTestComplete={(results) => {
                    console.log('测试完成:', results);
                    // 测试完成后刷新统计数据
                    handleGetStats();
                  }}
                />
              </div>
            ) : (
              <div className="text-gray-500 text-center py-8">
                暂无统计数据，请先获取统计信息
              </div>
            )}
          </div>
        </TabsContent>

        {/* 算法介绍 */}
        <TabsContent value="algorithms">
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-xl font-bold mb-2">多臂老虎机算法介绍</h2>
              <p className="text-gray-600">
                了解不同算法的特点和适用场景
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {algorithms.map((algorithm) => (
                <Card key={algorithm.name} className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      {algorithm.display_name}
                      <div className="flex space-x-1">
                        {algorithm.supports_context && (
                          <Badge variant="outline" className="text-xs">上下文</Badge>
                        )}
                        {algorithm.supports_binary_feedback && (
                          <Badge variant="outline" className="text-xs">二元反馈</Badge>
                        )}
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600 mb-4">{algorithm.description}</p>
                    
                    {/* 算法特点 */}
                    <div className="space-y-2">
                      <h4 className="font-medium text-sm">算法特点:</h4>
                      <div className="text-sm text-gray-600">
                        {algorithm.name === 'ucb' && (
                          <ul className="list-disc list-inside space-y-1">
                            <li>基于置信区间上界的决策</li>
                            <li>理论保证最优的遗憾界</li>
                            <li>适合探索与利用平衡</li>
                            <li>收敛速度较快</li>
                          </ul>
                        )}
                        {algorithm.name === 'thompson_sampling' && (
                          <ul className="list-disc list-inside space-y-1">
                            <li>贝叶斯方法，使用后验分布</li>
                            <li>自然的随机性探索</li>
                            <li>适合二元反馈场景</li>
                            <li>计算效率高</li>
                          </ul>
                        )}
                        {algorithm.name === 'epsilon_greedy' && (
                          <ul className="list-disc list-inside space-y-1">
                            <li>简单直观的探索策略</li>
                            <li>支持ε衰减</li>
                            <li>实现简单，易于理解</li>
                            <li>适合入门学习</li>
                          </ul>
                        )}
                        {algorithm.name === 'linear_contextual' && (
                          <ul className="list-disc list-inside space-y-1">
                            <li>利用用户和物品特征</li>
                            <li>支持个性化推荐</li>
                            <li>基于线性回归模型</li>
                            <li>适合复杂上下文场景</li>
                          </ul>
                        )}
                      </div>
                    </div>

                    {/* 适用场景 */}
                    <div className="mt-4 space-y-2">
                      <h4 className="font-medium text-sm">适用场景:</h4>
                      <div className="text-sm text-gray-600">
                        {algorithm.name === 'ucb' && '新闻推荐、广告投放、A/B测试'}
                        {algorithm.name === 'thompson_sampling' && '点击率预测、转化率优化'}
                        {algorithm.name === 'epsilon_greedy' && '快速原型、基准对比'}
                        {algorithm.name === 'linear_contextual' && '个性化推荐、内容匹配'}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* 多臂老虎机概念解释 */}
            <Card>
              <CardHeader>
                <CardTitle>多臂老虎机问题</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-gray-600">
                  多臂老虎机（Multi-Armed Bandit, MAB）是强化学习中的经典问题，
                  模拟在多个选择中寻找最优策略的场景。
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-medium text-blue-800 mb-2">探索 vs 利用</h4>
                    <p className="text-sm text-blue-700">
                      平衡尝试新选择（探索）和选择已知最好选择（利用）之间的关系
                    </p>
                  </div>
                  
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h4 className="font-medium text-green-800 mb-2">在线学习</h4>
                    <p className="text-sm text-green-700">
                      算法在获得反馈时实时调整策略，无需预先训练
                    </p>
                  </div>
                  
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <h4 className="font-medium text-purple-800 mb-2">遗憾最小化</h4>
                    <p className="text-sm text-purple-700">
                      目标是最小化累积遗憾，即与最优选择的差距
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BanditRecommendationPage;