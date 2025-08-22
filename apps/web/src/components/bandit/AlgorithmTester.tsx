import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Alert } from '../ui/Alert';
import { Badge } from '../ui/Badge';
import { banditRecommendationService } from '../../services/banditRecommendationService';

interface TestSession {
  sessionId: string;
  userId: string;
  numInteractions: number;
  status: 'idle' | 'running' | 'completed' | 'error';
  progress: number;
  results: {
    recommendations: any[];
    feedbacks: any[];
    totalReward: number;
    comparisonData?: any[];
  } | null;
  startTime?: Date;
  endTime?: Date;
}

interface AlgorithmTesterProps {
  onTestComplete?: (results: any) => void;
}

const AlgorithmTester: React.FC<AlgorithmTesterProps> = ({ onTestComplete }) => {
  const [testConfig, setTestConfig] = useState({
    userId: 'test_user_performance',
    numInteractions: 20,
    includeContext: true,
    algorithmType: 'ucb'
  });

  const [currentSession, setCurrentSession] = useState<TestSession | null>(null);
  const [testHistory, setTestHistory] = useState<TestSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 运行性能测试
  const runPerformanceTest = async () => {
    if (!currentSession || currentSession.status !== 'idle') {
      const newSession: TestSession = {
        sessionId: `test_${Date.now()}`,
        userId: testConfig.userId,
        numInteractions: testConfig.numInteractions,
        status: 'idle',
        progress: 0,
        results: null
      };
      setCurrentSession(newSession);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const session = { ...currentSession, status: 'running' as const, startTime: new Date() };
      setCurrentSession(session);

      // 上下文生成器
      const contextGenerator = testConfig.includeContext ? () => ({
        age: 20 + Math.floor(Math.random() * 40),
        location: ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'][Math.floor(Math.random() * 4)],
        interests: ['tech', 'sports', 'music', 'travel'][Math.floor(Math.random() * 4)],
        session_time: Date.now()
      }) : undefined;

      // 运行模拟会话
      const results = await banditRecommendationService.simulateUserSession(
        testConfig.userId,
        testConfig.numInteractions,
        contextGenerator
      );

      // 完成测试
      const completedSession: TestSession = {
        ...session,
        status: 'completed',
        progress: 100,
        results,
        endTime: new Date()
      };

      setCurrentSession(completedSession);
      setTestHistory(prev => [...prev, completedSession]);

      if (onTestComplete) {
        onTestComplete(results);
      }

    } catch (err: any) {
      setError(`测试失败: ${err.message}`);
      setCurrentSession(prev => prev ? { ...prev, status: 'error' } : null);
    } finally {
      setLoading(false);
    }
  };

  // 批量对比测试
  const runComparisonTest = async () => {
    setLoading(true);
    setError(null);

    const algorithms = ['ucb', 'thompson_sampling', 'epsilon_greedy'];
    const comparisonResults: any[] = [];

    try {
      for (const algorithm of algorithms) {
        // 为每个算法运行测试
        const userId = `${algorithm}_test_${Date.now()}`;
        
        const results = await banditRecommendationService.simulateUserSession(
          userId,
          testConfig.numInteractions,
          testConfig.includeContext ? () => ({
            age: 25,
            location: 'Beijing',
            interests: 'tech'
          }) : undefined
        );

        comparisonResults.push({
          algorithm,
          results,
          averageReward: results.totalReward / testConfig.numInteractions,
          totalReward: results.totalReward
        });
      }

      // 创建对比测试会话
      const comparisonSession: TestSession = {
        sessionId: `comparison_${Date.now()}`,
        userId: 'comparison_test',
        numInteractions: testConfig.numInteractions * algorithms.length,
        status: 'completed',
        progress: 100,
        results: {
          recommendations: [],
          feedbacks: [],
          totalReward: comparisonResults.reduce((sum, r) => sum + r.totalReward, 0),
          comparisonData: comparisonResults
        },
        startTime: new Date(),
        endTime: new Date()
      };

      setCurrentSession(comparisonSession);
      setTestHistory(prev => [...prev, comparisonSession]);

    } catch (err: any) {
      setError(`对比测试失败: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 清除测试历史
  const clearHistory = () => {
    setTestHistory([]);
    setCurrentSession(null);
  };

  return (
    <div className="space-y-6">
      {/* 测试配置 */}
      <Card>
        <CardHeader>
          <CardTitle>算法性能测试器</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">测试用户ID</label>
              <Input
                value={testConfig.userId}
                onChange={(e) => setTestConfig(prev => ({ ...prev, userId: e.target.value }))}
                placeholder="输入测试用户ID"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">交互次数</label>
              <Input
                type="number"
                value={testConfig.numInteractions}
                onChange={(e) => setTestConfig(prev => ({ ...prev, numInteractions: Number(e.target.value) }))}
                min="5"
                max="100"
              />
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="includeContext"
                checked={testConfig.includeContext}
                onChange={(e) => setTestConfig(prev => ({ ...prev, includeContext: e.target.checked }))}
                className="rounded"
              />
              <label htmlFor="includeContext" className="text-sm">
                包含上下文特征
              </label>
            </div>
          </div>

          <div className="flex space-x-3">
            <Button
              onClick={runPerformanceTest}
              disabled={loading}
              className="flex-1"
            >
              {loading && currentSession?.status === 'running' ? '测试进行中...' : '🚀 运行单次测试'}
            </Button>
            
            <Button
              variant="outline"
              onClick={runComparisonTest}
              disabled={loading}
              className="flex-1"
            >
              {loading ? '对比测试中...' : '📊 算法对比测试'}
            </Button>
            
            <Button
              variant="outline"
              onClick={clearHistory}
              disabled={loading}
            >
              🗑️ 清除历史
            </Button>
          </div>

          {error && (
            <Alert className="border-red-200 bg-red-50">
              <div className="text-red-800">{error}</div>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* 当前测试状态 */}
      {currentSession && (
        <Card>
          <CardHeader>
            <CardTitle>当前测试状态</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* 测试进度 */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>测试进度</span>
                  <span>{currentSession.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      currentSession.status === 'completed' ? 'bg-green-500' :
                      currentSession.status === 'error' ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{ width: `${currentSession.progress}%` }}
                  ></div>
                </div>
              </div>

              {/* 测试信息 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-gray-500">用户ID</div>
                  <div className="font-medium">{currentSession.userId}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">交互次数</div>
                  <div className="font-medium">{currentSession.numInteractions}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">状态</div>
                  <Badge
                    variant={
                      currentSession.status === 'completed' ? 'default' :
                      currentSession.status === 'error' ? 'destructive' : 'secondary'
                    }
                  >
                    {currentSession.status}
                  </Badge>
                </div>
                <div>
                  <div className="text-sm text-gray-500">耗时</div>
                  <div className="font-medium">
                    {currentSession.startTime && currentSession.endTime
                      ? `${(currentSession.endTime.getTime() - currentSession.startTime.getTime()) / 1000}s`
                      : currentSession.startTime
                      ? `${((new Date().getTime() - currentSession.startTime.getTime()) / 1000).toFixed(1)}s`
                      : '--'
                    }
                  </div>
                </div>
              </div>

              {/* 测试结果 */}
              {currentSession.results && (
                <div className="space-y-4">
                  <h4 className="font-medium">测试结果</h4>
                  
                  {/* 基本统计 */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-blue-50 p-3 rounded-lg">
                      <div className="text-blue-800 text-sm">总奖励</div>
                      <div className="text-blue-600 text-lg font-semibold">
                        {currentSession.results.totalReward.toFixed(2)}
                      </div>
                    </div>
                    
                    <div className="bg-green-50 p-3 rounded-lg">
                      <div className="text-green-800 text-sm">平均奖励</div>
                      <div className="text-green-600 text-lg font-semibold">
                        {(currentSession.results.totalReward / currentSession.numInteractions).toFixed(3)}
                      </div>
                    </div>
                    
                    <div className="bg-purple-50 p-3 rounded-lg">
                      <div className="text-purple-800 text-sm">转化率</div>
                      <div className="text-purple-600 text-lg font-semibold">
                        {((currentSession.results.feedbacks.filter(f => f.feedback_type === 'click').length / 
                           currentSession.results.feedbacks.length) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* 对比测试结果 */}
                  {(currentSession.results as any).comparisonData && (
                    <div className="space-y-3">
                      <h5 className="font-medium">算法对比结果</h5>
                      <div className="space-y-2">
                        {(currentSession.results as any).comparisonData.map((item: any, index: number) => (
                          <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                            <div>
                              <span className="font-medium capitalize">
                                {item.algorithm.replace('_', ' ')}
                              </span>
                            </div>
                            <div className="flex space-x-4 text-sm">
                              <div>
                                <span className="text-gray-500">总奖励: </span>
                                <span className="font-medium">{item.totalReward.toFixed(2)}</span>
                              </div>
                              <div>
                                <span className="text-gray-500">平均: </span>
                                <span className="font-medium">{item.averageReward.toFixed(3)}</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      
                      {/* 最佳算法 */}
                      {(() => {
                        const best = (currentSession.results as any).comparisonData.reduce((prev: any, current: any) => 
                          prev.averageReward > current.averageReward ? prev : current
                        );
                        return (
                          <div className="bg-yellow-50 border border-yellow-200 p-3 rounded-lg">
                            <div className="text-yellow-800 font-medium">
                              🏆 最佳算法: {best.algorithm.replace('_', ' ')} 
                              (平均奖励: {best.averageReward.toFixed(3)})
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 测试历史 */}
      {testHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>测试历史</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {testHistory.slice(-5).reverse().map((session) => (
                <div key={session.sessionId} className="border border-gray-200 p-3 rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-medium">{session.userId}</div>
                      <div className="text-sm text-gray-500">
                        {session.numInteractions} 次交互 • {session.sessionId}
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge
                        variant={session.status === 'completed' ? 'default' : 'destructive'}
                      >
                        {session.status}
                      </Badge>
                      {session.results && (
                        <div className="text-sm text-gray-600 mt-1">
                          总奖励: {session.results.totalReward.toFixed(2)}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default AlgorithmTester;