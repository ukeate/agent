import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Alert } from '../components/ui/alert';
import { Progress } from '../components/ui/progress';
import { buildApiUrl, apiFetch } from '../utils/apiBase'

interface Experiment {
  id: string;
  name: string;
  description?: string;
  status: string;
  algorithm: string;
  objective: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  best_value?: number;
  total_trials: number;
  successful_trials: number;
  pruned_trials: number;
  failed_trials: number;
}

const statusColors = {
  'created': 'bg-gray-500',
  'running': 'bg-blue-500',
  'completed': 'bg-green-500',
  'failed': 'bg-red-500',
  'stopped': 'bg-yellow-500'
};

const algorithmNames = {
  'tpe': 'TPE (Tree Parzen Estimator)',
  'cmaes': 'CMA-ES',
  'random': '随机搜索',
  'grid': '网格搜索'
};

const HyperparameterExperimentsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = buildApiUrl('/api/v1/hyperparameter-optimization');

  // 加载实验列表
  const loadExperiments = async () => {
    try {
      setLoading(true);
      const response = await apiFetch(`${API_BASE}/experiments`);
      
      const data = await response.json();
      setExperiments(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误');
    } finally {
      setLoading(false);
    }
  };

  // 启动实验
  const startExperiment = async (experimentId: string) => {
    try {
      const response = await apiFetch(`${API_BASE}/experiments/${experimentId}/start`, {
        method: 'POST',
      });
      await response.json().catch(() => null);
      await loadExperiments();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误');
    }
  };

  // 停止实验
  const stopExperiment = async (experimentId: string) => {
    try {
      const response = await apiFetch(`${API_BASE}/experiments/${experimentId}/stop`, {
        method: 'POST',
      });
      await response.json().catch(() => null);
      await loadExperiments();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误');
    }
  };

  // 删除实验
  const deleteExperiment = async (experimentId: string) => {
    if (!confirm('确认删除此实验？此操作不可撤销。')) {
      return;
    }
    
    try {
      const response = await apiFetch(`${API_BASE}/experiments/${experimentId}`, {
        method: 'DELETE',
      });
      await response.json().catch(() => null);
      await loadExperiments();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误');
    }
  };

  useEffect(() => {
    loadExperiments();
    // 定期刷新运行中的实验
    const interval = setInterval(() => {
      if (experiments.some(exp => exp.status === 'running')) {
        loadExperiments();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => statusColors[status as keyof typeof statusColors] || 'bg-gray-500';
  const getSuccessRate = (experiment: Experiment) => 
    experiment.total_trials > 0 ? (experiment.successful_trials / experiment.total_trials) * 100 : 0;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">实验列表管理</h1>
            <p className="mt-2 text-gray-600">
              管理所有超参数优化实验的状态和进度
            </p>
          </div>
          <Button onClick={loadExperiments} disabled={loading}>
            {loading ? '刷新中...' : '刷新'}
          </Button>
        </div>

        {/* 错误提示 */}
        {error && (
          <Alert variant="destructive">
            {error}
          </Alert>
        )}

        {/* 统计概览 */}
        {experiments.length > 0 && (
          <Card className="p-6">
            <h2 className="text-lg font-semibold mb-4">统计概览</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-gray-900">{experiments.length}</div>
                <div className="text-sm text-gray-500">总实验数</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-blue-600">
                  {experiments.filter(e => e.status === 'running').length}
                </div>
                <div className="text-sm text-gray-500">运行中</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {experiments.filter(e => e.status === 'completed').length}
                </div>
                <div className="text-sm text-gray-500">已完成</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-red-600">
                  {experiments.filter(e => e.status === 'failed').length}
                </div>
                <div className="text-sm text-gray-500">失败</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-yellow-600">
                  {experiments.filter(e => e.status === 'stopped').length}
                </div>
                <div className="text-sm text-gray-500">已停止</div>
              </div>
            </div>
          </Card>
        )}

        {/* 实验列表 */}
        {loading && experiments.length === 0 ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-gray-500">加载实验列表中...</div>
          </div>
        ) : experiments.length === 0 ? (
          <Card className="p-8 text-center">
            <div className="text-gray-500 mb-4">暂无实验</div>
            <p className="text-sm text-gray-400">
              前往实验管理中心创建第一个实验
            </p>
          </Card>
        ) : (
          <div className="space-y-4">
            {experiments.map((experiment) => (
              <Card key={experiment.id} className="p-6">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {experiment.name}
                      </h3>
                      <Badge className={`${getStatusColor(experiment.status)} text-white`}>
                        {experiment.status}
                      </Badge>
                      <Badge className="bg-purple-100 text-purple-800">
                        {algorithmNames[experiment.algorithm as keyof typeof algorithmNames] || experiment.algorithm}
                      </Badge>
                      <Badge className="bg-blue-100 text-blue-800">
                        {experiment.objective === 'maximize' ? '最大化' : '最小化'}
                      </Badge>
                    </div>
                    
                    {experiment.description && (
                      <p className="text-gray-600 mb-3">{experiment.description}</p>
                    )}

                    {/* 实验指标 */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div>
                        <div className="text-sm font-medium text-gray-500">总试验数</div>
                        <div className="text-lg font-semibold">{experiment.total_trials}</div>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-500">成功率</div>
                        <div className="text-lg font-semibold text-green-600">
                          {getSuccessRate(experiment).toFixed(1)}%
                        </div>
                      </div>
                      {experiment.best_value !== undefined && (
                        <div>
                          <div className="text-sm font-medium text-gray-500">最佳值</div>
                          <div className="text-lg font-semibold text-blue-600 font-mono">
                            {experiment.best_value.toFixed(6)}
                          </div>
                        </div>
                      )}
                      <div>
                        <div className="text-sm font-medium text-gray-500">创建时间</div>
                        <div className="text-sm text-gray-600">
                          {new Date(experiment.created_at).toLocaleString('zh-CN')}
                        </div>
                      </div>
                    </div>

                    {/* 进度条 */}
                    <div className="mb-4">
                      <div className="flex justify-between text-sm text-gray-600 mb-1">
                        <span>试验进度</span>
                        <span>{experiment.successful_trials}/{experiment.total_trials}</span>
                      </div>
                      <Progress 
                        value={getSuccessRate(experiment)} 
                        className="h-2" 
                      />
                    </div>
                  </div>

                  {/* 操作按钮 */}
                  <div className="flex space-x-2 ml-4">
                    {experiment.status === 'created' && (
                      <Button 
                        size="sm" 
                        onClick={() => startExperiment(experiment.id)}
                      >
                        启动
                      </Button>
                    )}
                    {experiment.status === 'running' && (
                      <Button 
                        size="sm" 
                        variant="destructive"
                        onClick={() => stopExperiment(experiment.id)}
                      >
                        停止
                      </Button>
                    )}
                    {['completed', 'failed', 'stopped'].includes(experiment.status) && (
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => deleteExperiment(experiment.id)}
                      >
                        删除
                      </Button>
                    )}
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => window.open(`/hyperparameter-optimization`, '_blank')}
                    >
                      详情
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default HyperparameterExperimentsPage;
