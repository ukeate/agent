import React, { useState, useEffect } from 'react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Alert } from '../components/ui/lalert';
import { 
  ExperimentCard, 
  ExperimentForm, 
  ExperimentDetail 
} from '../components/hyperparameter';

interface Experiment {
  id: string;
  name: string;
  status: string;
  algorithm: string;
  objective: string;
  created_at: string;
  best_value?: number;
  total_trials?: number;
  successful_trials?: number;
}

interface Trial {
  id: string;
  trial_number: number;
  parameters: Record<string, any>;
  value?: number;
  state: string;
  start_time?: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
}

const HyperparameterOptimizationPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<any>(null);
  const [trials, setTrials] = useState<Trial[]>([]);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [presetTasks, setPresetTasks] = useState<string[]>([]);

  // API基础URL
  const API_BASE = '/api/v1/hyperparameter-optimization';

  // 加载实验列表
  const loadExperiments = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/experiments`);
      if (!response.ok) throw new Error('Failed to load experiments');
      
      const data = await response.json();
      setExperiments(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // 加载预设任务
  const loadPresetTasks = async () => {
    try {
      const response = await fetch(`${API_BASE}/tasks`);
      if (!response.ok) throw new Error('Failed to load preset tasks');
      
      const data = await response.json();
      setPresetTasks(data);
    } catch (err) {
      console.error('Failed to load preset tasks:', err);
    }
  };

  // 加载实验详情
  const loadExperimentDetail = async (experimentId: string) => {
    try {
      setLoading(true);
      
      // 获取实验详情
      const [experimentResponse, trialsResponse] = await Promise.all([
        fetch(`${API_BASE}/experiments/${experimentId}`),
        fetch(`${API_BASE}/experiments/${experimentId}/trials`)
      ]);
      
      if (!experimentResponse.ok || !trialsResponse.ok) {
        throw new Error('Failed to load experiment details');
      }
      
      const experimentData = await experimentResponse.json();
      const trialsData = await trialsResponse.json();
      
      setSelectedExperiment(experimentData);
      setTrials(trialsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // 加载预设配置
  const loadPresetConfig = async (taskName: string) => {
    try {
      const response = await fetch(`${API_BASE}/tasks/${taskName}`);
      if (!response.ok) throw new Error('Failed to load preset config');
      
      return await response.json();
    } catch (err) {
      console.error('Failed to load preset config:', err);
      throw err;
    }
  };

  // 创建实验
  const createExperiment = async (experimentData: any) => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE}/experiments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(experimentData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create experiment');
      }
      
      await loadExperiments();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // 启动实验
  const startExperiment = async (experimentId: string) => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE}/experiments/${experimentId}/start`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start experiment');
      }
      
      await loadExperiments();
      if (selectedExperiment?.id === experimentId) {
        await loadExperimentDetail(experimentId);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // 停止实验
  const stopExperiment = async (experimentId: string) => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE}/experiments/${experimentId}/stop`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to stop experiment');
      }
      
      await loadExperiments();
      if (selectedExperiment?.id === experimentId) {
        await loadExperimentDetail(experimentId);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // 删除实验
  const deleteExperiment = async (experimentId: string) => {
    if (!confirm('确认删除此实验？此操作不可撤销。')) {
      return;
    }
    
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE}/experiments/${experimentId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete experiment');
      }
      
      if (selectedExperiment?.id === experimentId) {
        setSelectedExperiment(null);
        setTrials([]);
      }
      
      await loadExperiments();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // 页面加载时获取数据
  useEffect(() => {
    loadExperiments();
    loadPresetTasks();
  }, []);

  const handleViewExperiment = (experimentId: string) => {
    loadExperimentDetail(experimentId);
  };

  const handleBackToList = () => {
    setSelectedExperiment(null);
    setTrials([]);
  };

  const handleRefreshDetail = () => {
    if (selectedExperiment) {
      loadExperimentDetail(selectedExperiment.id);
    }
  };

  // 渲染实验列表
  const renderExperimentList = () => (
    <div className="space-y-6">
      {/* 页面标题和创建按钮 */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">超参数优化</h1>
          <p className="mt-2 text-gray-600">
            创建和管理自动化超参数优化实验，提升模型性能
          </p>
        </div>
        <Button onClick={() => setIsFormOpen(true)}>
          创建实验
        </Button>
      </div>

      {/* 错误提示 */}
      {error && (
        <Alert variant="destructive" className="mb-4">
          {error}
        </Alert>
      )}

      {/* 实验列表 */}
      {loading && experiments.length === 0 ? (
        <div className="flex justify-center items-center h-64">
          <div className="text-gray-500">加载中...</div>
        </div>
      ) : experiments.length === 0 ? (
        <Card className="p-8 text-center">
          <div className="text-gray-500 mb-4">暂无实验</div>
          <Button onClick={() => setIsFormOpen(true)}>
            创建第一个实验
          </Button>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {experiments.map((experiment) => (
            <ExperimentCard
              key={experiment.id}
              experiment={experiment}
              onView={handleViewExperiment}
              onStart={startExperiment}
              onStop={stopExperiment}
              onDelete={deleteExperiment}
            />
          ))}
        </div>
      )}

      {/* 统计信息 */}
      {experiments.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">实验统计</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {experiments.length}
              </div>
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
    </div>
  );

  // 渲染实验详情
  const renderExperimentDetail = () => (
    <div className="space-y-6">
      <div className="flex items-center space-x-4">
        <Button variant="outline" onClick={handleBackToList}>
          ← 返回列表
        </Button>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">实验详情</h1>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          {error}
        </Alert>
      )}

      {loading && !selectedExperiment ? (
        <div className="flex justify-center items-center h-64">
          <div className="text-gray-500">加载中...</div>
        </div>
      ) : selectedExperiment ? (
        <ExperimentDetail
          experiment={selectedExperiment}
          trials={trials}
          onRefresh={handleRefreshDetail}
          onStart={() => startExperiment(selectedExperiment.id)}
          onStop={() => stopExperiment(selectedExperiment.id)}
          onDelete={() => deleteExperiment(selectedExperiment.id)}
        />
      ) : null}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      {selectedExperiment ? renderExperimentDetail() : renderExperimentList()}
      
      {/* 创建实验表单 */}
      <ExperimentForm
        isOpen={isFormOpen}
        onClose={() => setIsFormOpen(false)}
        onSubmit={createExperiment}
        presetTasks={presetTasks}
        onLoadPreset={loadPresetConfig}
      />
    </div>
  );
};

export default HyperparameterOptimizationPage;