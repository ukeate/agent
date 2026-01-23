/**
 * 超参数优化增强页面
 * 
 * 覆盖所有API端点功能:
 * 1. 实验管理 - 创建、启动、监控、停止实验
 * 2. 任务模板 - 预设任务、自定义任务配置
 * 3. 算法比较 - 多算法性能对比分析
 * 4. 实时监控 - 系统资源、实验进度监控
 * 5. 高级分析 - 参数重要性、敏感性分析
 * 6. 系统配置 - 算法选择、剪枝策略配置
 */

import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription } from '../components/ui/alert';
import { hyperparameterServiceEnhanced } from '../services/hyperparameterServiceEnhanced';
import { logger } from '../utils/logger'
import type { 
  ExperimentProgress, 
  TaskInfo, 
  AlgorithmComparison, 
  ResourceStats,
  AlgorithmInfo,
  PruningInfo,
  ParameterTypeInfo,
  ExperimentVisualization,
  CustomTaskRequest
} from '../services/hyperparameterServiceEnhanced';

// 自定义Tabs组件
interface TabsProps {
  children: React.ReactNode;
  defaultValue: string;
  className?: string;
}

interface TabsListProps {
  children: React.ReactNode;
  className?: string;
}

interface TabsTriggerProps {
  children: React.ReactNode;
  value: string;
  onClick: () => void;
  className?: string;
  isActive: boolean;
}

interface TabsContentProps {
  children: React.ReactNode;
  value: string;
  className?: string;
  isActive: boolean;
}

const Tabs: React.FC<TabsProps> = ({ children, defaultValue, className = '' }) => {
  const [activeTab, setActiveTab] = useState(defaultValue);
  
  return (
    <div className={`tabs ${className}`} data-active-tab={activeTab}>
      {React.Children.map(children, child => 
        React.isValidElement(child) 
          ? React.cloneElement(child as any, { activeTab, setActiveTab })
          : child
      )}
    </div>
  );
};

const TabsList: React.FC<TabsListProps & { activeTab?: string; setActiveTab?: (tab: string) => void }> = ({ 
  children, 
  className = '', 
  activeTab, 
  setActiveTab 
}) => (
  <div className={`flex border-b ${className}`}>
    {React.Children.map(children, child => 
      React.isValidElement(child) 
        ? React.cloneElement(child as any, { 
            onClick: () => setActiveTab?.(child.props.value),
            isActive: activeTab === child.props.value
          })
        : child
    )}
  </div>
);

const TabsTrigger: React.FC<TabsTriggerProps> = ({ 
  children, 
  value, 
  onClick, 
  className = '', 
  isActive 
}) => (
  <button
    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
      isActive 
        ? 'border-blue-500 text-blue-600' 
        : 'border-transparent text-gray-500 hover:text-gray-700'
    } ${className}`}
    onClick={onClick}
  >
    {children}
  </button>
);

const TabsContent: React.FC<TabsContentProps & { activeTab?: string }> = ({ 
  children, 
  value, 
  className = '', 
  activeTab 
}) => {
  if (activeTab !== value) return null;
  
  return (
    <div className={`mt-4 ${className}`}>
      {children}
    </div>
  );
};

const HyperparameterOptimizationPageEnhanced: React.FC = () => {
  // 状态管理
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 实验相关状态
  const [experiments, setExperiments] = useState<any[]>([]);
  const [activeExperiments, setActiveExperiments] = useState<any[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<any>(null);
  const [experimentProgress, setExperimentProgress] = useState<ExperimentProgress | null>(null);
  const [experimentVisualization, setExperimentVisualization] = useState<ExperimentVisualization | null>(null);
  
  // 任务模板相关状态
  const [presetTasks, setPresetTasks] = useState<string[]>([]);
  const [selectedTaskInfo, setSelectedTaskInfo] = useState<TaskInfo | null>(null);
  const [customTaskForm, setCustomTaskForm] = useState<Partial<CustomTaskRequest>>({
    algorithm: 'tpe',
    direction: 'minimize',
    n_trials: 100
  });
  
  // 算法比较相关状态
  const [algorithmComparison, setAlgorithmComparison] = useState<AlgorithmComparison | null>(null);
  const [comparisonAlgorithms, setComparisonAlgorithms] = useState<string[]>(['tpe', 'cmaes', 'random']);
  
  // 系统监控相关状态
  const [resourceStats, setResourceStats] = useState<ResourceStats | null>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  
  // 配置信息相关状态
  const [algorithmInfo, setAlgorithmInfo] = useState<AlgorithmInfo | null>(null);
  const [pruningInfo, setPruningInfo] = useState<PruningInfo | null>(null);
  const [parameterTypeInfo, setParameterTypeInfo] = useState<ParameterTypeInfo | null>(null);

  // 初始化数据加载
  useEffect(() => {
    loadInitialData();
    // 设置定时刷新资源状态
    const interval = setInterval(loadResourceStats, 30000); // 30秒刷新一次
    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      const [experiments, presetTasks, algorithmInfo, pruningInfo, parameterTypeInfo] = await Promise.allSettled([
        hyperparameterServiceEnhanced.listExperiments(),
        hyperparameterServiceEnhanced.getPresetTasks(),
        hyperparameterServiceEnhanced.getAvailableAlgorithms(),
        hyperparameterServiceEnhanced.getPruningStrategies(),
        hyperparameterServiceEnhanced.getParameterTypes()
      ]);

      if (experiments.status === 'fulfilled') setExperiments(experiments.value);
      if (presetTasks.status === 'fulfilled') setPresetTasks(presetTasks.value);
      if (algorithmInfo.status === 'fulfilled') setAlgorithmInfo(algorithmInfo.value);
      if (pruningInfo.status === 'fulfilled') setPruningInfo(pruningInfo.value);
      if (parameterTypeInfo.status === 'fulfilled') setParameterTypeInfo(parameterTypeInfo.value);

      await loadResourceStats();
      await loadActiveExperiments();
    } catch (error) {
      setError('加载初始数据失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const loadResourceStats = async () => {
    try {
      const stats = await hyperparameterServiceEnhanced.getResourceStatus();
      setResourceStats(stats);
    } catch (error) {
      logger.error('加载资源状态失败:', error);
    }
  };

  const loadActiveExperiments = async () => {
    try {
      const active = await hyperparameterServiceEnhanced.getActiveExperiments();
      setActiveExperiments(active);
    } catch (error) {
      logger.error('加载活跃实验失败:', error);
    }
  };

  const loadSystemHealth = async () => {
    try {
      const health = await hyperparameterServiceEnhanced.healthCheck();
      setSystemHealth(health);
    } catch (error) {
      logger.error('健康检查失败:', error);
    }
  };

  // 实验管理函数
  const handleStartExperiment = async (experimentId: string) => {
    try {
      setLoading(true);
      await hyperparameterServiceEnhanced.startExperiment(experimentId);
      await loadActiveExperiments();
      setError(null);
    } catch (error) {
      setError('启动实验失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGetExperimentProgress = async (experimentId: string) => {
    try {
      const progress = await hyperparameterServiceEnhanced.getExperimentProgress(experimentId);
      setExperimentProgress(progress);
    } catch (error) {
      setError('获取实验进度失败: ' + error.message);
    }
  };

  const handleGetExperimentVisualization = async (experimentId: string) => {
    try {
      const viz = await hyperparameterServiceEnhanced.getExperimentVisualizations(experimentId);
      setExperimentVisualization(viz);
    } catch (error) {
      setError('获取实验可视化失败: ' + error.message);
    }
  };

  // 任务管理函数
  const handleSelectPresetTask = async (taskName: string) => {
    try {
      const taskInfo = await hyperparameterServiceEnhanced.getTaskInfo(taskName);
      setSelectedTaskInfo(taskInfo);
    } catch (error) {
      setError('获取任务信息失败: ' + error.message);
    }
  };

  const handleCreateCustomTask = async () => {
    try {
      setLoading(true);
      if (!customTaskForm.task_name || !customTaskForm.parameters) {
        setError('请填写完整的任务信息');
        return;
      }
      
      const taskId = await hyperparameterServiceEnhanced.createCustomTask(customTaskForm as CustomTaskRequest);
      setError(null);
      
      // 重新加载任务列表
      const tasks = await hyperparameterServiceEnhanced.getPresetTasks();
      setPresetTasks(tasks);
    } catch (error) {
      setError('创建自定义任务失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOptimizeForTask = async (taskName: string) => {
    try {
      setLoading(true);
      const result = await hyperparameterServiceEnhanced.optimizeForTask(taskName);
      setError(null);
    } catch (error) {
      setError('启动任务优化失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // 算法比较函数
  const handleCompareAlgorithms = async (taskName: string) => {
    try {
      setLoading(true);
      const comparison = await hyperparameterServiceEnhanced.compareAlgorithms(taskName, comparisonAlgorithms);
      setAlgorithmComparison(comparison);
      setError(null);
    } catch (error) {
      setError('算法比较失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // 渲染实验管理标签页
  const renderExperimentManagement = () => (
    <div className="space-y-4">
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">实验管理</h3>
        
        {experiments.length === 0 ? (
          <div className="text-center text-gray-500 py-8">暂无实验</div>
        ) : (
          <div className="space-y-2">
            {experiments.map((experiment) => (
              <div key={experiment.id} className="flex items-center justify-between p-3 border rounded">
                <div>
                  <div className="font-medium">{experiment.name}</div>
                  <div className="text-sm text-gray-500">
                    状态: {experiment.status} | 算法: {experiment.algorithm}
                  </div>
                </div>
                <div className="space-x-2">
                  <Button
                    size="sm"
                    onClick={() => handleStartExperiment(experiment.id)}
                    disabled={experiment.status === 'running'}
                  >
                    启动
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleGetExperimentProgress(experiment.id)}
                  >
                    查看进度
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleGetExperimentVisualization(experiment.id)}
                  >
                    可视化
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {experimentProgress && (
        <Card className="p-4">
          <h4 className="font-semibold mb-2">实验进度</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>当前试验: {experimentProgress.current_trial}</div>
            <div>总试验数: {experimentProgress.total_trials}</div>
            <div>最佳值: {experimentProgress.best_value?.toFixed(4) || 'N/A'}</div>
            <div>运行时间: {Math.round(experimentProgress.elapsed_time)}秒</div>
          </div>
          {experimentProgress.best_params && (
            <div className="mt-2">
              <div className="text-sm font-medium">最佳参数:</div>
              <pre className="text-xs bg-gray-100 p-2 rounded mt-1">
                {JSON.stringify(experimentProgress.best_params, null, 2)}
              </pre>
            </div>
          )}
        </Card>
      )}

      {experimentVisualization && (
        <Card className="p-4">
          <h4 className="font-semibold mb-2">实验可视化</h4>
          <div className="space-y-4">
            <div>
              <div className="text-sm font-medium mb-1">优化历史</div>
              <div className="text-xs text-gray-500">
                数据点数量: {experimentVisualization.optimization_history?.data?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-sm font-medium mb-1">参数重要性</div>
              <div className="text-xs bg-gray-100 p-2 rounded">
                {experimentVisualization.parameter_importance?.data 
                  ? Object.entries(experimentVisualization.parameter_importance.data)
                      .map(([param, importance]) => `${param}: ${importance}`)
                      .join(', ')
                  : '暂无数据'
                }
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );

  // 渲染任务模板标签页
  const renderTaskTemplates = () => (
    <div className="space-y-4">
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">预设任务模板</h3>
        
        {presetTasks.length === 0 ? (
          <div className="text-center text-gray-500 py-8">暂无预设任务</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {presetTasks.map((taskName) => (
              <div key={taskName} className="border rounded p-3">
                <div className="font-medium mb-2">{taskName}</div>
                <div className="space-y-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleSelectPresetTask(taskName)}
                    className="w-full"
                  >
                    查看配置
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => handleOptimizeForTask(taskName)}
                    className="w-full"
                  >
                    启动优化
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {selectedTaskInfo && (
        <Card className="p-4">
          <h4 className="font-semibold mb-2">任务配置详情</h4>
          <div className="space-y-2 text-sm">
            <div><strong>名称:</strong> {selectedTaskInfo.name}</div>
            <div><strong>描述:</strong> {selectedTaskInfo.description}</div>
            <div><strong>算法:</strong> {selectedTaskInfo.algorithm}</div>
            <div><strong>方向:</strong> {selectedTaskInfo.direction}</div>
            <div><strong>试验数:</strong> {selectedTaskInfo.n_trials}</div>
            {selectedTaskInfo.parameters && (
              <div>
                <strong>参数:</strong>
                <div className="mt-1 space-y-1">
                  {selectedTaskInfo.parameters.map((param, index) => (
                    <div key={index} className="text-xs bg-gray-100 p-2 rounded">
                      {param.name} ({param.type}) - {param.description}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      <Card className="p-4">
        <h4 className="font-semibold mb-4">创建自定义任务</h4>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">任务名称</label>
            <input
              type="text"
              className="w-full p-2 border rounded"
              value={customTaskForm.task_name || ''}
              onChange={(e) => setCustomTaskForm({...customTaskForm, task_name: e.target.value})}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium mb-1">算法</label>
              <select
                className="w-full p-2 border rounded"
                value={customTaskForm.algorithm || 'tpe'}
                onChange={(e) => setCustomTaskForm({...customTaskForm, algorithm: e.target.value})}
              >
                {algorithmInfo?.algorithms?.map((alg) => (
                  <option key={alg} value={alg}>{alg}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">优化方向</label>
              <select
                className="w-full p-2 border rounded"
                value={customTaskForm.direction || 'minimize'}
                onChange={(e) => setCustomTaskForm({...customTaskForm, direction: e.target.value as 'minimize' | 'maximize'})}
              >
                <option value="minimize">最小化</option>
                <option value="maximize">最大化</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">试验数量</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={customTaskForm.n_trials || 100}
              onChange={(e) => setCustomTaskForm({...customTaskForm, n_trials: parseInt(e.target.value)})}
            />
          </div>
          <Button onClick={handleCreateCustomTask} disabled={loading}>
            创建任务
          </Button>
        </div>
      </Card>
    </div>
  );

  // 渲染算法比较标签页
  const renderAlgorithmComparison = () => (
    <div className="space-y-4">
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">算法性能比较</h3>
        
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">选择任务</label>
            <select className="w-full p-2 border rounded">
              <option value="">请选择任务</option>
              {presetTasks.map((task) => (
                <option key={task} value={task}>{task}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">比较算法</label>
            <div className="space-y-2">
              {algorithmInfo?.algorithms?.map((alg) => (
                <label key={alg} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={comparisonAlgorithms.includes(alg)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setComparisonAlgorithms([...comparisonAlgorithms, alg]);
                      } else {
                        setComparisonAlgorithms(comparisonAlgorithms.filter(a => a !== alg));
                      }
                    }}
                    className="mr-2"
                  />
                  {alg}
                </label>
              ))}
            </div>
          </div>
          
          <Button
            onClick={() => {
              const selectedTask = (document.querySelector('select') as HTMLSelectElement)?.value;
              if (selectedTask) {
                handleCompareAlgorithms(selectedTask);
              }
            }}
            disabled={loading || comparisonAlgorithms.length < 2}
          >
            开始比较
          </Button>
        </div>
      </Card>

      {algorithmComparison && (
        <Card className="p-4">
          <h4 className="font-semibold mb-2">比较结果</h4>
          <div className="space-y-3">
            <div>
              <strong>获胜算法:</strong> {algorithmComparison.winner}
            </div>
            
            <div>
              <strong>各算法性能:</strong>
              <div className="mt-2 space-y-2">
                {algorithmComparison.algorithms?.map((alg, index) => (
                  <div key={index} className="border rounded p-2">
                    <div className="font-medium">{alg.name}</div>
                    <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                      <div>最佳值: {alg.best_value?.toFixed(4)}</div>
                      <div>试验数: {alg.total_trials}</div>
                      <div>收敛率: {alg.convergence_rate?.toFixed(3)}</div>
                      <div>效率分数: {alg.efficiency_score?.toFixed(3)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );

  // 渲染系统监控标签页
  const renderSystemMonitoring = () => (
    <div className="space-y-4">
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">系统资源状态</h3>
        
        {resourceStats ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{resourceStats.cpu_usage}%</div>
              <div className="text-sm text-gray-500">CPU使用率</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{resourceStats.memory_usage}%</div>
              <div className="text-sm text-gray-500">内存使用率</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{resourceStats.active_experiments}</div>
              <div className="text-sm text-gray-500">活跃实验</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{resourceStats.pending_trials}</div>
              <div className="text-sm text-gray-500">待处理试验</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{resourceStats.completed_trials}</div>
              <div className="text-sm text-gray-500">已完成试验</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{resourceStats.failed_trials}</div>
              <div className="text-sm text-gray-500">失败试验</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-indigo-600">
                {(resourceStats.storage_used / 1024 / 1024).toFixed(1)}MB
              </div>
              <div className="text-sm text-gray-500">存储使用</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-teal-600">
                {(resourceStats.storage_available / 1024 / 1024).toFixed(1)}MB
              </div>
              <div className="text-sm text-gray-500">可用存储</div>
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">加载资源状态中...</div>
        )}
      </Card>

      <Card className="p-4">
        <h4 className="font-semibold mb-4">活跃实验</h4>
        {activeExperiments.length === 0 ? (
          <div className="text-center text-gray-500 py-4">暂无活跃实验</div>
        ) : (
          <div className="space-y-2">
            {activeExperiments.map((exp, index) => (
              <div key={index} className="flex items-center justify-between p-2 border rounded">
                <div>
                  <div className="font-medium">{exp.name || `实验-${index + 1}`}</div>
                  <div className="text-sm text-gray-500">状态: {exp.status}</div>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleGetExperimentProgress(exp.id)}
                >
                  查看详情
                </Button>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold">系统健康检查</h4>
          <Button size="sm" variant="outline" onClick={loadSystemHealth}>
            检查健康状态
          </Button>
        </div>
        
        {systemHealth ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span>系统状态:</span>
              <span className={`px-2 py-1 rounded text-sm ${
                systemHealth.status === 'healthy' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {systemHealth.status}
              </span>
            </div>
            <div className="text-sm text-gray-600">
              检查时间: {new Date(systemHealth.timestamp).toLocaleString()}
            </div>
            {systemHealth.services && (
              <div className="space-y-1">
                <div className="text-sm font-medium">服务状态:</div>
                {Object.entries(systemHealth.services).map(([service, status]) => (
                  <div key={service} className="flex justify-between text-sm">
                    <span>{service}:</span>
                    <span className={status === 'connected' || status === 'running' ? 'text-green-600' : 'text-red-600'}>
                      {status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-4">点击检查健康状态</div>
        )}
      </Card>
    </div>
  );

  // 渲染配置信息标签页
  const renderConfigurationInfo = () => (
    <div className="space-y-4">
      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">可用算法</h3>
        {algorithmInfo ? (
          <div className="space-y-3">
            {algorithmInfo.algorithms?.map((alg) => (
              <div key={alg} className="border rounded p-3">
                <div className="font-medium">{alg}</div>
                <div className="text-sm text-gray-600 mt-1">
                  {algorithmInfo.descriptions?.[alg] || '暂无描述'}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-4">加载算法信息中...</div>
        )}
      </Card>

      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">剪枝策略</h3>
        {pruningInfo ? (
          <div className="space-y-3">
            {pruningInfo.pruning_strategies?.map((strategy) => (
              <div key={strategy} className="border rounded p-3">
                <div className="font-medium">{strategy}</div>
                <div className="text-sm text-gray-600 mt-1">
                  {pruningInfo.descriptions?.[strategy] || '暂无描述'}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-4">加载剪枝策略信息中...</div>
        )}
      </Card>

      <Card className="p-4">
        <h3 className="text-lg font-semibold mb-4">参数类型</h3>
        {parameterTypeInfo ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {parameterTypeInfo.parameter_types?.map((type) => (
                <div key={type} className="border rounded p-3">
                  <div className="font-medium">{type}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    {parameterTypeInfo.descriptions?.[type] || '暂无描述'}
                  </div>
                </div>
              ))}
            </div>
            
            {parameterTypeInfo.examples && (
              <div>
                <h4 className="font-medium mb-2">参数配置示例</h4>
                <div className="space-y-2">
                  {Object.entries(parameterTypeInfo.examples).map(([type, example]) => (
                    <div key={type} className="border rounded p-2">
                      <div className="text-sm font-medium">{type} 类型示例:</div>
                      <pre className="text-xs bg-gray-100 p-2 rounded mt-1">
                        {JSON.stringify(example, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-4">加载参数类型信息中...</div>
        )}
      </Card>
    </div>
  );

  return (
    <div className="p-6">
      {/* 页面头部 */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">超参数优化管理中心</h1>
        <p className="mt-2 text-gray-600">
          全面的超参数优化管理平台：实验管理、任务模板、算法比较、系统监控
        </p>
      </div>

      {/* 全局错误提示 */}
      {error && (
        <Alert className="mb-4" variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* 主要功能标签页 */}
      <Tabs defaultValue="experiments" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="experiments">实验管理</TabsTrigger>
          <TabsTrigger value="tasks">任务模板</TabsTrigger>
          <TabsTrigger value="comparison">算法比较</TabsTrigger>
          <TabsTrigger value="monitoring">系统监控</TabsTrigger>
          <TabsTrigger value="configuration">系统配置</TabsTrigger>
        </TabsList>

        <TabsContent value="experiments">
          {renderExperimentManagement()}
        </TabsContent>

        <TabsContent value="tasks">
          {renderTaskTemplates()}
        </TabsContent>

        <TabsContent value="comparison">
          {renderAlgorithmComparison()}
        </TabsContent>

        <TabsContent value="monitoring">
          {renderSystemMonitoring()}
        </TabsContent>

        <TabsContent value="configuration">
          {renderConfigurationInfo()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default HyperparameterOptimizationPageEnhanced;
