import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Alert } from '../components/ui/alert';
import { Input } from '../components/ui/input';
import { Tabs } from '../components/ui/Tabs';

interface AlgorithmConfig {
  name: string;
  displayName: string;
  description: string;
  parameters: Record<string, any>;
  advantages: string[];
  disadvantages: string[];
  bestFor: string[];
}

const algorithms: AlgorithmConfig[] = [
  {
    name: 'tpe',
    displayName: 'TPE (Tree Parzen Estimator)',
    description: '基于贝叶斯优化的高效算法，使用历史试验数据指导搜索方向',
    parameters: {
      n_startup_trials: { type: 'int', default: 10, min: 5, max: 100, description: '随机试验数量' },
      n_ei_candidates: { type: 'int', default: 24, min: 10, max: 100, description: 'EI候选点数量' },
      gamma: { type: 'float', default: 0.25, min: 0.1, max: 0.5, description: '好/坏试验分割比例' }
    },
    advantages: [
      '对连续参数优化效果好',
      '能处理条件参数空间',
      '计算效率高',
      '对噪声有一定鲁棒性'
    ],
    disadvantages: [
      '对离散参数效果一般',
      '高维空间性能下降',
      '需要足够的历史数据'
    ],
    bestFor: [
      '连续参数为主的优化',
      '中等维度参数空间',
      '对效率要求高的场景'
    ]
  },
  {
    name: 'cmaes',
    displayName: 'CMA-ES (协方差矩阵适应进化策略)',
    description: '进化算法，通过适应协方差矩阵来探索参数空间',
    parameters: {
      sigma: { type: 'float', default: 0.1, min: 0.01, max: 1.0, description: '初始步长' },
      population_size: { type: 'int', default: null, min: 4, max: 100, description: '种群大小' },
      restart_strategy: { type: 'select', default: 'ipop', options: ['ipop', 'bipop'], description: '重启策略' }
    },
    advantages: [
      '对连续优化效果极佳',
      '自适应步长调整',
      '对多峰函数效果好',
      '理论基础扎实'
    ],
    disadvantages: [
      '仅支持连续参数',
      '计算复杂度较高',
      '参数数量较多时收敛慢'
    ],
    bestFor: [
      '纯连续参数优化',
      '复杂多峰优化问题',
      '对精度要求高的场景'
    ]
  },
  {
    name: 'random',
    displayName: '随机搜索',
    description: '在参数空间中随机采样，简单有效的基准算法',
    parameters: {
      seed: { type: 'int', default: null, min: 0, max: 99999, description: '随机种子' }
    },
    advantages: [
      '实现简单',
      '无参数偏见',
      '易于理解和调试',
      '适合快速baseline'
    ],
    disadvantages: [
      '收敛速度慢',
      '无法利用历史信息',
      '效率较低'
    ],
    bestFor: [
      '初始基准测试',
      '参数空间不确定',
      '快速验证可行性'
    ]
  },
  {
    name: 'grid',
    displayName: '网格搜索',
    description: '系统性地遍历参数空间的所有组合',
    parameters: {
      n_points_per_axis: { type: 'int', default: 10, min: 2, max: 20, description: '每个轴的点数' }
    },
    advantages: [
      '系统性完整搜索',
      '结果可重现',
      '易于并行化',
      '适合离散参数'
    ],
    disadvantages: [
      '维度诅咒严重',
      '计算量随维度指数增长',
      '无法处理条件参数'
    ],
    bestFor: [
      '低维参数空间',
      '离散参数组合',
      '需要完整覆盖的场景'
    ]
  }
];

const HyperparameterAlgorithmsPage: React.FC = () => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmConfig>(algorithms[0]);
  const [algorithmConfigs, setAlgorithmConfigs] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  // 初始化配置
  useEffect(() => {
    const initialConfigs: Record<string, any> = {};
    algorithms.forEach(alg => {
      initialConfigs[alg.name] = {};
      Object.entries(alg.parameters).forEach(([param, config]) => {
        initialConfigs[alg.name][param] = config.default;
      });
    });
    setAlgorithmConfigs(initialConfigs);
  }, []);

  // 更新参数配置
  const updateParameter = (algorithmName: string, paramName: string, value: any) => {
    setAlgorithmConfigs(prev => ({
      ...prev,
      [algorithmName]: {
        ...prev[algorithmName],
        [paramName]: value
      }
    }));
  };

  // 重置配置
  const resetConfig = (algorithmName: string) => {
    const algorithm = algorithms.find(alg => alg.name === algorithmName);
    if (algorithm) {
      const resetConfig: Record<string, any> = {};
      Object.entries(algorithm.parameters).forEach(([param, config]) => {
        resetConfig[param] = config.default;
      });
      setAlgorithmConfigs(prev => ({
        ...prev,
        [algorithmName]: resetConfig
      }));
    }
  };

  // 渲染参数配置表单
  const renderParameterForm = (algorithm: AlgorithmConfig) => {
    const config = algorithmConfigs[algorithm.name] || {};
    
    return (
      <div className="space-y-4">
        {Object.entries(algorithm.parameters).map(([paramName, paramConfig]) => (
          <div key={paramName} className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {paramName}
              </label>
              <p className="text-xs text-gray-500">{paramConfig.description}</p>
            </div>
            
            <div>
              {paramConfig.type === 'int' ? (
                <Input
                  type="number"
                  value={config[paramName] || paramConfig.default}
                  min={paramConfig.min}
                  max={paramConfig.max}
                  onChange={(e) => updateParameter(
                    algorithm.name, 
                    paramName, 
                    parseInt(e.target.value) || paramConfig.default
                  )}
                />
              ) : paramConfig.type === 'float' ? (
                <Input
                  type="number"
                  step="0.01"
                  value={config[paramName] || paramConfig.default}
                  min={paramConfig.min}
                  max={paramConfig.max}
                  onChange={(e) => updateParameter(
                    algorithm.name, 
                    paramName, 
                    parseFloat(e.target.value) || paramConfig.default
                  )}
                />
              ) : paramConfig.type === 'select' ? (
                <select
                  className="w-full p-2 border border-gray-300 rounded-md"
                  value={config[paramName] || paramConfig.default}
                  onChange={(e) => updateParameter(algorithm.name, paramName, e.target.value)}
                >
                  {paramConfig.options.map((option: string) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              ) : (
                <Input
                  value={config[paramName] || paramConfig.default}
                  onChange={(e) => updateParameter(algorithm.name, paramName, e.target.value)}
                />
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  // 渲染算法概览
  const renderAlgorithmOverview = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-2">算法描述</h3>
        <p className="text-gray-600">{selectedAlgorithm.description}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="p-4">
          <h4 className="font-semibold text-green-600 mb-3">优势</h4>
          <ul className="space-y-2">
            {selectedAlgorithm.advantages.map((advantage, index) => (
              <li key={index} className="flex items-start">
                <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span className="text-sm text-gray-600">{advantage}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card className="p-4">
          <h4 className="font-semibold text-red-600 mb-3">劣势</h4>
          <ul className="space-y-2">
            {selectedAlgorithm.disadvantages.map((disadvantage, index) => (
              <li key={index} className="flex items-start">
                <span className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span className="text-sm text-gray-600">{disadvantage}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card className="p-4">
          <h4 className="font-semibold text-blue-600 mb-3">适用场景</h4>
          <ul className="space-y-2">
            {selectedAlgorithm.bestFor.map((scenario, index) => (
              <li key={index} className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span className="text-sm text-gray-600">{scenario}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );

  // 渲染参数配置
  const renderParameterConfig = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">参数配置</h3>
        <Button
          variant="outline"
          onClick={() => resetConfig(selectedAlgorithm.name)}
        >
          重置默认值
        </Button>
      </div>

      <Card className="p-6">
        {renderParameterForm(selectedAlgorithm)}
      </Card>

      <Card className="p-4 bg-blue-50">
        <h4 className="font-semibold text-blue-800 mb-2">当前配置</h4>
        <pre className="text-sm text-blue-700 bg-blue-100 p-3 rounded overflow-x-auto">
          {JSON.stringify(algorithmConfigs[selectedAlgorithm.name] || {}, null, 2)}
        </pre>
      </Card>
    </div>
  );

  const tabsData = [
    { id: 'overview', label: '算法概览', content: renderAlgorithmOverview() },
    { id: 'parameters', label: '参数配置', content: renderParameterConfig() }
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">算法配置管理</h1>
          <p className="mt-2 text-gray-600">
            配置和比较不同的超参数优化算法
          </p>
        </div>

        {error && (
          <Alert variant="destructive">
            {error}
          </Alert>
        )}

        {/* 算法选择 */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold mb-4">选择算法</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {algorithms.map((algorithm) => (
              <div
                key={algorithm.name}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedAlgorithm.name === algorithm.name
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedAlgorithm(algorithm)}
              >
                <div className="flex items-center justify-between mb-2">
                  <Badge className="bg-blue-100 text-blue-800">
                    {algorithm.name.toUpperCase()}
                  </Badge>
                  {selectedAlgorithm.name === algorithm.name && (
                    <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  )}
                </div>
                <h3 className="font-medium text-gray-900 mb-1">
                  {algorithm.displayName}
                </h3>
                <p className="text-sm text-gray-600 line-clamp-2">
                  {algorithm.description}
                </p>
              </div>
            ))}
          </div>
        </Card>

        {/* 算法详情 */}
        <Card className="p-6">
          <div className="flex items-center space-x-4 mb-6">
            <Badge className="bg-blue-500 text-white px-3 py-1">
              {selectedAlgorithm.name.toUpperCase()}
            </Badge>
            <h2 className="text-xl font-semibold text-gray-900">
              {selectedAlgorithm.displayName}
            </h2>
          </div>

          <Tabs
            tabs={tabsData}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        </Card>

        {/* 算法比较 */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold mb-4">算法性能比较</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    算法
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    参数类型支持
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    收敛速度
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    计算复杂度
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    推荐场景
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge className="bg-blue-100 text-blue-800">TPE</Badge>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">连续 + 离散</td>
                  <td className="px-6 py-4 text-sm text-gray-600">快</td>
                  <td className="px-6 py-4 text-sm text-gray-600">低</td>
                  <td className="px-6 py-4 text-sm text-gray-600">通用推荐</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge className="bg-purple-100 text-purple-800">CMA-ES</Badge>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">连续</td>
                  <td className="px-6 py-4 text-sm text-gray-600">中</td>
                  <td className="px-6 py-4 text-sm text-gray-600">高</td>
                  <td className="px-6 py-4 text-sm text-gray-600">连续优化</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge className="bg-green-100 text-green-800">随机</Badge>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">全部</td>
                  <td className="px-6 py-4 text-sm text-gray-600">慢</td>
                  <td className="px-6 py-4 text-sm text-gray-600">极低</td>
                  <td className="px-6 py-4 text-sm text-gray-600">基准测试</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge className="bg-yellow-100 text-yellow-800">网格</Badge>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">离散</td>
                  <td className="px-6 py-4 text-sm text-gray-600">系统性</td>
                  <td className="px-6 py-4 text-sm text-gray-600">极高</td>
                  <td className="px-6 py-4 text-sm text-gray-600">低维空间</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default HyperparameterAlgorithmsPage;