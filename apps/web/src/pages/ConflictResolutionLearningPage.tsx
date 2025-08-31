import React, { useState, useEffect } from 'react';
import { Card, Progress, Tabs, Badge, Button } from 'antd';
import { 
  GitMerge, AlertTriangle, CheckCircle, Clock,
  Activity, Zap, RefreshCw, Settings, Database,
  ArrowRight, GitBranch, Users, Bot, Play, Pause
} from 'lucide-react';

const { TabPane } = Tabs;

// 冲突类型和解决策略的教学数据
interface ConflictScenario {
  id: string;
  conflict_type: 'UPDATE_UPDATE' | 'CREATE_CREATE' | 'UPDATE_DELETE' | 'DELETE_UPDATE';
  description: string;
  local_data: any;
  remote_data: any;
  local_timestamp: string;
  remote_timestamp: string;
  suggested_strategies: {
    strategy: string;
    description: string;
    result: any;
    confidence: number;
    pros: string[];
    cons: string[];
  }[];
}

// 解决策略执行结果
interface ResolutionResult {
  scenario_id: string;
  strategy: string;
  resolved_data: any;
  confidence: number;
  execution_time_ms: number;
  explanation: string;
  algorithm_steps: string[];
}

// 三路合并算法演示
interface ThreeWayMergeDemo {
  base_data: any;
  local_data: any;
  remote_data: any;
  merge_result: any;
  conflicts: Array<{
    field: string;
    conflict_type: string;
    resolution: string;
  }>;
  algorithm_steps: string[];
}

const ConflictResolutionLearningPage: React.FC = () => {
  const [scenarios, setScenarios] = useState<ConflictScenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<ConflictScenario | null>(null);
  const [resolutionResults, setResolutionResults] = useState<ResolutionResult[]>([]);
  const [threeWayDemo, setThreeWayDemo] = useState<ThreeWayMergeDemo | null>(null);
  const [isPlayingDemo, setIsPlayingDemo] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // 冲突解决算法实现 - 教学版本
  const ConflictResolutionAlgorithms = {
    // 最后写入者获胜
    lastWriterWins: (local: any, remote: any, localTime: string, remoteTime: string) => {
      const localTimestamp = new Date(localTime).getTime();
      const remoteTimestamp = new Date(remoteTime).getTime();
      
      const steps = [
        `比较时间戳: local(${localTime}) vs remote(${remoteTime})`,
        `Local timestamp: ${localTimestamp}`,
        `Remote timestamp: ${remoteTimestamp}`,
        localTimestamp >= remoteTimestamp ? 'Local 时间更新或相等 -> 选择 Local' : 'Remote 时间更新 -> 选择 Remote'
      ];

      return {
        result: localTimestamp >= remoteTimestamp ? local : remote,
        confidence: 0.9,
        steps
      };
    },

    // 三路合并
    threeWayMerge: (base: any, local: any, remote: any) => {
      const merged = { ...base };
      const conflicts: Array<{field: string, conflict_type: string, resolution: string}> = [];
      const steps: string[] = [];

      steps.push('开始三路合并算法:');
      steps.push(`Base: ${JSON.stringify(base)}`);
      steps.push(`Local: ${JSON.stringify(local)}`);
      steps.push(`Remote: ${JSON.stringify(remote)}`);

      const allFields = new Set([
        ...Object.keys(base || {}),
        ...Object.keys(local || {}),
        ...Object.keys(remote || {})
      ]);

      for (const field of allFields) {
        const baseValue = base?.[field];
        const localValue = local?.[field];
        const remoteValue = remote?.[field];

        steps.push(`\n字段 '${field}':`);
        steps.push(`  Base: ${baseValue}, Local: ${localValue}, Remote: ${remoteValue}`);

        // 三路合并逻辑
        if (localValue === remoteValue) {
          // 本地和远程相同，无冲突
          merged[field] = localValue;
          steps.push(`  -> 无冲突: Local = Remote = ${localValue}`);
        } else if (localValue === baseValue) {
          // 本地未改变，使用远程
          merged[field] = remoteValue;
          steps.push(`  -> Local 未变，使用 Remote: ${remoteValue}`);
        } else if (remoteValue === baseValue) {
          // 远程未改变，使用本地
          merged[field] = localValue;
          steps.push(`  -> Remote 未变，使用 Local: ${localValue}`);
        } else {
          // 都有改变，产生冲突
          conflicts.push({
            field,
            conflict_type: 'UPDATE_UPDATE',
            resolution: 'manual_required'
          });
          merged[field] = `<<<< LOCAL\n${localValue}\n====\n${remoteValue}\n>>>> REMOTE`;
          steps.push(`  -> 冲突: 两边都有修改，需要手动解决`);
        }
      }

      return {
        result: merged,
        conflicts,
        confidence: conflicts.length === 0 ? 0.95 : 0.3,
        steps
      };
    },

    // 语义合并 (简化版)
    semanticMerge: (local: any, remote: any, fieldType: string) => {
      const steps: string[] = [];
      steps.push(`语义合并 - 字段类型: ${fieldType}`);

      switch (fieldType) {
        case 'array':
          // 数组合并：去重并排序
          const localArray = Array.isArray(local) ? local : [];
          const remoteArray = Array.isArray(remote) ? remote : [];
          const mergedArray = [...new Set([...localArray, ...remoteArray])].sort();
          steps.push(`数组合并: 合并 + 去重 + 排序`);
          steps.push(`结果: [${mergedArray.join(', ')}]`);
          return { result: mergedArray, confidence: 0.8, steps };

        case 'number':
          // 数值合并：取平均值或最大值
          const avg = ((local || 0) + (remote || 0)) / 2;
          steps.push(`数值合并: 取平均值 = (${local} + ${remote}) / 2 = ${avg}`);
          return { result: avg, confidence: 0.7, steps };

        default:
          // 默认：字符串拼接
          const merged = `${local || ''} ${remote || ''}`.trim();
          steps.push(`字符串合并: 拼接两个值`);
          return { result: merged, confidence: 0.6, steps };
      }
    }
  };

  // 生成冲突场景演示数据
  const generateConflictScenarios = () => {
    const scenarios: ConflictScenario[] = [
      {
        id: 'scenario-1',
        conflict_type: 'UPDATE_UPDATE',
        description: '用户信息同时修改冲突',
        local_data: {
          id: 'user-123',
          name: 'Alice Johnson',
          email: 'alice.johnson@company.com',
          department: 'Engineering',
          last_modified: '2025-01-15T10:30:00Z'
        },
        remote_data: {
          id: 'user-123',
          name: 'Alice J. Smith',
          email: 'alice.smith@company.com',
          department: 'Product',
          last_modified: '2025-01-15T10:32:00Z'
        },
        local_timestamp: '2025-01-15T10:30:00Z',
        remote_timestamp: '2025-01-15T10:32:00Z',
        suggested_strategies: [
          {
            strategy: 'LAST_WRITER_WINS',
            description: '基于时间戳选择最新版本',
            result: null, // 将在运行时计算
            confidence: 0.9,
            pros: ['简单快速', '自动化程度高', '时间逻辑清晰'],
            cons: ['可能丢失重要更改', '依赖时钟同步', '忽略更改质量']
          },
          {
            strategy: 'THREE_WAY_MERGE',
            description: '智能合并两个版本的更改',
            result: null,
            confidence: 0.8,
            pros: ['保留大部分更改', '减少数据丢失', '智能冲突检测'],
            cons: ['复杂度高', '可能产生新冲突', '需要基础版本']
          },
          {
            strategy: 'MANUAL',
            description: '人工审查和解决',
            result: null,
            confidence: 1.0,
            pros: ['100%准确性', '考虑业务逻辑', '灵活性最高'],
            cons: ['耗时较长', '需要人工介入', '可能产生延迟']
          }
        ]
      },
      {
        id: 'scenario-2',
        conflict_type: 'CREATE_CREATE',
        description: '同时创建相同资源冲突',
        local_data: {
          id: 'doc-456',
          title: 'Project Proposal',
          content: 'Local version of the project proposal...',
          author: 'Alice',
          created_at: '2025-01-15T09:15:00Z'
        },
        remote_data: {
          id: 'doc-456',
          title: 'Project Proposal',
          content: 'Remote version of the project proposal...',
          author: 'Bob',
          created_at: '2025-01-15T09:17:00Z'
        },
        local_timestamp: '2025-01-15T09:15:00Z',
        remote_timestamp: '2025-01-15T09:17:00Z',
        suggested_strategies: [
          {
            strategy: 'FIRST_WRITER_WINS',
            description: '保留最先创建的版本',
            result: null,
            confidence: 0.8,
            pros: ['避免重复创建', '先到先得公平', '操作简单'],
            cons: ['后来者工作丢失', '可能选择质量较低版本']
          },
          {
            strategy: 'RENAME_STRATEGY',
            description: '重命名其中一个资源',
            result: null,
            confidence: 0.9,
            pros: ['保留所有工作', '避免数据丢失', '清晰区分'],
            cons: ['可能造成混乱', '需要额外命名逻辑']
          }
        ]
      },
      {
        id: 'scenario-3',
        conflict_type: 'UPDATE_DELETE',
        description: '修改已删除资源冲突',
        local_data: {
          id: 'task-789',
          title: 'Updated Task Title',
          status: 'completed',
          notes: 'Added completion notes',
          updated_at: '2025-01-15T11:00:00Z'
        },
        remote_data: null, // 已删除
        local_timestamp: '2025-01-15T11:00:00Z',
        remote_timestamp: '2025-01-15T10:45:00Z',
        suggested_strategies: [
          {
            strategy: 'RESTORE_AND_UPDATE',
            description: '恢复资源并应用更新',
            result: null,
            confidence: 0.7,
            pros: ['保留更新内容', '避免工作丢失', '恢复误删除'],
            cons: ['可能违反删除意图', '需要验证删除原因']
          },
          {
            strategy: 'CONFIRM_DELETE',
            description: '确认删除，放弃更新',
            result: null,
            confidence: 0.6,
            pros: ['尊重删除决定', '避免冗余数据', '操作一致性'],
            cons: ['丢失更新工作', '可能删除重要信息']
          }
        ]
      }
    ];

    setScenarios(scenarios);
    setSelectedScenario(scenarios[0]);
  };

  // 执行冲突解决策略
  const executeResolutionStrategy = async (scenario: ConflictScenario, strategy: string) => {
    const startTime = Date.now();
    let result;
    let explanation = '';
    let algorithmSteps: string[] = [];

    switch (strategy) {
      case 'LAST_WRITER_WINS':
        result = ConflictResolutionAlgorithms.lastWriterWins(
          scenario.local_data,
          scenario.remote_data,
          scenario.local_timestamp,
          scenario.remote_timestamp
        );
        explanation = '基于时间戳比较，选择最新的版本';
        algorithmSteps = result.steps;
        break;

      case 'THREE_WAY_MERGE':
        // 模拟基础版本
        const baseData = {
          id: scenario.local_data?.id,
          name: 'Alice',
          email: 'alice@company.com',
          department: 'Engineering'
        };
        result = ConflictResolutionAlgorithms.threeWayMerge(
          baseData,
          scenario.local_data,
          scenario.remote_data
        );
        explanation = '使用三路合并算法智能合并两个版本';
        algorithmSteps = result.steps;
        break;

      case 'FIRST_WRITER_WINS':
        result = ConflictResolutionAlgorithms.lastWriterWins(
          scenario.remote_data,
          scenario.local_data,
          scenario.remote_timestamp,
          scenario.local_timestamp
        );
        explanation = '基于时间戳比较，选择最早的版本';
        algorithmSteps = result.steps;
        break;

      default:
        result = {
          result: '需要手动解决',
          confidence: 1.0,
          steps: ['手动解决策略需要人工干预']
        };
        explanation = '该策略需要人工审查和解决';
        algorithmSteps = result.steps;
    }

    const executionTime = Date.now() - startTime;

    const resolutionResult: ResolutionResult = {
      scenario_id: scenario.id,
      strategy,
      resolved_data: result.result,
      confidence: result.confidence,
      execution_time_ms: executionTime,
      explanation,
      algorithm_steps: algorithmSteps
    };

    setResolutionResults(prev => [...prev, resolutionResult]);
    return resolutionResult;
  };

  // 生成三路合并演示
  const generateThreeWayMergeDemo = () => {
    const demo: ThreeWayMergeDemo = {
      base_data: {
        title: 'Original Document',
        content: 'Original content here.',
        tags: ['draft'],
        priority: 1
      },
      local_data: {
        title: 'Updated Document Title',
        content: 'Original content here.',
        tags: ['draft', 'review'],
        priority: 2
      },
      remote_data: {
        title: 'Original Document',
        content: 'Updated content with more details.',
        tags: ['draft', 'important'],
        priority: 1
      },
      merge_result: {},
      conflicts: [],
      algorithm_steps: []
    };

    const mergeResult = ConflictResolutionAlgorithms.threeWayMerge(
      demo.base_data,
      demo.local_data,
      demo.remote_data
    );

    demo.merge_result = mergeResult.result;
    demo.conflicts = mergeResult.conflicts;
    demo.algorithm_steps = mergeResult.steps;

    setThreeWayDemo(demo);
  };

  const getConflictTypeColor = (type: string) => {
    switch (type) {
      case 'UPDATE_UPDATE': return 'orange';
      case 'CREATE_CREATE': return 'blue';
      case 'UPDATE_DELETE': return 'red';
      case 'DELETE_UPDATE': return 'purple';
      default: return 'default';
    }
  };

  const getConflictTypeIcon = (type: string) => {
    switch (type) {
      case 'UPDATE_UPDATE': return <GitMerge className="h-4 w-4" />;
      case 'CREATE_CREATE': return <Activity className="h-4 w-4" />;
      case 'UPDATE_DELETE': return <AlertTriangle className="h-4 w-4" />;
      case 'DELETE_UPDATE': return <AlertTriangle className="h-4 w-4" />;
      default: return <GitBranch className="h-4 w-4" />;
    }
  };

  const getStrategyIcon = (strategy: string) => {
    switch (strategy) {
      case 'LAST_WRITER_WINS': return <Clock className="h-4 w-4" />;
      case 'THREE_WAY_MERGE': return <GitMerge className="h-4 w-4" />;
      case 'MANUAL': return <Users className="h-4 w-4" />;
      case 'FIRST_WRITER_WINS': return <Clock className="h-4 w-4" />;
      default: return <Bot className="h-4 w-4" />;
    }
  };

  useEffect(() => {
    generateConflictScenarios();
    generateThreeWayMergeDemo();
  }, []);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">冲突解决策略学习</h1>
        <div className="flex space-x-2">
          <Button onClick={() => window.location.reload()} type="default" size="small">
            <RefreshCw className="h-4 w-4 mr-2" />
            重置演示
          </Button>
        </div>
      </div>

      {/* 冲突类型概述 */}
      <Card title={
        <div className="flex items-center">
          <AlertTriangle className="h-5 w-5 mr-2" />
          分布式系统冲突类型
        </div>
      }>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-orange-50 border border-orange-200 rounded p-4">
            <h4 className="font-medium text-orange-800 mb-2">UPDATE-UPDATE</h4>
            <p className="text-sm text-orange-600">
              两个节点同时修改同一资源，产生内容冲突
            </p>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded p-4">
            <h4 className="font-medium text-blue-800 mb-2">CREATE-CREATE</h4>
            <p className="text-sm text-blue-600">
              两个节点同时创建相同ID的资源
            </p>
          </div>
          <div className="bg-red-50 border border-red-200 rounded p-4">
            <h4 className="font-medium text-red-800 mb-2">UPDATE-DELETE</h4>
            <p className="text-sm text-red-600">
              一个节点修改资源，另一个节点删除了该资源
            </p>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded p-4">
            <h4 className="font-medium text-purple-800 mb-2">DELETE-UPDATE</h4>
            <p className="text-sm text-purple-600">
              一个节点删除资源，另一个节点修改了该资源
            </p>
          </div>
        </div>
      </Card>

      <Tabs defaultActiveKey="scenarios" className="w-full">
        <TabPane tab="冲突场景" key="scenarios" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* 场景选择 */}
            <div className="space-y-4">
              <h3 className="font-medium">选择冲突场景:</h3>
              {scenarios.map((scenario) => (
                <Card 
                  key={scenario.id}
                  className={`cursor-pointer transition-all ${
                    selectedScenario?.id === scenario.id ? 'border-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                  }`}
                  onClick={() => setSelectedScenario(scenario)}
                  bodyStyle={{ padding: 16 }}
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <Badge color={getConflictTypeColor(scenario.conflict_type)}>
                        {scenario.conflict_type}
                      </Badge>
                      {getConflictTypeIcon(scenario.conflict_type)}
                    </div>
                    <p className="text-sm font-medium">{scenario.description}</p>
                  </div>
                </Card>
              ))}
            </div>

            {/* 场景详情 */}
            {selectedScenario && (
              <div className="lg:col-span-2 space-y-4">
                <Card title={
                  <div className="flex items-center">
                    {getConflictTypeIcon(selectedScenario.conflict_type)}
                    <span className="ml-2">{selectedScenario.description}</span>
                  </div>
                }>
                  <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium mb-2">本地版本:</h4>
                        <div className="bg-blue-50 border rounded p-3">
                          <p className="text-xs text-muted-foreground mb-1">
                            时间戳: {selectedScenario.local_timestamp}
                          </p>
                          <pre className="text-xs overflow-x-auto">
                            {JSON.stringify(selectedScenario.local_data, null, 2)}
                          </pre>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-medium mb-2">远程版本:</h4>
                        <div className="bg-green-50 border rounded p-3">
                          <p className="text-xs text-muted-foreground mb-1">
                            时间戳: {selectedScenario.remote_timestamp}
                          </p>
                          <pre className="text-xs overflow-x-auto">
                            {selectedScenario.remote_data ? 
                              JSON.stringify(selectedScenario.remote_data, null, 2) : 
                              '已删除'
                            }
                          </pre>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <h4 className="font-medium mb-2">推荐解决策略:</h4>
                      <div className="space-y-2">
                        {selectedScenario.suggested_strategies.map((strategy, index) => (
                          <div key={index} className="border rounded p-3">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center space-x-2">
                                {getStrategyIcon(strategy.strategy)}
                                <span className="font-medium">{strategy.strategy}</span>
                              </div>
                              <div className="flex items-center space-x-2">
                                <Badge color="blue">
                                  信心度: {(strategy.confidence * 100).toFixed(0)}%
                                </Badge>
                                <Button
                                  size="small"
                                  onClick={() => executeResolutionStrategy(selectedScenario, strategy.strategy)}
                                >
                                  执行
                                </Button>
                              </div>
                            </div>
                            <p className="text-sm text-muted-foreground mb-2">
                              {strategy.description}
                            </p>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                              <div>
                                <span className="font-medium text-green-600">优点:</span>
                                <ul className="list-disc list-inside ml-2">
                                  {strategy.pros.map((pro, i) => (
                                    <li key={i}>{pro}</li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <span className="font-medium text-red-600">缺点:</span>
                                <ul className="list-disc list-inside ml-2">
                                  {strategy.cons.map((con, i) => (
                                    <li key={i}>{con}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </div>

          {/* 解决结果 */}
          {resolutionResults.length > 0 && (
            <Card title="解决结果历史">
              <div>
                <div className="space-y-4">
                  {resolutionResults.map((result, index) => (
                    <div key={index} className="border rounded p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          {getStrategyIcon(result.strategy)}
                          <span className="font-medium">{result.strategy}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge color="gray">
                            {result.execution_time_ms}ms
                          </Badge>
                          <Badge color="blue">
                            信心度: {(result.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">
                        {result.explanation}
                      </p>
                      <div className="bg-gray-50 rounded p-3">
                        <p className="text-xs font-medium mb-1">算法执行步骤:</p>
                        <div className="space-y-1">
                          {result.algorithm_steps.map((step, i) => (
                            <p key={i} className="text-xs font-mono">{step}</p>
                          ))}
                        </div>
                      </div>
                      <div className="mt-2">
                        <p className="text-xs font-medium mb-1">解决结果:</p>
                        <pre className="text-xs bg-white border rounded p-2 overflow-x-auto">
                          {JSON.stringify(result.resolved_data, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}
        </TabPane>

        <TabPane tab="解决策略" key="strategies" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card title="自动解决策略">
              <div className="space-y-4">
                <div className="border rounded p-3">
                  <h4 className="font-medium flex items-center mb-2">
                    <Clock className="h-4 w-4 mr-2" />
                    时间戳策略 (LWW/FWW)
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    基于事件发生时间自动选择版本
                  </p>
                  <pre className="text-xs bg-gray-100 p-2 rounded">
{`function lastWriterWins(local, remote) {
  if (local.timestamp > remote.timestamp) {
    return local.data;
  } else {
    return remote.data;
  }
}`}
                  </pre>
                </div>

                <div className="border rounded p-3">
                  <h4 className="font-medium flex items-center mb-2">
                    <GitMerge className="h-4 w-4 mr-2" />
                    自动合并策略
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    基于字段类型进行智能合并
                  </p>
                  <pre className="text-xs bg-gray-100 p-2 rounded">
{`function autoMerge(local, remote, schema) {
  const merged = {};
  for (const field in schema) {
    if (schema[field].type === 'array') {
      merged[field] = union(local[field], remote[field]);
    } else if (schema[field].type === 'number') {
      merged[field] = Math.max(local[field], remote[field]);
    }
  }
  return merged;
}`}
                  </pre>
                </div>
              </div>
            </Card>

            <Card title="交互式解决策略">
              <div className="space-y-4">
                <div className="border rounded p-3">
                  <h4 className="font-medium flex items-center mb-2">
                    <Users className="h-4 w-4 mr-2" />
                    用户选择策略
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    向用户展示冲突选项，由用户决定
                  </p>
                  <div className="bg-blue-50 border-l-4 border-blue-400 p-2">
                    <p className="text-xs">
                      优点: 100%准确性，考虑业务逻辑<br/>
                      缺点: 需要人工介入，可能产生延迟
                    </p>
                  </div>
                </div>

                <div className="border rounded p-3">
                  <h4 className="font-medium flex items-center mb-2">
                    <Bot className="h-4 w-4 mr-2" />
                    策略预测
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    基于历史解决模式预测最佳策略
                  </p>
                  <div className="bg-green-50 border-l-4 border-green-400 p-2">
                    <p className="text-xs">
                      使用机器学习分析用户历史选择，
                      预测当前冲突的最佳解决策略
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </TabPane>

        <TabPane tab="三路合并" key="merge-algorithm" className="space-y-4">
          {threeWayDemo && (
            <Card title={
              <div className="flex items-center">
                <GitMerge className="h-5 w-5 mr-2" />
                三路合并算法演示
              </div>
            }>
              <div>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
                  <div>
                    <h4 className="font-medium mb-2">基础版本 (Base)</h4>
                    <div className="bg-gray-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(threeWayDemo.base_data, null, 2)}
                      </pre>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">本地版本 (Local)</h4>
                    <div className="bg-blue-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(threeWayDemo.local_data, null, 2)}
                      </pre>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">远程版本 (Remote)</h4>
                    <div className="bg-green-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(threeWayDemo.remote_data, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>

                <div className="mb-4">
                  <h4 className="font-medium mb-2">算法执行过程:</h4>
                  <div className="bg-gray-800 text-green-400 p-4 rounded text-xs overflow-x-auto max-h-60">
                    {threeWayDemo.algorithm_steps.map((step, index) => (
                      <div key={index}>{step}</div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">合并结果:</h4>
                    <div className="bg-purple-50 border rounded p-3">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(threeWayDemo.merge_result, null, 2)}
                      </pre>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">冲突检测:</h4>
                    <div className="space-y-2">
                      {threeWayDemo.conflicts.length === 0 ? (
                        <div className="bg-green-50 border border-green-200 rounded p-3">
                          <CheckCircle className="h-4 w-4 text-green-500 inline mr-2" />
                          <span className="text-sm text-green-700">无冲突，自动合并成功</span>
                        </div>
                      ) : (
                        threeWayDemo.conflicts.map((conflict, index) => (
                          <div key={index} className="bg-red-50 border border-red-200 rounded p-3">
                            <AlertTriangle className="h-4 w-4 text-red-500 inline mr-2" />
                            <span className="text-sm text-red-700">
                              字段 '{conflict.field}' 存在 {conflict.conflict_type} 冲突
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          )}
        </TabPane>

        <TabPane tab="性能分析" key="performance" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card title="策略性能对比">
              <div>
                <div className="space-y-4">
                  <div className="border rounded p-3">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">Last Writer Wins</span>
                      <Badge color="gray">O(1)</Badge>
                    </div>
                    <Progress percent={95} className="mb-1" />
                    <p className="text-xs text-muted-foreground">
                      执行速度: 极快 | 准确性: 中等 | 自动化: 高
                    </p>
                  </div>

                  <div className="border rounded p-3">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">Three-way Merge</span>
                      <Badge color="gray">O(n)</Badge>
                    </div>
                    <Progress percent={70} className="mb-1" />
                    <p className="text-xs text-muted-foreground">
                      执行速度: 中等 | 准确性: 高 | 自动化: 中等
                    </p>
                  </div>

                  <div className="border rounded p-3">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">Manual Resolution</span>
                      <Badge color="gray">O(∞)</Badge>
                    </div>
                    <Progress percent={20} className="mb-1" />
                    <p className="text-xs text-muted-foreground">
                      执行速度: 慢 | 准确性: 最高 | 自动化: 无
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            <Card title="冲突解决统计">
              <div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {resolutionResults.length}
                    </p>
                    <p className="text-sm text-muted-foreground">总解决次数</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-green-600">
                      {resolutionResults.length > 0 ? 
                        (resolutionResults.reduce((sum, r) => sum + r.execution_time_ms, 0) / resolutionResults.length).toFixed(1) : 0}ms
                    </p>
                    <p className="text-sm text-muted-foreground">平均执行时间</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-purple-600">
                      {resolutionResults.length > 0 ? 
                        (resolutionResults.reduce((sum, r) => sum + r.confidence, 0) / resolutionResults.length * 100).toFixed(1) : 0}%
                    </p>
                    <p className="text-sm text-muted-foreground">平均信心度</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-orange-600">
                      {resolutionResults.filter(r => r.confidence > 0.8).length}
                    </p>
                    <p className="text-sm text-muted-foreground">高信心度解决</p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default ConflictResolutionLearningPage;