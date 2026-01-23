import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Progress } from '../components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import {
  Clock,
  GitBranch,
  ArrowRight,
  AlertTriangle,
  CheckCircle,
  Activity,
  Zap,
  RefreshCw,
  Timer,
  Hash,
  TrendingUp,
  Network,
  ArrowUpDown,
  GitMerge,
  Eye,
  Code,
  Play,
  Pause,
} from 'lucide-react'

// 向量时钟数据结构 - 直接展示算法实现
interface VectorClock {
  node_id: string
  clock: Record<string, number>
}

// 算法步骤
interface AlgorithmStep {
  id: string
  node_id: string
  operation: 'local_event' | 'send_message' | 'receive_message'
  description: string
  before_clock: VectorClock
  after_clock: VectorClock
  explanation: string
  code_snippet: string
}

// 因果关系比较结果
interface CausalRelation {
  clock1: VectorClock
  clock2: VectorClock
  relation: 'before' | 'after' | 'concurrent' | 'equal'
  algorithm_steps: string[]
}

const VectorClockAlgorithmPage: React.FC = () => {
  const [nodes, setNodes] = useState<VectorClock[]>([])
  const [algorithmSteps, setAlgorithmSteps] = useState<AlgorithmStep[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedClocks, setSelectedClocks] = useState<VectorClock[]>([])
  const [causalComparison, setCausalComparison] =
    useState<CausalRelation | null>(null)
  const [activeTab, setActiveTab] = useState('algorithm')

  // 向量时钟算法实现 - 教学版本
  const VectorClockAlgorithm = {
    // 本地事件处理
    handleLocalEvent: (clock: VectorClock): VectorClock => {
      const newClock = { ...clock, clock: { ...clock.clock } }
      newClock.clock[clock.node_id] = (newClock.clock[clock.node_id] || 0) + 1
      return newClock
    },

    // 发送消息时的时钟处理
    handleSendMessage: (clock: VectorClock): VectorClock => {
      const newClock = { ...clock, clock: { ...clock.clock } }
      newClock.clock[clock.node_id] = (newClock.clock[clock.node_id] || 0) + 1
      return newClock
    },

    // 接收消息时的时钟处理
    handleReceiveMessage: (
      localClock: VectorClock,
      messageClock: VectorClock
    ): VectorClock => {
      const newClock = { ...localClock, clock: {} }

      // 获取所有节点
      const allNodes = new Set([
        ...Object.keys(localClock.clock),
        ...Object.keys(messageClock.clock),
      ])

      // 对每个节点取最大值
      for (const node of allNodes) {
        const localValue = localClock.clock[node] || 0
        const messageValue = messageClock.clock[node] || 0
        newClock.clock[node] = Math.max(localValue, messageValue)
      }

      // 递增本地节点时钟
      newClock.clock[localClock.node_id] =
        (newClock.clock[localClock.node_id] || 0) + 1

      return newClock
    },

    // 比较两个向量时钟
    compareClocks: (
      clock1: VectorClock,
      clock2: VectorClock
    ): CausalRelation => {
      const allNodes = new Set([
        ...Object.keys(clock1.clock),
        ...Object.keys(clock2.clock),
      ])

      let clock1Less = true
      let clock2Less = true
      const steps: string[] = []

      steps.push(`比较向量时钟 ${clock1.node_id} 和 ${clock2.node_id}:`)
      steps.push(`节点集合: [${Array.from(allNodes).join(', ')}]`)

      for (const node of allNodes) {
        const val1 = clock1.clock[node] || 0
        const val2 = clock2.clock[node] || 0

        steps.push(`节点 ${node}: Clock1[${val1}] vs Clock2[${val2}]`)

        if (val1 > val2) {
          clock2Less = false
          steps.push(`  -> Clock1 > Clock2 for ${node}, Clock2 不能 ≤ Clock1`)
        } else if (val1 < val2) {
          clock1Less = false
          steps.push(`  -> Clock1 < Clock2 for ${node}, Clock1 不能 ≤ Clock2`)
        } else {
          steps.push(`  -> Clock1 = Clock2 for ${node}`)
        }
      }

      let relation: 'before' | 'after' | 'concurrent' | 'equal'
      if (clock1Less && !clock2Less) {
        relation = 'before'
        steps.push('结果: Clock1 → Clock2 (Clock1 发生在 Clock2 之前)')
      } else if (!clock1Less && clock2Less) {
        relation = 'after'
        steps.push('结果: Clock1 ← Clock2 (Clock1 发生在 Clock2 之后)')
      } else if (clock1Less && clock2Less) {
        relation = 'equal'
        steps.push('结果: Clock1 = Clock2 (两个时钟相等)')
      } else {
        relation = 'concurrent'
        steps.push('结果: Clock1 || Clock2 (两个事件并发)')
      }

      return { clock1, clock2, relation, algorithm_steps: steps }
    },
  }

  // 生成算法演示数据
  const generateAlgorithmDemo = () => {
    const nodeA: VectorClock = { node_id: 'A', clock: {} }
    const nodeB: VectorClock = { node_id: 'B', clock: {} }
    const nodeC: VectorClock = { node_id: 'C', clock: {} }

    const steps: AlgorithmStep[] = []

    // 步骤1: Node A 本地事件
    const step1_before = { ...nodeA }
    const step1_after = VectorClockAlgorithm.handleLocalEvent(nodeA)
    steps.push({
      id: 'step-1',
      node_id: 'A',
      operation: 'local_event',
      description: 'Node A 执行本地事件',
      before_clock: step1_before,
      after_clock: step1_after,
      explanation: '本地事件：递增自己的时钟值',
      code_snippet: `// 本地事件处理
clock['A'] = clock['A'] + 1  // 0 + 1 = 1
result: A[1]`,
    })

    // 步骤2: Node B 本地事件
    const step2_before = { ...nodeB }
    const step2_after = VectorClockAlgorithm.handleLocalEvent(nodeB)
    steps.push({
      id: 'step-2',
      node_id: 'B',
      operation: 'local_event',
      description: 'Node B 执行本地事件',
      before_clock: step2_before,
      after_clock: step2_after,
      explanation: '本地事件：递增自己的时钟值',
      code_snippet: `// 本地事件处理
clock['B'] = clock['B'] + 1  // 0 + 1 = 1
result: B[1]`,
    })

    // 步骤3: Node A 发送消息给 B
    const step3_before = { ...step1_after }
    const step3_after = VectorClockAlgorithm.handleSendMessage(step1_after)
    steps.push({
      id: 'step-3',
      node_id: 'A',
      operation: 'send_message',
      description: 'Node A 发送消息给 Node B',
      before_clock: step3_before,
      after_clock: step3_after,
      explanation: '发送消息：递增自己的时钟，随消息发送时钟',
      code_snippet: `// 发送消息处理
clock['A'] = clock['A'] + 1  // 1 + 1 = 2
send_message(to: B, vector_clock: A[2])
result: A[2]`,
    })

    // 步骤4: Node B 接收来自 A 的消息
    const step4_before = { ...step2_after }
    const step4_after = VectorClockAlgorithm.handleReceiveMessage(
      step2_after,
      step3_after
    )
    steps.push({
      id: 'step-4',
      node_id: 'B',
      operation: 'receive_message',
      description: 'Node B 接收来自 Node A 的消息',
      before_clock: step4_before,
      after_clock: step4_after,
      explanation: '接收消息：更新为每个节点的最大值，然后递增自己的时钟',
      code_snippet: `// 接收消息处理
received_clock: A[2]
local_clock: B[1]

// 合并时钟
clock['A'] = max(0, 2) = 2
clock['B'] = max(1, 0) = 1

// 递增本地时钟
clock['B'] = clock['B'] + 1 = 2
result: A[2], B[2]`,
    })

    // 步骤5: Node C 本地事件（并发）
    const step5_before = { ...nodeC }
    const step5_after = VectorClockAlgorithm.handleLocalEvent(nodeC)
    steps.push({
      id: 'step-5',
      node_id: 'C',
      operation: 'local_event',
      description: 'Node C 执行本地事件（与 A,B 并发）',
      before_clock: step5_before,
      after_clock: step5_after,
      explanation: '并发事件：C 的事件与 A,B 的事件序列并发',
      code_snippet: `// 本地事件处理（并发）
clock['C'] = clock['C'] + 1  // 0 + 1 = 1
// 注意：C 不知道 A,B 的时钟状态
result: C[1]`,
    })

    setAlgorithmSteps(steps)

    // 设置最终的节点状态
    setNodes([
      step4_after, // A[2], B[2]
      step5_after, // C[1]
    ])
  }

  // 自动播放算法步骤
  useEffect(() => {
    if (isPlaying && currentStep < algorithmSteps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(currentStep + 1)
      }, 3000)
      return () => clearTimeout(timer)
    } else if (currentStep >= algorithmSteps.length - 1) {
      setIsPlaying(false)
    }
  }, [isPlaying, currentStep, algorithmSteps.length])

  // 比较选中的时钟
  const compareSelectedClocks = () => {
    if (selectedClocks.length === 2) {
      const comparison = VectorClockAlgorithm.compareClocks(
        selectedClocks[0],
        selectedClocks[1]
      )
      setCausalComparison(comparison)
    }
  }

  // 选择时钟进行比较
  const toggleClockSelection = (clock: VectorClock) => {
    setSelectedClocks(prev => {
      const isSelected = prev.some(c => c.node_id === clock.node_id)
      if (isSelected) {
        return prev.filter(c => c.node_id !== clock.node_id)
      } else if (prev.length < 2) {
        return [...prev, clock]
      } else {
        return [prev[1], clock]
      }
    })
  }

  const formatClock = (clock: VectorClock) => {
    return Object.entries(clock.clock)
      .map(([node, value]) => `${node}[${value}]`)
      .join(', ')
  }

  const getOperationIcon = (operation: string) => {
    switch (operation) {
      case 'local_event':
        return <Activity className="h-4 w-4" />
      case 'send_message':
        return <ArrowRight className="h-4 w-4" />
      case 'receive_message':
        return <ArrowUpDown className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  const getOperationColor = (operation: string) => {
    switch (operation) {
      case 'local_event':
        return 'bg-blue-500'
      case 'send_message':
        return 'bg-orange-500'
      case 'receive_message':
        return 'bg-green-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getRelationColor = (relation: string) => {
    switch (relation) {
      case 'before':
        return 'text-blue-600'
      case 'after':
        return 'text-green-600'
      case 'concurrent':
        return 'text-orange-600'
      case 'equal':
        return 'text-purple-600'
      default:
        return 'text-gray-600'
    }
  }

  useEffect(() => {
    generateAlgorithmDemo()
  }, [])

  useEffect(() => {
    if (selectedClocks.length === 2) {
      compareSelectedClocks()
    } else {
      setCausalComparison(null)
    }
  }, [selectedClocks])

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">向量时钟算法学习</h1>
        <div className="flex space-x-2">
          <Button
            onClick={() => setIsPlaying(!isPlaying)}
            variant="outline"
            size="sm"
          >
            {isPlaying ? (
              <Pause className="h-4 w-4 mr-2" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            {isPlaying ? '暂停' : '播放'}
          </Button>
          <Button
            onClick={() => {
              setCurrentStep(0)
              setIsPlaying(false)
            }}
            variant="outline"
            size="sm"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            重置
          </Button>
        </div>
      </div>

      {/* 算法概念说明 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Hash className="h-5 w-5 mr-2" />
            向量时钟算法核心概念
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 border border-blue-200 rounded p-4">
            <h4 className="font-medium text-blue-800 mb-2">
              偏序关系 (Happens-Before)
            </h4>
            <p className="text-sm text-blue-600">
              事件 A → 事件 B：当且仅当对所有节点 i，VC(A)[i] ≤ VC(B)[i]，
              且存在节点 j 使得 VC(A)[j] &lt; VC(B)[j]
            </p>
          </div>
          <div className="bg-green-50 border border-green-200 rounded p-4">
            <h4 className="font-medium text-green-800 mb-2">
              并发关系 (Concurrent)
            </h4>
            <p className="text-sm text-green-600">
              事件 A || 事件 B：当 A 不先于 B，且 B 不先于 A 时，
              两个事件为并发关系
            </p>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded p-4">
            <h4 className="font-medium text-purple-800 mb-2">时钟更新规则</h4>
            <p className="text-sm text-purple-600">
              本地事件：VC[i]++；发送消息：VC[i]++，附带时钟； 接收消息：VC[j] =
              max(VC[j], received_VC[j])，然后 VC[i]++
            </p>
          </div>
        </CardContent>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="algorithm">算法演示</TabsTrigger>
          <TabsTrigger value="comparison">因果关系比较</TabsTrigger>
          <TabsTrigger value="implementation">代码实现</TabsTrigger>
          <TabsTrigger value="analysis">复杂度分析</TabsTrigger>
        </TabsList>

        <TabsContent value="algorithm" className="space-y-4">
          {/* 算法步骤控制 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Play className="h-5 w-5 mr-2" />
                算法步骤演示 (步骤 {currentStep + 1} / {algorithmSteps.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Progress
                  value={((currentStep + 1) / algorithmSteps.length) * 100}
                  className="w-full"
                />

                <div className="flex justify-center space-x-2">
                  <Button
                    onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                    disabled={currentStep === 0}
                    size="sm"
                  >
                    上一步
                  </Button>
                  <Button
                    onClick={() =>
                      setCurrentStep(
                        Math.min(algorithmSteps.length - 1, currentStep + 1)
                      )
                    }
                    disabled={currentStep === algorithmSteps.length - 1}
                    size="sm"
                  >
                    下一步
                  </Button>
                </div>

                {algorithmSteps[currentStep] && (
                  <div className="border rounded-lg p-4 bg-gray-50">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        {getOperationIcon(
                          algorithmSteps[currentStep].operation
                        )}
                        <h4 className="font-medium">
                          {algorithmSteps[currentStep].description}
                        </h4>
                      </div>
                      <Badge
                        className={getOperationColor(
                          algorithmSteps[currentStep].operation
                        )}
                      >
                        {algorithmSteps[currentStep].operation}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium mb-2">执行前:</p>
                        <div className="bg-white border rounded p-2">
                          <span className="font-mono text-sm">
                            {formatClock(
                              algorithmSteps[currentStep].before_clock
                            ) || '初始状态'}
                          </span>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium mb-2">执行后:</p>
                        <div className="bg-white border rounded p-2">
                          <span className="font-mono text-sm">
                            {formatClock(
                              algorithmSteps[currentStep].after_clock
                            )}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <p className="text-sm font-medium mb-2">算法解释:</p>
                      <p className="text-sm text-gray-600">
                        {algorithmSteps[currentStep].explanation}
                      </p>
                    </div>

                    <div className="mt-4">
                      <p className="text-sm font-medium mb-2">代码实现:</p>
                      <pre className="bg-gray-800 text-green-400 p-3 rounded text-xs overflow-x-auto">
                        {algorithmSteps[currentStep].code_snippet}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <GitMerge className="h-5 w-5 mr-2" />
                因果关系比较器
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium mb-2">
                    选择两个向量时钟进行因果关系比较 (已选择:{' '}
                    {selectedClocks.length}/2)
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                    {nodes.map((clock, index) => (
                      <Button
                        key={index}
                        variant={
                          selectedClocks.some(c => c.node_id === clock.node_id)
                            ? 'default'
                            : 'outline'
                        }
                        onClick={() => toggleClockSelection(clock)}
                        className="justify-start"
                      >
                        <Clock className="h-4 w-4 mr-2" />
                        {clock.node_id}: {formatClock(clock)}
                      </Button>
                    ))}
                  </div>
                </div>

                {causalComparison && (
                  <div className="border rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-4">
                      <GitBranch
                        className={`h-5 w-5 ${getRelationColor(causalComparison.relation)}`}
                      />
                      <h4
                        className={`font-medium ${getRelationColor(causalComparison.relation)}`}
                      >
                        因果关系: {causalComparison.relation.toUpperCase()}
                      </h4>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                      <div className="bg-blue-50 rounded p-3">
                        <h5 className="font-medium text-blue-800">时钟 1</h5>
                        <p className="font-mono text-sm">
                          {formatClock(causalComparison.clock1)}
                        </p>
                      </div>
                      <div className="bg-green-50 rounded p-3">
                        <h5 className="font-medium text-green-800">时钟 2</h5>
                        <p className="font-mono text-sm">
                          {formatClock(causalComparison.clock2)}
                        </p>
                      </div>
                    </div>

                    <div className="bg-gray-50 rounded p-3">
                      <h5 className="font-medium mb-2">算法比较步骤:</h5>
                      <div className="space-y-1">
                        {causalComparison.algorithm_steps.map((step, index) => (
                          <p
                            key={index}
                            className="text-xs font-mono text-gray-700"
                          >
                            {step}
                          </p>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="implementation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Code className="h-5 w-5 mr-2" />
                完整代码实现
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">向量时钟数据结构</h4>
                  <pre className="bg-gray-800 text-green-400 p-4 rounded text-sm overflow-x-auto">
                    {`interface VectorClock {
  node_id: string;
  clock: Record<string, number>;
}

class VectorClockManager {
  private clocks: Map<string, VectorClock> = new Map();

  // 本地事件处理
  handleLocalEvent(nodeId: string): VectorClock {
    const clock = this.getOrCreateClock(nodeId);
    clock.clock[nodeId] = (clock.clock[nodeId] || 0) + 1;
    return clock;
  }

  // 发送消息处理
  handleSendMessage(nodeId: string): VectorClock {
    const clock = this.getOrCreateClock(nodeId);
    clock.clock[nodeId] = (clock.clock[nodeId] || 0) + 1;
    return clock; // 随消息发送此时钟
  }

  // 接收消息处理
  handleReceiveMessage(
    nodeId: string, 
    receivedClock: VectorClock
  ): VectorClock {
    const localClock = this.getOrCreateClock(nodeId);
    
    // 获取所有节点
    const allNodes = new Set([
      ...Object.keys(localClock.clock),
      ...Object.keys(receivedClock.clock)
    ]);

    // 更新为最大值
    for (const node of allNodes) {
      const localValue = localClock.clock[node] || 0;
      const receivedValue = receivedClock.clock[node] || 0;
      localClock.clock[node] = Math.max(localValue, receivedValue);
    }

    // 递增本地时钟
    localClock.clock[nodeId] = (localClock.clock[nodeId] || 0) + 1;
    
    return localClock;
  }

  // 比较向量时钟
  compareClock(clock1: VectorClock, clock2: VectorClock): string {
    const allNodes = new Set([
      ...Object.keys(clock1.clock),
      ...Object.keys(clock2.clock)
    ]);

    let clock1Less = true;
    let clock2Less = true;

    for (const node of allNodes) {
      const val1 = clock1.clock[node] || 0;
      const val2 = clock2.clock[node] || 0;

      if (val1 > val2) {
        clock2Less = false;
      } else if (val1 < val2) {
        clock1Less = false;
      }
    }

    if (clock1Less && !clock2Less) return 'before';
    if (!clock1Less && clock2Less) return 'after';
    if (clock1Less && clock2Less) return 'equal';
    return 'concurrent';
  }
}`}
                  </pre>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <TrendingUp className="h-5 w-5 mr-2" />
                算法复杂度分析
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-blue-50 border border-blue-200 rounded p-4">
                    <h4 className="font-medium text-blue-800 mb-2">
                      时间复杂度
                    </h4>
                    <ul className="text-sm text-blue-600 space-y-1">
                      <li>
                        • <strong>本地事件:</strong> O(1)
                      </li>
                      <li>
                        • <strong>发送消息:</strong> O(1)
                      </li>
                      <li>
                        • <strong>接收消息:</strong> O(n) - n为节点数
                      </li>
                      <li>
                        • <strong>时钟比较:</strong> O(n) - n为节点数
                      </li>
                    </ul>
                  </div>

                  <div className="bg-green-50 border border-green-200 rounded p-4">
                    <h4 className="font-medium text-green-800 mb-2">
                      空间复杂度
                    </h4>
                    <ul className="text-sm text-green-600 space-y-1">
                      <li>
                        • <strong>每个节点:</strong> O(n) - 存储n个节点的时钟
                      </li>
                      <li>
                        • <strong>消息开销:</strong> O(n) - 每条消息携带向量时钟
                      </li>
                      <li>
                        • <strong>总空间:</strong> O(n²) -
                        n个节点各自维护n维向量
                      </li>
                    </ul>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-orange-50 border border-orange-200 rounded p-4">
                    <h4 className="font-medium text-orange-800 mb-2">
                      优化策略
                    </h4>
                    <ul className="text-sm text-orange-600 space-y-1">
                      <li>
                        • <strong>时钟压缩:</strong> 移除非活跃节点的时钟条目
                      </li>
                      <li>
                        • <strong>增量传输:</strong> 只发送变化的时钟条目
                      </li>
                      <li>
                        • <strong>批量更新:</strong> 批量处理多个事件的时钟更新
                      </li>
                      <li>
                        • <strong>层次化时钟:</strong> 使用树形结构减少维度
                      </li>
                    </ul>
                  </div>

                  <div className="bg-purple-50 border border-purple-200 rounded p-4">
                    <h4 className="font-medium text-purple-800 mb-2">
                      应用场景
                    </h4>
                    <ul className="text-sm text-purple-600 space-y-1">
                      <li>
                        • <strong>分布式数据库:</strong> 事务排序和一致性
                      </li>
                      <li>
                        • <strong>版本控制:</strong> Git等系统的分支合并
                      </li>
                      <li>
                        • <strong>消息系统:</strong> 保证消息的因果顺序
                      </li>
                      <li>
                        • <strong>协作系统:</strong> 实时编辑冲突检测
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="mt-6 bg-gray-50 border rounded p-4">
                <h4 className="font-medium mb-2">算法权衡分析</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium text-green-600">优点:</p>
                    <ul className="space-y-1 text-gray-600">
                      <li>• 精确的因果关系检测</li>
                      <li>• 去中心化，无需全局时钟</li>
                      <li>• 支持任意网络拓扑</li>
                      <li>• 容错性强</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium text-red-600">缺点:</p>
                    <ul className="space-y-1 text-gray-600">
                      <li>• 空间开销随节点数平方增长</li>
                      <li>• 网络消息开销较大</li>
                      <li>• 无法提供全局时间排序</li>
                      <li>• 并发事件较多时效率低</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default VectorClockAlgorithmPage
