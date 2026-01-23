import React, { useState, useEffect } from 'react'
import { logger } from '../../utils/logger'
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Select,
  Tabs,
  Table,
  Progress,
  Tooltip,
  Tag,
  Typography,
  Alert,
  Space,
  Divider,
  Collapse,
  Timeline,
  Statistic,
} from 'antd'
import {
  PlayCircleOutlined,
  SwapOutlined,
  ExperimentOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined,
  BulbOutlined,
} from '@ant-design/icons'
import { useReasoningStore } from '../../stores/reasoningStore'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { Option } = Select
const { TabPane } = Tabs
const { Panel } = Collapse

interface StrategyComparison {
  strategy: 'ZERO_SHOT' | 'FEW_SHOT' | 'AUTO_COT'
  chain?: any
  isExecuting: boolean
  result?: {
    conclusion: string
    confidence: number
    steps: number
    executionTime: number
    qualityScore: number
  }
}

interface ComparisonExample {
  id: string
  name: string
  problem: string
  description: string
  expectedOutcome: string
  difficulty: 'easy' | 'medium' | 'hard'
  tags: string[]
}

export const ReasoningStrategiesComparison: React.FC = () => {
  const { executeReasoning, isLoading } = useReasoningStore()

  const [customProblem, setCustomProblem] = useState('')
  const [selectedExample, setSelectedExample] = useState<string>('')
  const [comparisons, setComparisons] = useState<StrategyComparison[]>([
    { strategy: 'ZERO_SHOT', isExecuting: false },
    { strategy: 'FEW_SHOT', isExecuting: false },
    { strategy: 'AUTO_COT', isExecuting: false },
  ])
  const [activeTab, setActiveTab] = useState('setup')

  // 预定义的测试用例
  const examples: ComparisonExample[] = [
    {
      id: 'math_word_problem',
      name: '数学应用题',
      problem:
        '一个水池有两个进水管和一个出水管。第一个进水管每小时进水20升，第二个进水管每小时进水30升，出水管每小时出水15升。如果同时打开所有管子，多长时间能把容量为180升的空水池装满？',
      description: '典型的数学推理问题，需要多步计算和逻辑推理',
      expectedOutcome:
        '需要通过计算净进水速率(20+30-15=35升/小时)，然后计算时间(180÷35≈5.14小时)',
      difficulty: 'medium',
      tags: ['数学', '逻辑推理', '多步计算'],
    },
    {
      id: 'logical_puzzle',
      name: '逻辑推理谜题',
      problem:
        '有三个盒子：红盒子、蓝盒子、绿盒子。每个盒子里都有一个球，分别是红球、蓝球、绿球，但是球的颜色和盒子的颜色不一定相同。已知：红盒子里不是红球，蓝盒子里不是蓝球。请问每个盒子里是什么颜色的球？',
      description: '纯逻辑推理问题，需要通过排除法得出结论',
      expectedOutcome: '红盒子里是绿球，蓝盒子里是红球，绿盒子里是蓝球',
      difficulty: 'easy',
      tags: ['逻辑推理', '排除法', '约束满足'],
    },
    {
      id: 'complex_analysis',
      name: '复杂场景分析',
      problem:
        '一家科技公司最近用户流失率上升了15%，同时新功能使用率只有30%，客服投诉增加了25%，但是付费用户转化率提升了5%。分析可能的原因并提出改进建议。',
      description: '复杂的商业分析问题，需要考虑多个因素和它们之间的关系',
      expectedOutcome: '需要分析各指标间的关联性，提出针对性的改进策略',
      difficulty: 'hard',
      tags: ['商业分析', '多因素分析', '策略建议'],
    },
  ]

  const strategyInfo = {
    ZERO_SHOT: {
      name: 'Zero-shot CoT',
      description: '直接使用"让我们一步一步思考"的提示，不提供任何示例',
      advantages: ['简单直接', '无需准备示例', '适用性广'],
      disadvantages: ['可能缺乏具体指导', '对复杂问题效果有限'],
      bestFor: ['简单到中等复杂度问题', '快速推理场景'],
      technicalDetails: {
        promptTemplate: '让我们一步一步地思考这个问题...',
        implementation: 'BaseCoTEngine + ZeroShotCoTEngine',
        features: ['自动步骤生成', '置信度评估', '结论总结'],
      },
    },
    FEW_SHOT: {
      name: 'Few-shot CoT',
      description: '提供几个相似问题的推理示例，引导模型学习推理模式',
      advantages: ['有具体示例指导', '推理质量较高', '适合特定领域'],
      disadvantages: ['需要准备合适示例', '可能过拟合示例', '示例质量影响结果'],
      bestFor: ['特定领域问题', '需要标准化推理流程', '质量要求较高'],
      technicalDetails: {
        promptTemplate: '以下是一些类似问题的推理过程...',
        implementation: 'BaseCoTEngine + FewShotCoTEngine',
        features: ['示例匹配', '模式学习', '推理风格一致性'],
      },
    },
    AUTO_COT: {
      name: 'Auto-CoT',
      description: '自动识别问题类型并选择最适合的推理策略',
      advantages: ['智能策略选择', '适应性强', '无需手动配置'],
      disadvantages: ['复杂度较高', '可能判断错误', '执行时间较长'],
      bestFor: ['混合类型问题', '自动化场景', '需要最佳效果'],
      technicalDetails: {
        promptTemplate: '分析问题类型并选择推理策略...',
        implementation: 'BaseCoTEngine + AutoCoTEngine + 问题分类器',
        features: ['问题分类', '策略选择', '动态适应'],
      },
    },
  }

  const handleStartComparison = async () => {
    const problem = selectedExample
      ? examples.find(e => e.id === selectedExample)?.problem
      : customProblem

    if (!problem) {
      return
    }

    setActiveTab('results')

    // 重置所有比较结果
    setComparisons(prev =>
      prev.map(comp => ({
        ...comp,
        isExecuting: true,
        result: undefined,
        chain: undefined,
      }))
    )

    // 并行执行所有策略
    const promises = comparisons.map(async (comp, index) => {
      try {
        const startTime = Date.now()

        const request = {
          problem,
          strategy: comp.strategy,
          max_steps: 10,
          stream: false,
          enable_branching: true,
          context: selectedExample
            ? `这是一个${examples.find(e => e.id === selectedExample)?.description}`
            : undefined,
        }

        const chain = await executeReasoning(request)
        const executionTime = Date.now() - startTime

        // 计算质量分数
        const qualityScore = calculateQualityScore(chain)

        const result = {
          conclusion: chain.conclusion || '未得出结论',
          confidence: chain.confidence_score || 0,
          steps: chain.steps?.length || 0,
          executionTime,
          qualityScore,
        }

        setComparisons(prev =>
          prev.map((c, i) =>
            i === index ? { ...c, isExecuting: false, result, chain } : c
          )
        )
      } catch (error) {
        logger.error(`策略 ${comp.strategy} 执行失败:`, error)
        setComparisons(prev =>
          prev.map((c, i) =>
            i === index
              ? {
                  ...c,
                  isExecuting: false,
                  result: {
                    conclusion: '执行失败',
                    confidence: 0,
                    steps: 0,
                    executionTime: 0,
                    qualityScore: 0,
                  },
                }
              : c
          )
        )
      }
    })

    await Promise.all(promises)
  }

  const calculateQualityScore = (chain: any): number => {
    if (!chain || !chain.steps) return 0

    let score = 0

    // 基础分: 平均置信度 (40%)
    const avgConfidence = chain.confidence_score || 0
    score += avgConfidence * 0.4

    // 完整性分: 是否有结论 (20%)
    if (chain.conclusion) {
      score += 0.2
    }

    // 步骤多样性分: 不同类型步骤的比例 (20%)
    const stepTypes = new Set(chain.steps.map(s => s.step_type))
    const diversityScore = Math.min(stepTypes.size / 5, 1) * 0.2
    score += diversityScore

    // 逻辑连贯性分: 简单评估 (20%)
    const coherenceScore = chain.steps.length > 0 ? 0.2 : 0
    score += coherenceScore

    return Math.min(score, 1)
  }

  const getWinner = () => {
    const completedResults = comparisons
      .filter(c => c.result && !c.isExecuting)
      .map(c => ({ strategy: c.strategy, score: c.result!.qualityScore }))
      .sort((a, b) => b.score - a.score)

    return completedResults.length > 0 ? completedResults[0] : null
  }

  const winner = getWinner()

  return (
    <div className="strategies-comparison">
      <div className="mb-6">
        <Title level={3}>
          <SwapOutlined className="mr-2" />
          推理策略对比实验
        </Title>
        <Text type="secondary">
          同时运行不同的CoT推理策略，比较它们在相同问题上的表现
        </Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane
          tab={
            <span>
              <ExperimentOutlined />
              实验设置
            </span>
          }
          key="setup"
        >
          {/* 策略说明 */}
          <Row gutter={16} className="mb-6">
            {Object.entries(strategyInfo).map(([key, info]) => (
              <Col span={8} key={key}>
                <Card
                  title={info.name}
                  size="small"
                  className="h-full"
                  extra={
                    <Tooltip title={info.description}>
                      <InfoCircleOutlined />
                    </Tooltip>
                  }
                >
                  <div className="mb-3">
                    <Text type="secondary">{info.description}</Text>
                  </div>

                  <Collapse ghost size="small">
                    <Panel header="技术细节" key="tech">
                      <div className="space-y-2 text-sm">
                        <div>
                          <strong>实现:</strong>{' '}
                          {info.technicalDetails.implementation}
                        </div>
                        <div>
                          <strong>模板:</strong>{' '}
                          {info.technicalDetails.promptTemplate}
                        </div>
                        <div>
                          <strong>特性:</strong>
                        </div>
                        <ul className="ml-4">
                          {info.technicalDetails.features.map((feature, i) => (
                            <li key={i}>{feature}</li>
                          ))}
                        </ul>
                      </div>
                    </Panel>
                    <Panel header="优缺点分析" key="analysis">
                      <div className="space-y-2 text-sm">
                        <div>
                          <strong className="text-green-600">优势:</strong>
                          <ul className="ml-4">
                            {info.advantages.map((adv, i) => (
                              <li key={i}>{adv}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <strong className="text-red-600">劣势:</strong>
                          <ul className="ml-4">
                            {info.disadvantages.map((dis, i) => (
                              <li key={i}>{dis}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <strong>适用场景:</strong> {info.bestFor.join('、')}
                        </div>
                      </div>
                    </Panel>
                  </Collapse>
                </Card>
              </Col>
            ))}
          </Row>

          {/* 问题选择 */}
          <Card title="选择测试问题" className="mb-4">
            <Row gutter={16}>
              <Col span={12}>
                <div className="mb-4">
                  <Text strong>预设问题:</Text>
                  <Select
                    value={selectedExample}
                    onChange={setSelectedExample}
                    placeholder="选择一个预设问题"
                    style={{ width: '100%' }}
                    className="mt-2"
                  >
                    {examples.map(example => (
                      <Option key={example.id} value={example.id}>
                        <div>
                          <Text strong>{example.name}</Text>
                          <div className="text-xs text-gray-500 flex items-center space-x-2">
                            <Tag
                              color={
                                example.difficulty === 'easy'
                                  ? 'green'
                                  : example.difficulty === 'medium'
                                    ? 'orange'
                                    : 'red'
                              }
                              size="small"
                            >
                              {example.difficulty}
                            </Tag>
                            {example.tags.map(tag => (
                              <Tag key={tag} size="small">
                                {tag}
                              </Tag>
                            ))}
                          </div>
                        </div>
                      </Option>
                    ))}
                  </Select>
                </div>

                {selectedExample && (
                  <Card size="small" className="bg-blue-50">
                    <Text strong>问题预览:</Text>
                    <Paragraph className="mt-2">
                      {examples.find(e => e.id === selectedExample)?.problem}
                    </Paragraph>
                    <Text type="secondary">
                      {
                        examples.find(e => e.id === selectedExample)
                          ?.description
                      }
                    </Text>
                  </Card>
                )}
              </Col>

              <Col span={12}>
                <div className="mb-4">
                  <Text strong>自定义问题:</Text>
                  <TextArea
                    value={customProblem}
                    onChange={e => setCustomProblem(e.target.value)}
                    placeholder="或者输入您自己的问题..."
                    rows={6}
                    className="mt-2"
                  />
                </div>
              </Col>
            </Row>

            <Divider />

            <div className="text-center">
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                size="large"
                onClick={handleStartComparison}
                disabled={!selectedExample && !customProblem.trim()}
                loading={comparisons.some(c => c.isExecuting)}
              >
                开始对比实验
              </Button>
            </div>
          </Card>
        </TabPane>

        <TabPane
          tab={
            <span>
              <TrophyOutlined />
              对比结果
              {winner && (
                <Tag color="gold" className="ml-2">
                  {strategyInfo[winner.strategy].name} 胜出
                </Tag>
              )}
            </span>
          }
          key="results"
        >
          {/* 结果概览 */}
          <Row gutter={16} className="mb-6">
            {comparisons.map(comp => (
              <Col span={8} key={comp.strategy}>
                <Card
                  title={strategyInfo[comp.strategy].name}
                  loading={comp.isExecuting}
                  className={`${winner?.strategy === comp.strategy ? 'border-yellow-400 shadow-lg' : ''}`}
                  extra={
                    winner?.strategy === comp.strategy && (
                      <Tag color="gold">
                        <TrophyOutlined /> 最佳
                      </Tag>
                    )
                  }
                >
                  {comp.result ? (
                    <div className="space-y-3">
                      <div>
                        <Text type="secondary">质量分数</Text>
                        <div className="text-2xl font-bold text-blue-600">
                          {(comp.result.qualityScore * 100).toFixed(1)}
                        </div>
                        <Progress
                          percent={comp.result.qualityScore * 100}
                          showInfo={false}
                          strokeColor="#1890ff"
                        />
                      </div>

                      <Row gutter={8}>
                        <Col span={12}>
                          <Statistic
                            title="置信度"
                            value={comp.result.confidence * 100}
                            precision={1}
                            suffix="%"
                            valueStyle={{ fontSize: '16px' }}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic
                            title="步骤数"
                            value={comp.result.steps}
                            valueStyle={{ fontSize: '16px' }}
                          />
                        </Col>
                      </Row>

                      <div>
                        <Text type="secondary">
                          执行时间: {comp.result.executionTime}ms
                        </Text>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-gray-500">
                      等待执行结果...
                    </div>
                  )}
                </Card>
              </Col>
            ))}
          </Row>

          {/* 详细结果对比 */}
          {comparisons.some(c => c.result) && (
            <Card title="详细结果对比">
              <Table
                dataSource={comparisons
                  .filter(c => c.result)
                  .map(c => ({
                    strategy: c.strategy,
                    name: strategyInfo[c.strategy].name,
                    ...c.result!,
                  }))}
                pagination={false}
                size="small"
                columns={[
                  {
                    title: '策略',
                    key: 'strategy',
                    render: (_, record) => (
                      <div>
                        <Tag color="blue">{record.name}</Tag>
                        {winner?.strategy === record.strategy && (
                          <Tag color="gold">
                            <TrophyOutlined /> 最佳
                          </Tag>
                        )}
                      </div>
                    ),
                  },
                  {
                    title: '质量分数',
                    dataIndex: 'qualityScore',
                    sorter: (a, b) => a.qualityScore - b.qualityScore,
                    render: (score: number) => (
                      <div>
                        <span className="font-bold">
                          {(score * 100).toFixed(1)}
                        </span>
                        <Progress
                          percent={score * 100}
                          showInfo={false}
                          size="small"
                          className="mt-1"
                        />
                      </div>
                    ),
                  },
                  {
                    title: '置信度',
                    dataIndex: 'confidence',
                    sorter: (a, b) => a.confidence - b.confidence,
                    render: (confidence: number) =>
                      `${(confidence * 100).toFixed(1)}%`,
                  },
                  {
                    title: '推理步骤',
                    dataIndex: 'steps',
                    sorter: (a, b) => a.steps - b.steps,
                  },
                  {
                    title: '执行时间',
                    dataIndex: 'executionTime',
                    sorter: (a, b) => a.executionTime - b.executionTime,
                    render: (time: number) => `${time}ms`,
                  },
                  {
                    title: '结论',
                    dataIndex: 'conclusion',
                    width: 300,
                    render: (conclusion: string) => (
                      <Tooltip title={conclusion}>
                        <Text>
                          {conclusion.length > 50
                            ? `${conclusion.substring(0, 50)}...`
                            : conclusion}
                        </Text>
                      </Tooltip>
                    ),
                  },
                ]}
              />
            </Card>
          )}
        </TabPane>

        <TabPane
          tab={
            <span>
              <BulbOutlined />
              推理过程
            </span>
          }
          key="process"
        >
          <Row gutter={16}>
            {comparisons
              .filter(c => c.chain && c.chain.steps)
              .map(comp => (
                <Col span={8} key={comp.strategy}>
                  <Card
                    title={`${strategyInfo[comp.strategy].name} 推理过程`}
                    size="small"
                  >
                    <Timeline size="small">
                      {comp.chain.steps.map((step, index) => (
                        <Timeline.Item
                          key={index}
                          color={
                            step.confidence >= 0.8
                              ? 'green'
                              : step.confidence >= 0.6
                                ? 'blue'
                                : 'red'
                          }
                        >
                          <div>
                            <Text strong>步骤 {step.step_number}</Text>
                            <Tag size="small" className="ml-2">
                              {step.step_type}
                            </Tag>
                            <Tag color="blue" size="small" className="ml-1">
                              {(step.confidence * 100).toFixed(0)}%
                            </Tag>
                          </div>
                          <div className="mt-1 text-sm">
                            {step.content.length > 100
                              ? `${step.content.substring(0, 100)}...`
                              : step.content}
                          </div>
                        </Timeline.Item>
                      ))}
                    </Timeline>

                    {comp.chain.conclusion && (
                      <Alert
                        message="最终结论"
                        description={comp.chain.conclusion}
                        type="success"
                        showIcon
                        className="mt-4"
                      />
                    )}
                  </Card>
                </Col>
              ))}
          </Row>
        </TabPane>
      </Tabs>
    </div>
  )
}
