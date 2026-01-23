import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Steps,
  Tag,
  Progress,
  Timeline,
  Collapse,
  Typography,
  Space,
  Button,
  Tooltip,
  Alert,
  Divider,
} from 'antd'
import {
  BranchesOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  EyeOutlined,
  CodeOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'

const { Step } = Steps
const { Panel } = Collapse
const { Text, Title, Paragraph } = Typography

interface ReasoningChain {
  id: string
  problem: string
  strategy: string
  steps: ThoughtStep[]
  branches?: ReasoningBranch[]
  conclusion?: string
  confidence_score?: number
  total_duration_ms?: number
  created_at?: string
  completed_at?: string
}

interface ThoughtStep {
  id: string
  step_number: number
  step_type: string
  content: string
  reasoning: string
  confidence: number
  duration_ms?: number
  metadata?: Record<string, any>
}

interface ReasoningBranch {
  id: string
  parent_step_id?: string
  branch_reason: string
  priority: number
  is_active: boolean
  steps: ThoughtStep[]
}

interface ReasoningChainVisualizationProps {
  chain?: ReasoningChain | null
  streamingSteps?: ThoughtStep[]
  isExecuting?: boolean
}

export const ReasoningChainVisualization: React.FC<
  ReasoningChainVisualizationProps
> = ({ chain, streamingSteps = [], isExecuting = false }) => {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(true)
  const [viewMode, setViewMode] = useState<'steps' | 'timeline' | 'tree'>(
    'steps'
  )

  const allSteps = [...(chain?.steps || []), ...streamingSteps]
  const hasData = chain || streamingSteps.length > 0

  const getStepTypeIcon = (stepType: string) => {
    const icons = {
      observation: '👁️',
      analysis: '🧠',
      hypothesis: '💡',
      validation: '✅',
      reflection: '🤔',
      conclusion: '🎯',
    }
    return icons[stepType as keyof typeof icons] || '📝'
  }

  const getStepTypeColor = (stepType: string) => {
    const colors = {
      observation: 'blue',
      analysis: 'green',
      hypothesis: 'orange',
      validation: 'purple',
      reflection: 'cyan',
      conclusion: 'red',
    }
    return colors[stepType as keyof typeof colors] || 'default'
  }

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return { level: '高', color: 'success' }
    if (confidence >= 0.6) return { level: '中', color: 'warning' }
    return { level: '低', color: 'error' }
  }

  if (!hasData) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 text-6xl mb-4">🧠</div>
        <Title level={4} type="secondary">
          等待推理开始
        </Title>
        <Text type="secondary">
          请在左侧"推理输入"标签页配置推理参数并开始推理
        </Text>
      </div>
    )
  }

  return (
    <div className="reasoning-visualization">
      <Card size="small" className="mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div>
              <Text strong>推理策略：</Text>
              <Tag color="blue" className="ml-1">
                {chain?.strategy?.toUpperCase() || 'STREAMING'}
              </Tag>
            </div>

            {chain?.confidence_score && (
              <div>
                <Text strong>总体置信度：</Text>
                <Tag
                  color={getConfidenceLevel(chain.confidence_score).color}
                  className="ml-1"
                >
                  {Math.round(chain.confidence_score * 100)}%
                </Tag>
              </div>
            )}

            {isExecuting && (
              <div className="flex items-center">
                <ThunderboltOutlined className="text-green-500 mr-1" />
                <Text type="success">执行中</Text>
              </div>
            )}
          </div>

          <Space>
            <Button
              size="small"
              type={showTechnicalDetails ? 'primary' : 'default'}
              icon={<CodeOutlined />}
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            >
              技术细节
            </Button>

            <Button.Group size="small">
              <Button
                type={viewMode === 'steps' ? 'primary' : 'default'}
                onClick={() => setViewMode('steps')}
              >
                步骤视图
              </Button>
              <Button
                type={viewMode === 'timeline' ? 'primary' : 'default'}
                onClick={() => setViewMode('timeline')}
              >
                时间线
              </Button>
            </Button.Group>
          </Space>
        </div>
      </Card>

      {chain?.problem && (
        <Card title="推理问题" size="small" className="mb-4">
          <Paragraph className="text-lg font-medium">{chain.problem}</Paragraph>
        </Card>
      )}

      <div className="space-y-4">
        {viewMode === 'steps' && (
          <div>
            {allSteps.map((step, index) => (
              <Card
                key={step.id}
                title={
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-lg mr-2">
                        {getStepTypeIcon(step.step_type)}
                      </span>
                      <span>
                        步骤 {step.step_number}: {step.step_type.toUpperCase()}
                      </span>
                      <Tag
                        color={getStepTypeColor(step.step_type)}
                        className="ml-2"
                      >
                        {step.step_type}
                      </Tag>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Progress
                        type="circle"
                        size={40}
                        percent={Math.round(step.confidence * 100)}
                        format={() => `${Math.round(step.confidence * 100)}%`}
                      />
                      {step.duration_ms && (
                        <Tag icon={<ClockCircleOutlined />}>
                          {step.duration_ms}ms
                        </Tag>
                      )}
                    </div>
                  </div>
                }
                className="mb-4"
                size="small"
              >
                <div className="space-y-4">
                  <div>
                    <Text strong>推理内容：</Text>
                    <Paragraph className="bg-blue-50 p-3 rounded mt-2">
                      {step.content}
                    </Paragraph>
                  </div>

                  <div>
                    <Text strong>推理过程：</Text>
                    <Paragraph className="bg-green-50 p-3 rounded mt-2">
                      {step.reasoning}
                    </Paragraph>
                  </div>

                  {showTechnicalDetails && (
                    <Collapse size="small">
                      <Panel header="技术细节" key="tech">
                        <div className="space-y-2">
                          <div>
                            <strong>步骤ID:</strong> {step.id}
                          </div>
                          <div>
                            <strong>置信度:</strong>{' '}
                            {step.confidence.toFixed(3)}
                          </div>
                          <div>
                            <strong>执行时间:</strong>{' '}
                            {step.duration_ms || 'N/A'}ms
                          </div>
                          <div>
                            <strong>步骤类型:</strong> {step.step_type}
                          </div>
                        </div>
                      </Panel>
                    </Collapse>
                  )}
                </div>
              </Card>
            ))}
          </div>
        )}

        {viewMode === 'timeline' && (
          <Timeline
            pending={isExecuting ? '推理进行中...' : undefined}
            mode="left"
          >
            {allSteps.map((step, index) => (
              <Timeline.Item
                key={step.id}
                color={getStepTypeColor(step.step_type)}
                dot={
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-white border-2">
                    {getStepTypeIcon(step.step_type)}
                  </div>
                }
              >
                <div className="ml-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Tag color={getStepTypeColor(step.step_type)}>
                      {step.step_type}
                    </Tag>
                    <Tag>置信度: {Math.round(step.confidence * 100)}%</Tag>
                    {step.duration_ms && <Tag>{step.duration_ms}ms</Tag>}
                  </div>
                  <div className="text-gray-800 mb-1">{step.content}</div>
                  <div className="text-gray-600 text-sm">{step.reasoning}</div>
                </div>
              </Timeline.Item>
            ))}
          </Timeline>
        )}
      </div>

      {chain?.conclusion && (
        <Card
          title={
            <div className="flex items-center">
              <CheckCircleOutlined className="text-green-500 mr-2" />
              推理结论
            </div>
          }
          className="mt-4"
        >
          <Paragraph className="text-lg font-medium bg-green-50 p-4 rounded">
            {chain.conclusion}
          </Paragraph>

          {showTechnicalDetails && (
            <div className="mt-4 pt-4 border-t">
              <Row gutter={[16, 16]}>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {allSteps.length}
                    </div>
                    <div className="text-gray-500">推理步骤</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {chain.total_duration_ms || 'N/A'}
                    </div>
                    <div className="text-gray-500">总耗时(ms)</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {chain.confidence_score
                        ? `${(chain.confidence_score * 100).toFixed(1)}%`
                        : 'N/A'}
                    </div>
                    <div className="text-gray-500">平均置信度</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {chain.branches?.length || 0}
                    </div>
                    <div className="text-gray-500">分支数量</div>
                  </div>
                </Col>
              </Row>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
