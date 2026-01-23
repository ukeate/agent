import React, { useState } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Select,
  Progress,
  Tag,
  Alert,
  Typography,
  Space,
  Statistic,
  Collapse,
  Descriptions,
} from 'antd'
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
  ControlOutlined,
  BugOutlined,
} from '@ant-design/icons'

const { Text, Title } = Typography
const { Option } = Select
const { Panel } = Collapse

interface ReasoningValidation {
  step_id: string
  is_valid: boolean
  consistency_score: number
  issues: string[]
  suggestions: string[]
}

interface RecoveryStats {
  total_failures: number
  recovery_attempts: number
  recovery_success_rate: number
  strategy_effectiveness: Record<string, number>
}

interface ReasoningQualityControlProps {
  currentChain?: any
  validationResults?: ReasoningValidation | null
  recoveryStats?: RecoveryStats | null
}

export const ReasoningQualityControl: React.FC<
  ReasoningQualityControlProps
> = ({ currentChain, validationResults, recoveryStats }) => {
  const [selectedRecoveryStrategy, setSelectedRecoveryStrategy] =
    useState<string>('backtrack')
  const [isValidating, setIsValidating] = useState(false)
  const [isRecovering, setIsRecovering] = useState(false)

  const handleValidateChain = async () => {
    setIsValidating(true)
    // 这里会调用验证API
    setTimeout(() => setIsValidating(false), 2000)
  }

  const handleRecoverChain = async () => {
    setIsRecovering(true)
    // 这里会调用恢复API
    setTimeout(() => setIsRecovering(false), 3000)
  }

  const recoveryStrategies = {
    backtrack: {
      name: '回溯',
      description: '回到上一个高置信度步骤重新推理',
      icon: '⏪',
      effectiveness: recoveryStats?.strategy_effectiveness?.backtrack || 0,
    },
    branch: {
      name: '分支',
      description: '创建新的推理分支探索替代路径',
      icon: '🌿',
      effectiveness: recoveryStats?.strategy_effectiveness?.branch || 0,
    },
    restart: {
      name: '重启',
      description: '从头开始重新推理',
      icon: '🔄',
      effectiveness: recoveryStats?.strategy_effectiveness?.restart || 0,
    },
    refine: {
      name: '细化',
      description: '优化当前推理步骤的内容',
      icon: '✨',
      effectiveness: recoveryStats?.strategy_effectiveness?.refine || 0,
    },
    alternative: {
      name: '替代',
      description: '尝试不同的推理策略',
      icon: '🔀',
      effectiveness: recoveryStats?.strategy_effectiveness?.alternative || 0,
    },
  }

  const getValidationStatus = () => {
    if (!validationResults)
      return { status: 'warning', text: '未验证', color: 'orange' }
    if (validationResults.is_valid)
      return { status: 'success', text: '验证通过', color: 'green' }
    return { status: 'error', text: '验证失败', color: 'red' }
  }

  const validationStatus = getValidationStatus()

  return (
    <div className="reasoning-quality-control">
      <Row gutter={16} className="mb-6">
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="验证状态"
              value={validationStatus.text}
              prefix={
                validationStatus.status === 'success' ? (
                  <CheckCircleOutlined style={{ color: 'green' }} />
                ) : validationStatus.status === 'error' ? (
                  <ExclamationCircleOutlined style={{ color: 'red' }} />
                ) : (
                  <WarningOutlined style={{ color: 'orange' }} />
                )
              }
              valueStyle={{ color: validationStatus.color }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="一致性分数"
              value={validationResults?.consistency_score || 0}
              precision={3}
              suffix="/ 1.0"
              prefix={<ControlOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="恢复成功率"
              value={recoveryStats?.recovery_success_rate || 0}
              precision={1}
              suffix="%"
              prefix={<ReloadOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="失败次数"
              value={recoveryStats?.total_failures || 0}
              prefix={<BugOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card title="质量验证" className="mb-4">
        <Row gutter={16}>
          <Col span={16}>
            {validationResults ? (
              <div>
                <div className="mb-4">
                  <Tag color={validationStatus.color} className="mb-2">
                    {validationStatus.text}
                  </Tag>
                  <Text className="ml-2">
                    一致性分数:{' '}
                    {(validationResults.consistency_score * 100).toFixed(1)}%
                  </Text>
                </div>

                {validationResults.issues.length > 0 && (
                  <Alert
                    message="发现的问题"
                    description={
                      <ul className="mb-0">
                        {validationResults.issues.map((issue, index) => (
                          <li key={index}>{issue}</li>
                        ))}
                      </ul>
                    }
                    type="warning"
                    showIcon
                    className="mb-3"
                  />
                )}

                {validationResults.suggestions.length > 0 && (
                  <Alert
                    message="改进建议"
                    description={
                      <ul className="mb-0">
                        {validationResults.suggestions.map(
                          (suggestion, index) => (
                            <li key={index}>{suggestion}</li>
                          )
                        )}
                      </ul>
                    }
                    type="info"
                    showIcon
                  />
                )}
              </div>
            ) : (
              <div className="text-center text-gray-500 py-4">
                <WarningOutlined className="text-2xl mb-2" />
                <div>暂无验证结果</div>
                <Text type="secondary">点击右侧按钮开始验证推理链质量</Text>
              </div>
            )}
          </Col>

          <Col span={8}>
            <div className="text-center">
              <Button
                type="primary"
                icon={<CheckCircleOutlined />}
                loading={isValidating}
                onClick={handleValidateChain}
                disabled={!currentChain}
                size="large"
              >
                {isValidating ? '验证中...' : '验证推理链'}
              </Button>

              {!currentChain && (
                <div className="mt-2">
                  <Text type="secondary">请先开始推理</Text>
                </div>
              )}
            </div>
          </Col>
        </Row>
      </Card>

      <Card title="恢复控制" className="mb-4">
        <Row gutter={16}>
          <Col span={16}>
            <div className="mb-4">
              <Text strong>选择恢复策略：</Text>
              <Select
                value={selectedRecoveryStrategy}
                onChange={setSelectedRecoveryStrategy}
                style={{ width: '100%', marginTop: 8 }}
              >
                {Object.entries(recoveryStrategies).map(([key, strategy]) => (
                  <Option key={key} value={key}>
                    <div className="flex items-center">
                      <span className="mr-2">{strategy.icon}</span>
                      <div>
                        <div>{strategy.name}</div>
                        <div className="text-xs text-gray-500">
                          {strategy.description}
                        </div>
                      </div>
                    </div>
                  </Option>
                ))}
              </Select>
            </div>

            {selectedRecoveryStrategy && (
              <div className="bg-blue-50 p-3 rounded">
                <div className="flex items-center mb-2">
                  <span className="text-lg mr-2">
                    {recoveryStrategies[selectedRecoveryStrategy].icon}
                  </span>
                  <Text strong>
                    {recoveryStrategies[selectedRecoveryStrategy].name}
                  </Text>
                </div>
                <Text className="text-sm">
                  {recoveryStrategies[selectedRecoveryStrategy].description}
                </Text>
                <div className="mt-2">
                  <Text className="text-xs">
                    历史有效性:{' '}
                    {(
                      recoveryStrategies[selectedRecoveryStrategy]
                        .effectiveness * 100
                    ).toFixed(1)}
                    %
                  </Text>
                  <Progress
                    percent={
                      recoveryStrategies[selectedRecoveryStrategy]
                        .effectiveness * 100
                    }
                    size="small"
                    className="mt-1"
                  />
                </div>
              </div>
            )}
          </Col>

          <Col span={8}>
            <div className="text-center">
              <Button
                type="primary"
                danger
                icon={<ReloadOutlined />}
                loading={isRecovering}
                onClick={handleRecoverChain}
                disabled={!currentChain || validationResults?.is_valid === true}
                size="large"
              >
                {isRecovering ? '恢复中...' : '执行恢复'}
              </Button>

              <div className="mt-2">
                {!currentChain ? (
                  <Text type="secondary">请先开始推理</Text>
                ) : validationResults?.is_valid === true ? (
                  <Text type="secondary">推理链状态良好</Text>
                ) : (
                  <Text type="secondary">
                    将执行{recoveryStrategies[selectedRecoveryStrategy]?.name}
                    策略
                  </Text>
                )}
              </div>
            </div>
          </Col>
        </Row>
      </Card>

      <Card title="技术实现" size="small">
        <Collapse ghost>
          <Panel header="验证器详情" key="validators">
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="一致性验证器">
                检查推理步骤之间的逻辑连贯性和矛盾
              </Descriptions.Item>
              <Descriptions.Item label="置信度验证器">
                分析置信度水平和趋势变化
              </Descriptions.Item>
              <Descriptions.Item label="自我检查验证器">
                验证推理支持内容匹配度和循环推理
              </Descriptions.Item>
              <Descriptions.Item label="组合验证器">
                综合所有验证器结果给出最终评估
              </Descriptions.Item>
            </Descriptions>
          </Panel>

          <Panel header="恢复机制" key="recovery">
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="失败检测">
                自动识别推理问题和质量下降
              </Descriptions.Item>
              <Descriptions.Item label="检查点管理">
                保存推理状态以支持回溯操作
              </Descriptions.Item>
              <Descriptions.Item label="替代路径生成">
                基于问题分析生成新的推理策略
              </Descriptions.Item>
              <Descriptions.Item label="恢复管理器">
                协调整个恢复过程的执行
              </Descriptions.Item>
            </Descriptions>
          </Panel>
        </Collapse>
      </Card>
    </div>
  )
}
