import React, { useState } from 'react'
import { logger } from '../../utils/logger'
import {
  Card,
  Row,
  Col,
  Form,
  Input,
  Select,
  Switch,
  Button,
  Space,
  Tag,
  Alert,
  Collapse,
  InputNumber,
  Typography,
  Divider,
  Tooltip,
} from 'antd'
import {
  ThunderboltOutlined,
  SettingOutlined,
  ExperimentOutlined,
  InfoCircleOutlined,
  SaveOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons'

const { Option } = Select
const { Panel } = Collapse
const { TextArea } = Input
const { Title, Text, Paragraph } = Typography

interface EnvironmentConfig {
  name: string
  description: string
  stateSpaceType: 'discrete' | 'continuous' | 'hybrid'
  actionSpaceType: 'discrete' | 'continuous'
  stateFeatures: StateFeature[]
  actions: Action[]
  rewardFunction: RewardFunction
}

interface StateFeature {
  name: string
  type: 'continuous' | 'discrete' | 'categorical'
  min?: number
  max?: number
  categories?: string[]
  description: string
}

interface Action {
  id: string
  name: string
  description: string
}

interface RewardFunction {
  type: 'simple' | 'shaped' | 'multi_objective'
  baseReward: number
  shaping: {
    enabled: boolean
    factors: { [key: string]: number }
  }
  penalties: {
    enabled: boolean
    factors: { [key: string]: number }
  }
}

const QLearningEnvironmentPanel: React.FC = () => {
  const [form] = Form.useForm()
  const [environments, setEnvironments] = useState<EnvironmentConfig[]>([
    {
      name: 'GridWorld',
      description: '经典网格世界环境，智能体需要找到从起点到终点的最优路径',
      stateSpaceType: 'discrete',
      actionSpaceType: 'discrete',
      stateFeatures: [
        {
          name: 'x_position',
          type: 'discrete',
          min: 0,
          max: 9,
          description: 'X坐标位置',
        },
        {
          name: 'y_position',
          type: 'discrete',
          min: 0,
          max: 9,
          description: 'Y坐标位置',
        },
      ],
      actions: [
        { id: 'up', name: '向上', description: '向上移动一格' },
        { id: 'down', name: '向下', description: '向下移动一格' },
        { id: 'left', name: '向左', description: '向左移动一格' },
        { id: 'right', name: '向右', description: '向右移动一格' },
      ],
      rewardFunction: {
        type: 'simple',
        baseReward: -0.1,
        shaping: {
          enabled: true,
          factors: {
            distance_to_goal: -0.1,
            step_penalty: -0.01,
          },
        },
        penalties: {
          enabled: true,
          factors: {
            wall_collision: -1.0,
            out_of_bounds: -1.0,
          },
        },
      },
    },
    {
      name: 'CartPole',
      description: '倒立摆控制问题，需要通过左右移动保持杆子平衡',
      stateSpaceType: 'continuous',
      actionSpaceType: 'discrete',
      stateFeatures: [
        {
          name: 'cart_position',
          type: 'continuous',
          min: -2.4,
          max: 2.4,
          description: '小车位置',
        },
        {
          name: 'cart_velocity',
          type: 'continuous',
          min: -3.0,
          max: 3.0,
          description: '小车速度',
        },
        {
          name: 'pole_angle',
          type: 'continuous',
          min: -0.2,
          max: 0.2,
          description: '杆子角度',
        },
        {
          name: 'pole_velocity',
          type: 'continuous',
          min: -2.0,
          max: 2.0,
          description: '杆子角速度',
        },
      ],
      actions: [
        { id: 'left', name: '向左推', description: '向左推动小车' },
        { id: 'right', name: '向右推', description: '向右推动小车' },
      ],
      rewardFunction: {
        type: 'simple',
        baseReward: 1.0,
        shaping: {
          enabled: false,
          factors: {},
        },
        penalties: {
          enabled: true,
          factors: {
            episode_end: -100.0,
          },
        },
      },
    },
  ])

  const [selectedEnvironment, setSelectedEnvironment] =
    useState<string>('GridWorld')
  const [isCustomizing, setIsCustomizing] = useState(false)

  const getSpaceTypeColor = (type: string) => {
    switch (type) {
      case 'discrete':
        return 'blue'
      case 'continuous':
        return 'green'
      case 'hybrid':
        return 'orange'
      default:
        return 'default'
    }
  }

  const getFeatureTypeColor = (type: string) => {
    switch (type) {
      case 'continuous':
        return 'green'
      case 'discrete':
        return 'blue'
      case 'categorical':
        return 'purple'
      default:
        return 'default'
    }
  }

  const currentEnvironment = environments.find(
    env => env.name === selectedEnvironment
  )

  const handleSaveEnvironment = async (values: any) => {
    try {
      logger.log('保存环境配置:', values)
      // 这里应该调用API保存配置
      setIsCustomizing(false)
    } catch (error) {
      logger.error('保存环境配置失败:', error)
    }
  }

  return (
    <Row gutter={[16, 16]}>
      {/* 环境选择和控制 */}
      <Col span={24}>
        <Card>
          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
            <Space>
              <ThunderboltOutlined
                style={{ fontSize: '16px', color: '#1890ff' }}
              />
              <Text strong>强化学习环境配置</Text>
              <Select
                value={selectedEnvironment}
                onChange={setSelectedEnvironment}
                style={{ width: 200 }}
              >
                {environments.map(env => (
                  <Option key={env.name} value={env.name}>
                    <Space>
                      <ExperimentOutlined />
                      {env.name}
                    </Space>
                  </Option>
                ))}
              </Select>
            </Space>

            <Space>
              <Button
                type={isCustomizing ? 'default' : 'primary'}
                icon={<SettingOutlined />}
                onClick={() => setIsCustomizing(!isCustomizing)}
              >
                {isCustomizing ? '取消自定义' : '自定义环境'}
              </Button>
              <Button icon={<ReloadOutlined />}>重置配置</Button>
              <Button type="primary" icon={<PlayCircleOutlined />}>
                测试环境
              </Button>
            </Space>
          </Space>
        </Card>
      </Col>

      {/* 环境描述 */}
      {currentEnvironment && (
        <Col span={24}>
          <Alert
            message={`${currentEnvironment.name} 环境`}
            description={currentEnvironment.description}
            type="info"
            showIcon
          />
        </Col>
      )}

      {/* 环境详细配置 */}
      <Col span={16}>
        <Space direction="vertical" style={{ width: '100%' }}>
          {/* 状态空间配置 */}
          <Card
            title={
              <Space>
                <InfoCircleOutlined />
                状态空间配置
                {currentEnvironment && (
                  <Tag
                    color={getSpaceTypeColor(currentEnvironment.stateSpaceType)}
                  >
                    {currentEnvironment.stateSpaceType}
                  </Tag>
                )}
              </Space>
            }
          >
            {currentEnvironment && (
              <Row gutter={16}>
                {currentEnvironment.stateFeatures.map((feature, index) => (
                  <Col span={12} key={index}>
                    <Card size="small" hoverable>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Tag color={getFeatureTypeColor(feature.type)}>
                            {feature.type}
                          </Tag>
                          <Text strong>{feature.name}</Text>
                        </div>

                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {feature.description}
                        </Text>

                        {feature.type !== 'categorical' && (
                          <div>
                            <Text type="secondary">范围: </Text>
                            <Tag color="default">
                              [{feature.min}, {feature.max}]
                            </Tag>
                          </div>
                        )}

                        {feature.categories && (
                          <div>
                            <Text type="secondary">类别: </Text>
                            <Space wrap>
                              {feature.categories.map(cat => (
                                <Tag key={cat} size="small">
                                  {cat}
                                </Tag>
                              ))}
                            </Space>
                          </div>
                        )}
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            )}

            {isCustomizing && (
              <div
                style={{
                  marginTop: 16,
                  padding: 16,
                  backgroundColor: '#fafafa',
                  borderRadius: '4px',
                }}
              >
                <Button type="dashed" block icon={<ExperimentOutlined />}>
                  添加新的状态特征
                </Button>
              </div>
            )}
          </Card>

          {/* 动作空间配置 */}
          <Card
            title={
              <Space>
                <PlayCircleOutlined />
                动作空间配置
                {currentEnvironment && (
                  <Tag
                    color={getSpaceTypeColor(
                      currentEnvironment.actionSpaceType
                    )}
                  >
                    {currentEnvironment.actionSpaceType}
                  </Tag>
                )}
              </Space>
            }
          >
            {currentEnvironment && (
              <Row gutter={16}>
                {currentEnvironment.actions.map((action, index) => (
                  <Col span={8} key={index}>
                    <Card size="small" hoverable>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <PlayCircleOutlined
                            style={{ color: '#1890ff', marginRight: '4px' }}
                          />
                          <Text strong>{action.name}</Text>
                        </div>

                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {action.description}
                        </Text>

                        <Tag size="small" color="blue">
                          ID: {action.id}
                        </Tag>
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            )}

            {isCustomizing && (
              <div
                style={{
                  marginTop: 16,
                  padding: 16,
                  backgroundColor: '#fafafa',
                  borderRadius: '4px',
                }}
              >
                <Button type="dashed" block icon={<PlayCircleOutlined />}>
                  添加新动作
                </Button>
              </div>
            )}
          </Card>

          {/* 奖励函数配置 */}
          <Card
            title={
              <Space>
                <ThunderboltOutlined />
                奖励函数配置
              </Space>
            }
          >
            {currentEnvironment && (
              <Collapse defaultActiveKey={['basic']}>
                <Panel header="基础奖励设置" key="basic">
                  <Row gutter={16}>
                    <Col span={8}>
                      <div>
                        <Text type="secondary">奖励类型</Text>
                        <br />
                        <Tag color="blue">
                          {currentEnvironment.rewardFunction.type}
                        </Tag>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div>
                        <Text type="secondary">基础奖励</Text>
                        <br />
                        <Text strong>
                          {currentEnvironment.rewardFunction.baseReward}
                        </Text>
                      </div>
                    </Col>
                    <Col span={8}>
                      <div>
                        <Text type="secondary">奖励塑形</Text>
                        <br />
                        <Tag
                          color={
                            currentEnvironment.rewardFunction.shaping.enabled
                              ? 'green'
                              : 'default'
                          }
                        >
                          {currentEnvironment.rewardFunction.shaping.enabled
                            ? '启用'
                            : '禁用'}
                        </Tag>
                      </div>
                    </Col>
                  </Row>
                </Panel>

                {currentEnvironment.rewardFunction.shaping.enabled && (
                  <Panel header="奖励塑形因子" key="shaping">
                    <Row gutter={16}>
                      {Object.entries(
                        currentEnvironment.rewardFunction.shaping.factors
                      ).map(([key, value]) => (
                        <Col span={8} key={key}>
                          <Space direction="vertical">
                            <Text type="secondary">{key}</Text>
                            <Tag color={value < 0 ? 'red' : 'green'}>
                              {value > 0 ? '+' : ''}
                              {value}
                            </Tag>
                          </Space>
                        </Col>
                      ))}
                    </Row>
                  </Panel>
                )}

                {currentEnvironment.rewardFunction.penalties.enabled && (
                  <Panel header="惩罚因子" key="penalties">
                    <Row gutter={16}>
                      {Object.entries(
                        currentEnvironment.rewardFunction.penalties.factors
                      ).map(([key, value]) => (
                        <Col span={8} key={key}>
                          <Space direction="vertical">
                            <Text type="secondary">{key}</Text>
                            <Tag color="red">{value}</Tag>
                          </Space>
                        </Col>
                      ))}
                    </Row>
                  </Panel>
                )}
              </Collapse>
            )}

            {isCustomizing && (
              <div
                style={{
                  marginTop: 16,
                  padding: 16,
                  backgroundColor: '#fafafa',
                  borderRadius: '4px',
                }}
              >
                <Button type="dashed" block icon={<ThunderboltOutlined />}>
                  添加自定义奖励规则
                </Button>
              </div>
            )}
          </Card>
        </Space>
      </Col>

      {/* 环境统计和预设 */}
      <Col span={8}>
        <Space direction="vertical" style={{ width: '100%' }}>
          {/* 环境统计 */}
          <Card title="环境统计" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row>
                <Col span={12}>
                  <Text type="secondary">可用环境</Text>
                </Col>
                <Col span={12} style={{ textAlign: 'right' }}>
                  <Text strong>{environments.length}</Text>
                </Col>
              </Row>

              {currentEnvironment && (
                <>
                  <Row>
                    <Col span={12}>
                      <Text type="secondary">状态特征</Text>
                    </Col>
                    <Col span={12} style={{ textAlign: 'right' }}>
                      <Text strong>
                        {currentEnvironment.stateFeatures.length}
                      </Text>
                    </Col>
                  </Row>

                  <Row>
                    <Col span={12}>
                      <Text type="secondary">可用动作</Text>
                    </Col>
                    <Col span={12} style={{ textAlign: 'right' }}>
                      <Text strong>{currentEnvironment.actions.length}</Text>
                    </Col>
                  </Row>
                </>
              )}
            </Space>
          </Card>

          {/* 环境推荐 */}
          <Card title="算法-环境推荐" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="Classic Q-Learning"
                description="推荐使用GridWorld等离散状态空间环境"
                type="info"
                size="small"
                showIcon
              />

              <Alert
                message="Deep Q-Network"
                description="推荐使用CartPole等连续状态空间环境"
                type="success"
                size="small"
                showIcon
              />

              <Alert
                message="Dueling DQN"
                description="适用于动作价值差异明显的环境"
                type="warning"
                size="small"
                showIcon
              />
            </Space>
          </Card>

          {/* 快速配置模板 */}
          <Card title="快速配置模板" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button size="small" block>
                🎯 简单导航任务
              </Button>
              <Button size="small" block>
                🚗 自动驾驶模拟
              </Button>
              <Button size="small" block>
                🎮 游戏AI训练
              </Button>
              <Button size="small" block>
                📈 投资组合优化
              </Button>
            </Space>
          </Card>
        </Space>
      </Col>

      {/* 自定义配置表单 */}
      {isCustomizing && (
        <Col span={24}>
          <Card
            title="自定义环境配置"
            extra={
              <Space>
                <Button icon={<SaveOutlined />} type="primary">
                  保存配置
                </Button>
                <Button onClick={() => setIsCustomizing(false)}>取消</Button>
              </Space>
            }
          >
            <Form
              form={form}
              layout="vertical"
              onFinish={handleSaveEnvironment}
            >
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item
                    name="environmentName"
                    label="环境名称"
                    rules={[{ required: true, message: '请输入环境名称' }]}
                  >
                    <Input placeholder="输入环境名称" />
                  </Form.Item>
                </Col>
                <Col span={16}>
                  <Form.Item
                    name="description"
                    label="环境描述"
                    rules={[{ required: true, message: '请输入环境描述' }]}
                  >
                    <Input placeholder="描述环境的特点和目标" />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="stateSpaceType"
                    label="状态空间类型"
                    initialValue="discrete"
                  >
                    <Select>
                      <Option value="discrete">离散状态空间</Option>
                      <Option value="continuous">连续状态空间</Option>
                      <Option value="hybrid">混合状态空间</Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="actionSpaceType"
                    label="动作空间类型"
                    initialValue="discrete"
                  >
                    <Select>
                      <Option value="discrete">离散动作空间</Option>
                      <Option value="continuous">连续动作空间</Option>
                    </Select>
                  </Form.Item>
                </Col>
              </Row>
            </Form>
          </Card>
        </Col>
      )}
    </Row>
  )
}

export { QLearningEnvironmentPanel }
