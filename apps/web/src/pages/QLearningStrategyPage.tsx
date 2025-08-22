import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Space, 
  Table, 
  Input,
  Select,
  Tag,
  Statistic,
  Alert,
  Progress,
  Typography,
  Divider,
  Form,
  InputNumber,
  Tooltip,
  Modal,
  Tabs
} from 'antd'
import { 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  BarChart,
  Bar
} from 'recharts'
import {
  BulbOutlined,
  PlayCircleOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  RadarChartOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
  SettingOutlined,
  ReloadOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TabPane } = Tabs

// 模拟策略推理数据
const generateInferenceData = () => {
  const states = []
  const actions = ['向上', '向下', '向左', '向右']
  
  for (let i = 0; i < 50; i++) {
    const state = [
      Math.random() * 10 - 5,  // x坐标
      Math.random() * 10 - 5,  // y坐标
      Math.random() * 2 - 1,   // 速度x
      Math.random() * 2 - 1    // 速度y
    ]
    
    const qValues = actions.map(() => Math.random() * 100 - 50)
    const bestAction = qValues.indexOf(Math.max(...qValues))
    
    states.push({
      id: i + 1,
      state: state,
      stateStr: `[${state.map(x => x.toFixed(2)).join(', ')}]`,
      qValues: qValues,
      recommendedAction: bestAction,
      actionName: actions[bestAction],
      confidence: Math.random() * 100,
      inferenceTime: Math.random() * 10 + 1
    })
  }
  
  return states
}

// 生成策略比较数据
const generateStrategyComparison = () => [
  {
    strategy: 'Greedy',
    confidence: 95,
    speed: 90,
    accuracy: 88,
    stability: 85,
    exploration: 20
  },
  {
    strategy: 'Epsilon-Greedy',
    confidence: 85,
    speed: 85,
    accuracy: 82,
    stability: 88,
    exploration: 75
  },
  {
    strategy: 'Boltzmann',
    confidence: 78,
    speed: 75,
    accuracy: 80,
    stability: 80,
    exploration: 85
  },
  {
    strategy: 'UCB',
    confidence: 82,
    speed: 70,
    accuracy: 85,
    stability: 90,
    exploration: 88
  }
]

const QLearningStrategyPage: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState('dqn-agent-1')
  const [currentState, setCurrentState] = useState([0, 0, 0, 0])
  const [inferenceData, setInferenceData] = useState(() => generateInferenceData())
  const [isLoading, setIsLoading] = useState(false)
  const [strategyComparison] = useState(() => generateStrategyComparison())
  const [batchInferenceVisible, setBatchInferenceVisible] = useState(false)

  const handleSingleInference = async () => {
    setIsLoading(true)
    // 模拟API调用延迟
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    const qValues = [0, 1, 2, 3].map(() => Math.random() * 100 - 50)
    const bestAction = qValues.indexOf(Math.max(...qValues))
    
    console.log('推理结果:', {
      state: currentState,
      qValues,
      recommendedAction: bestAction,
      confidence: Math.random() * 100
    })
    
    setIsLoading(false)
  }

  const handleBatchInference = () => {
    setInferenceData(generateInferenceData())
  }

  // 状态输入表单
  const StateInputForm = () => (
    <Card title="状态输入" size="small">
      <Form layout="vertical">
        <Row gutter={16}>
          <Col span={6}>
            <Form.Item label="X坐标">
              <InputNumber
                value={currentState[0]}
                onChange={(val) => setCurrentState([val || 0, currentState[1], currentState[2], currentState[3]])}
                style={{ width: '100%' }}
                step={0.1}
              />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="Y坐标">
              <InputNumber
                value={currentState[1]}
                onChange={(val) => setCurrentState([currentState[0], val || 0, currentState[2], currentState[3]])}
                style={{ width: '100%' }}
                step={0.1}
              />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="X速度">
              <InputNumber
                value={currentState[2]}
                onChange={(val) => setCurrentState([currentState[0], currentState[1], val || 0, currentState[3]])}
                style={{ width: '100%' }}
                step={0.1}
              />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item label="Y速度">
              <InputNumber
                value={currentState[3]}
                onChange={(val) => setCurrentState([currentState[0], currentState[1], currentState[2], val || 0])}
                style={{ width: '100%' }}
                step={0.1}
              />
            </Form.Item>
          </Col>
        </Row>
        
        <Space>
          <Button 
            type="primary" 
            icon={<PlayCircleOutlined />}
            loading={isLoading}
            onClick={handleSingleInference}
          >
            单次推理
          </Button>
          <Button 
            icon={<ExperimentOutlined />}
            onClick={() => setBatchInferenceVisible(true)}
          >
            批量推理
          </Button>
          <Button 
            icon={<ReloadOutlined />}
            onClick={() => setCurrentState([0, 0, 0, 0])}
          >
            重置状态
          </Button>
        </Space>
      </Form>
    </Card>
  )

  // 推理结果展示
  const InferenceResultsTable = () => {
    const columns = [
      {
        title: 'ID',
        dataIndex: 'id',
        key: 'id',
        width: 60,
      },
      {
        title: '状态',
        dataIndex: 'stateStr',
        key: 'state',
        ellipsis: true,
      },
      {
        title: '推荐动作',
        dataIndex: 'actionName',
        key: 'action',
        render: (action: string, record: any) => (
          <Tag color="blue">{action}</Tag>
        )
      },
      {
        title: '置信度',
        dataIndex: 'confidence',
        key: 'confidence',
        render: (confidence: number) => (
          <Progress 
            percent={confidence} 
            size="small" 
            format={percent => `${percent?.toFixed(1)}%`}
          />
        )
      },
      {
        title: '推理时间',
        dataIndex: 'inferenceTime',
        key: 'time',
        render: (time: number) => `${time.toFixed(2)}ms`
      },
      {
        title: 'Q值分布',
        key: 'qValues',
        render: (record: any) => (
          <Space size="small">
            {record.qValues.map((q: number, idx: number) => (
              <Tag 
                key={idx} 
                color={idx === record.recommendedAction ? 'green' : 'default'}
              >
                {q.toFixed(1)}
              </Tag>
            ))}
          </Space>
        )
      }
    ]

    return (
      <Card title="推理结果历史" size="small">
        <Table
          columns={columns}
          dataSource={inferenceData}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      </Card>
    )
  }

  // 策略性能比较雷达图
  const StrategyComparisonRadar = () => (
    <Card title="策略性能比较" size="small">
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart data={strategyComparison}>
          <PolarGrid />
          <PolarAngleAxis dataKey="strategy" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />
          <Radar
            name="置信度"
            dataKey="confidence"
            stroke="#8884d8"
            fill="#8884d8"
            fillOpacity={0.1}
          />
          <Radar
            name="准确率"
            dataKey="accuracy"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.1}
          />
          <Radar
            name="稳定性"
            dataKey="stability"
            stroke="#ffc658"
            fill="#ffc658"
            fillOpacity={0.1}
          />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </Card>
  )

  // 状态-动作散点图
  const StateActionScatter = () => {
    const scatterData = inferenceData.map(item => ({
      x: item.state[0],
      y: item.state[1],
      action: item.recommendedAction,
      confidence: item.confidence
    }))

    return (
      <Card title="状态-动作分布" size="small">
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart data={scatterData}>
            <CartesianGrid />
            <XAxis dataKey="x" name="X坐标" />
            <YAxis dataKey="y" name="Y坐标" />
            <Tooltip 
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(value, name, props) => {
                if (name === 'confidence') return [`${value.toFixed(1)}%`, '置信度']
                return [value, name]
              }}
            />
            <Legend />
            <Scatter 
              name="动作决策" 
              dataKey="confidence" 
              fill="#1890ff"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  // 智能体选择器
  const AgentSelector = () => (
    <Card title="智能体选择" size="small">
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <Text strong>当前智能体:</Text>
          <Select 
            value={selectedAgent} 
            onChange={setSelectedAgent}
            style={{ width: '100%', marginTop: 8 }}
          >
            <Option value="dqn-agent-1">DQN智能体 #1</Option>
            <Option value="double-dqn-agent-2">Double DQN智能体 #2</Option>
            <Option value="dueling-dqn-agent-3">Dueling DQN智能体 #3</Option>
            <Option value="tabular-q-agent-4">表格Q学习智能体 #4</Option>
          </Select>
        </div>
        
        <Row gutter={8}>
          <Col span={8}>
            <Statistic 
              title="训练Episodes" 
              value={1250} 
              prefix={<ExperimentOutlined />}
            />
          </Col>
          <Col span={8}>
            <Statistic 
              title="平均奖励" 
              value={87.5} 
              precision={1}
              prefix={<TrophyOutlined />}
            />
          </Col>
          <Col span={8}>
            <Statistic 
              title="推理次数" 
              value={inferenceData.length} 
              prefix={<ThunderboltOutlined />}
            />
          </Col>
        </Row>
      </Space>
    </Card>
  )

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <BulbOutlined /> Q-Learning策略推理
      </Title>
      <Paragraph type="secondary">
        对训练好的Q-Learning智能体进行策略推理分析，支持单次推理、批量推理和策略比较分析
      </Paragraph>
      
      <Divider />

      <Tabs defaultActiveKey="1">
        <TabPane tab="策略推理" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Row gutter={16}>
              <Col span={8}>
                <AgentSelector />
              </Col>
              <Col span={16}>
                <StateInputForm />
              </Col>
            </Row>

            <InferenceResultsTable />
          </Space>
        </TabPane>

        <TabPane tab="策略分析" key="2">
          <Row gutter={16}>
            <Col span={12}>
              <StrategyComparisonRadar />
            </Col>
            <Col span={12}>
              <StateActionScatter />
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="性能洞察" key="3">
          <Row gutter={16}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均推理时间"
                  value={5.2}
                  precision={1}
                  suffix="ms"
                  prefix={<ClockCircleOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="置信度均值"
                  value={78.5}
                  precision={1}
                  suffix="%"
                  prefix={<BarChartOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="策略一致性"
                  value={92.3}
                  precision={1}
                  suffix="%"
                  prefix={<RadarChartOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="缓存命中率"
                  value={85.7}
                  precision={1}
                  suffix="%"
                  prefix={<ThunderboltOutlined />}
                />
              </Card>
            </Col>
          </Row>
          
          <div style={{ marginTop: 16 }}>
            <Card title="推理性能分析" size="small">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={inferenceData.slice(-10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="id" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="inferenceTime" fill="#1890ff" name="推理时间(ms)" />
                  <Bar dataKey="confidence" fill="#52c41a" name="置信度(%)" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </TabPane>
      </Tabs>

      {/* 批量推理模态框 */}
      <Modal
        title="批量推理"
        visible={batchInferenceVisible}
        onOk={() => {
          handleBatchInference()
          setBatchInferenceVisible(false)
        }}
        onCancel={() => setBatchInferenceVisible(false)}
        width={600}
      >
        <Alert
          message="批量推理"
          description="将对50个随机生成的状态进行批量推理，生成新的推理结果数据"
          variant="default"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Text>确认执行批量推理操作吗？这将替换当前的推理结果数据。</Text>
      </Modal>
    </div>
  )
}

export default QLearningStrategyPage