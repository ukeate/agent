import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Select,
  Table,
  Tag,
  Space,
  Typography,
  Tabs,
  Progress,
  Statistic,
  Alert,
  List,
  Badge,
  notification,
  Spin,
  Switch,
  Slider
} from 'antd'
import {
  GlobalOutlined,
  TranslationOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  CheckCircleOutlined,
  ExclamationTriangleOutlined,
  EyeOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  DatabaseOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface LanguageSupport {
  code: string
  name: string
  status: 'active' | 'inactive' | 'developing'
  entityAccuracy: number
  relationAccuracy: number
  coverage: number
  modelVersion: string
  lastUpdated: string
}

const MultilingualProcessingPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('processing')
  const [languages, setLanguages] = useState<LanguageSupport[]>([])
  const [processText, setProcessText] = useState('')
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(['zh', 'en'])

  useEffect(() => {
    loadMockData()
  }, [])

  const loadMockData = () => {
    setLoading(true)

    const mockLanguages: LanguageSupport[] = [
      {
        code: 'zh',
        name: '中文 (Chinese)',
        status: 'active',
        entityAccuracy: 94.5,
        relationAccuracy: 92.1,
        coverage: 88.3,
        modelVersion: 'v2.1.0',
        lastUpdated: '2024-01-20'
      },
      {
        code: 'en',
        name: '英文 (English)',
        status: 'active',
        entityAccuracy: 96.2,
        relationAccuracy: 94.8,
        coverage: 92.1,
        modelVersion: 'v2.2.0',
        lastUpdated: '2024-01-20'
      },
      {
        code: 'ja',
        name: '日文 (Japanese)',
        status: 'active',
        entityAccuracy: 89.7,
        relationAccuracy: 87.3,
        coverage: 81.5,
        modelVersion: 'v1.8.0',
        lastUpdated: '2024-01-18'
      },
      {
        code: 'ko',
        name: '韩文 (Korean)',
        status: 'developing',
        entityAccuracy: 85.2,
        relationAccuracy: 82.1,
        coverage: 75.8,
        modelVersion: 'v1.5.0-beta',
        lastUpdated: '2024-01-15'
      }
    ]

    setTimeout(() => {
      setLanguages(mockLanguages)
      setLoading(false)
    }, 1000)
  }

  const handleProcessing = async () => {
    if (!processText.trim()) {
      notification.warning({ message: '请输入要处理的文本' })
      return
    }

    setLoading(true)
    setTimeout(() => {
      notification.success({ 
        message: `成功处理 ${selectedLanguages.length} 种语言的文本，识别出 12 个实体和 5 个关系` 
      })
      setLoading(false)
    }, 3000)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'inactive': return 'default'
      case 'developing': return 'processing'
      default: return 'default'
    }
  }

  const languageColumns = [
    {
      title: '语言',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: LanguageSupport) => (
        <Space>
          <GlobalOutlined />
          <div>
            <Text strong>{name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.code.toUpperCase()}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status === 'active' ? '活跃' :
           status === 'inactive' ? '未激活' : '开发中'}
        </Tag>
      )
    },
    {
      title: '实体准确率',
      dataIndex: 'entityAccuracy',
      key: 'entityAccuracy',
      render: (accuracy: number) => (
        <Progress
          percent={accuracy}
          size="small"
          format={percent => `${percent}%`}
        />
      )
    },
    {
      title: '关系准确率',
      dataIndex: 'relationAccuracy',
      key: 'relationAccuracy',
      render: (accuracy: number) => (
        <Progress
          percent={accuracy}
          size="small"
          format={percent => `${percent}%`}
        />
      )
    },
    {
      title: '覆盖率',
      dataIndex: 'coverage',
      key: 'coverage',
      render: (coverage: number) => `${coverage}%`
    },
    {
      title: '模型版本',
      dataIndex: 'modelVersion',
      key: 'modelVersion',
      render: (version: string) => <Tag color="blue">{version}</Tag>
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: LanguageSupport) => (
        <Space>
          <Switch checked={record.status === 'active'} size="small" />
          <Button size="small" type="link" icon={<SettingOutlined />}>
            配置
          </Button>
          <Button size="small" type="link" icon={<EyeOutlined />}>
            详情
          </Button>
        </Space>
      )
    }
  ]

  const renderProcessing = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="多语言知识抽取" extra={
          <Space>
            <Select 
              mode="multiple"
              value={selectedLanguages} 
              onChange={setSelectedLanguages}
              style={{ width: 300 }}
              placeholder="选择要处理的语言"
            >
              {languages.filter(lang => lang.status === 'active').map(lang => (
                <Option key={lang.code} value={lang.code}>
                  {lang.name}
                </Option>
              ))}
            </Select>
            <Button icon={<SettingOutlined />}>
              配置
            </Button>
          </Space>
        }>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Input.TextArea
              rows={8}
              placeholder="请输入多语言文本，系统将自动识别语言并进行知识抽取..."
              value={processText}
              onChange={(e) => setProcessText(e.target.value)}
            />
            <Row justify="space-between">
              <Col>
                <Space>
                  <Button>
                    语言检测
                  </Button>
                  <Button>
                    上传文件
                  </Button>
                </Space>
              </Col>
              <Col>
                <Button 
                  type="primary" 
                  icon={<TranslationOutlined />}
                  loading={loading}
                  onClick={handleProcessing}
                >
                  开始处理
                </Button>
              </Col>
            </Row>
          </Space>
        </Card>
      </Col>
      
      <Col span={24}>
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <Card>
              <Statistic
                title="支持语言数"
                value={languages.filter(l => l.status === 'active').length}
                prefix={<GlobalOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="平均实体准确率"
                value={92.6}
                suffix="%"
                prefix={<NodeIndexOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="平均关系准确率"
                value={89.1}
                suffix="%"
                prefix={<ShareAltOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        </Row>
      </Col>
    </Row>
  )

  const renderLanguageSupport = () => (
    <div>
      <Card 
        title="语言支持管理"
        extra={
          <Space>
            <Button type="primary">
              添加语言
            </Button>
            <Button icon={<ThunderboltOutlined />}>
              性能测试
            </Button>
          </Space>
        }
      >
        <Table
          columns={languageColumns}
          dataSource={languages}
          rowKey="code"
          loading={loading}
          pagination={false}
        />
      </Card>
    </div>
  )

  const renderStatistics = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="多语言处理统计">
          <Row gutter={[16, 16]}>
            {languages.map(lang => (
              <Col span={6} key={lang.code}>
                <Card size="small" title={lang.name}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>实体准确率</Text>
                      <Progress percent={lang.entityAccuracy} size="small" />
                    </div>
                    <div>
                      <Text>关系准确率</Text>
                      <Progress percent={lang.relationAccuracy} size="small" />
                    </div>
                    <div>
                      <Text>覆盖率</Text>
                      <Progress percent={lang.coverage} size="small" />
                    </div>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>
      </Col>
    </Row>
  )

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <GlobalOutlined style={{ marginRight: 8 }} />
          多语言处理
        </Title>
        <Paragraph type="secondary">
          支持多种语言的知识抽取，包括实体识别、关系抽取和跨语言实体链接
        </Paragraph>
      </div>

      <Alert
        message="多语言处理服务运行正常"
        description="当前支持 3 种主要语言，1 种语言正在开发中"
        type="success"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="多语言处理" key="processing">
          {renderProcessing()}
        </TabPane>
        <TabPane tab="语言支持" key="languages">
          {renderLanguageSupport()}
        </TabPane>
        <TabPane tab="性能统计" key="statistics">
          {renderStatistics()}
        </TabPane>
      </Tabs>
    </div>
  )
}

export default MultilingualProcessingPage