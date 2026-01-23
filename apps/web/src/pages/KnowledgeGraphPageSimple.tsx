import React from 'react'
import { Card, Typography, Space, Button, Row, Col, Statistic } from 'antd'
import {
  NodeIndexOutlined,
  DatabaseOutlined,
  SearchOutlined,
  BarChartOutlined,
  ExperimentOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography

const KnowledgeGraphPageSimple: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title
          level={2}
          style={{ margin: 0, display: 'flex', alignItems: 'center' }}
        >
          <NodeIndexOutlined
            style={{ marginRight: '12px', color: '#1890ff' }}
          />
          知识图谱系统
        </Title>
        <Text type="secondary">智能知识图谱构建、查询与可视化平台</Text>
      </div>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="实体总数"
              value={15420}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="关系总数"
              value={45280}
              prefix={<NodeIndexOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="查询次数"
              value={3420}
              prefix={<SearchOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="图谱密度"
              value={0.85}
              precision={2}
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 功能模块卡片 */}
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card
            title="图谱可视化"
            extra={
              <Button type="primary" size="small">
                进入
              </Button>
            }
            style={{ height: '200px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>🎨 交互式图谱可视化界面</Text>
              <Text>🔍 节点和关系的动态探索</Text>
              <Text>⚡ 高性能渲染引擎</Text>
              <Text type="secondary">支持10,000+节点实时渲染</Text>
            </Space>
          </Card>
        </Col>

        <Col span={12}>
          <Card
            title="自然语言查询"
            extra={
              <Button type="primary" size="small">
                进入
              </Button>
            }
            style={{ height: '200px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>💬 智能语言理解</Text>
              <Text>🎯 精准结果匹配</Text>
              <Text>📊 结果可视化展示</Text>
              <Text type="secondary">支持复杂查询语句解析</Text>
            </Space>
          </Card>
        </Col>

        <Col span={12}>
          <Card
            title="图谱推理引擎"
            extra={
              <Button type="primary" size="small">
                进入
              </Button>
            }
            style={{ height: '200px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>🧠 智能推理算法</Text>
              <Text>🔗 关系路径发现</Text>
              <Text>⚖️ 置信度计算</Text>
              <Text type="secondary">基于规则和嵌入混合推理</Text>
            </Space>
          </Card>
        </Col>

        <Col span={12}>
          <Card
            title="数据管理"
            extra={
              <Button type="primary" size="small">
                进入
              </Button>
            }
            style={{ height: '200px' }}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>📁 实体关系管理</Text>
              <Text>🔄 数据同步更新</Text>
              <Text>📤 数据导入导出</Text>
              <Text type="secondary">支持多种数据源接入</Text>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 快速操作 */}
      <Card title="快速操作" style={{ marginTop: '24px' }}>
        <Space wrap>
          <Button type="primary" icon={<SearchOutlined />}>
            新建查询
          </Button>
          <Button icon={<DatabaseOutlined />}>导入数据</Button>
          <Button icon={<ExperimentOutlined />}>创建实验</Button>
          <Button icon={<BarChartOutlined />}>查看报告</Button>
        </Space>
      </Card>
    </div>
  )
}

export default KnowledgeGraphPageSimple
