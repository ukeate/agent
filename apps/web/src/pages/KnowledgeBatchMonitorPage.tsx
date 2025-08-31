import React from 'react'
import { Card, Row, Col, Statistic, Progress, Typography } from 'antd'
import { MonitorOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgeBatchMonitorPage: React.FC = () => {
  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <MonitorOutlined style={{ marginRight: 8 }} />
          批处理监控
        </Title>
        <Paragraph type="secondary">
          实时监控批处理任务的执行状态和性能指标
        </Paragraph>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card>
            <Statistic title="活跃任务" value={3} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="队列任务" value={5} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="系统负载" value={72} suffix="%" />
          </Card>
        </Col>
      </Row>

      <Card title="任务执行进度" style={{ marginTop: 16 }}>
        <Progress percent={68} status="active" />
      </Card>
    </div>
  )
}

export default KnowledgeBatchMonitorPage