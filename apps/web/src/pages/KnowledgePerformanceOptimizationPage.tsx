import React from 'react'
import { Card, Row, Col, Statistic, Typography } from 'antd'
import { ThunderboltOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgePerformanceOptimizationPage: React.FC = () => {
  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <ThunderboltOutlined style={{ marginRight: 8 }} />
          性能优化中心
        </Title>
        <Paragraph type="secondary">
          知识图谱系统的性能监控和优化
        </Paragraph>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={6}>
          <Card>
            <Statistic title="查询响应时间" value={125} suffix="ms" />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="吞吐量" value={1250} suffix="/min" />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="CPU使用率" value={68} suffix="%" />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="内存使用率" value={72} suffix="%" />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default KnowledgePerformanceOptimizationPage