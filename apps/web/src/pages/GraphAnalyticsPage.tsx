import React from 'react'
import { Card, Row, Col, Statistic, Typography } from 'antd'
import { BarChartOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const GraphAnalyticsPage: React.FC = () => {
  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <BarChartOutlined style={{ marginRight: 8 }} />
          图谱分析统计
        </Title>
        <Paragraph type="secondary">
          知识图谱的统计分析和可视化展示
        </Paragraph>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={6}>
          <Card>
            <Statistic title="节点总数" value={12450} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="关系总数" value={5680} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="实体类型" value={25} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="关系类型" value={15} />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default GraphAnalyticsPage