import React from 'react'
import { Card, Row, Col, Statistic, Progress, Typography } from 'antd'
import { FundOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgeConfidenceAnalysisPage: React.FC = () => {
  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <FundOutlined style={{ marginRight: 8 }} />
          置信度分析
        </Title>
        <Paragraph type="secondary">
          分析知识图谱中实体和关系的置信度分布
        </Paragraph>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card title="实体置信度">
            <Progress type="circle" percent={94} />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="关系置信度">
            <Progress type="circle" percent={89} />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="低置信度项">
            <Statistic value={156} suffix="个" />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default KnowledgeConfidenceAnalysisPage