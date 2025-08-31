import React from 'react'
import { Card, Row, Col, Statistic, Typography, Button, Space } from 'antd'
import { DatabaseOutlined, ReloadOutlined, DeleteOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgeCacheManagementPage: React.FC = () => {
  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: 8 }} />
          缓存管理
        </Title>
        <Paragraph type="secondary">
          知识图谱系统的缓存管理和优化
        </Paragraph>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={6}>
          <Card>
            <Statistic title="缓存命中率" value={85.2} suffix="%" />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="缓存大小" value={1.2} suffix="GB" />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="缓存项数" value={15680} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Space>
              <Button icon={<ReloadOutlined />}>刷新</Button>
              <Button icon={<DeleteOutlined />} danger>清空</Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default KnowledgeCacheManagementPage