import React, { useState } from 'react'
import { Card, Row, Col, Button, Space, Typography } from 'antd'
import { PlayCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import { MainLayout } from '../components/layout/MainLayout'

const { Title, Text } = Typography

const WorkflowPage: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false)

  return (
    <MainLayout>
      <div style={{ padding: '24px' }}>
        <Title level={2}>LangGraph 工作流可视化</Title>
        <Text type="secondary">学习 LangGraph 多代理工作流</Text>
        
        <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
          <Col span={24}>
            <Card title="工作流控制">
              <Space>
                <Button 
                  type="primary" 
                  icon={<PlayCircleOutlined />}
                  loading={isRunning}
                  onClick={() => setIsRunning(!isRunning)}
                >
                  启动工作流
                </Button>
                <Button icon={<ReloadOutlined />}>重置状态</Button>
              </Space>
            </Card>
          </Col>
        </Row>
      </div>
    </MainLayout>
  )
}

export default WorkflowPage