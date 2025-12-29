import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Button, Space, Typography, message, Spin, Divider } from 'antd'
import { PlayCircleOutlined, ReloadOutlined, PauseCircleOutlined } from '@ant-design/icons'
import { workflowService } from '../services/workflowService'
import WorkflowVisualization from '../components/workflow/WorkflowVisualization'

import { logger } from '../utils/logger'
const { Title, Text } = Typography

interface Workflow {
  id: string
  name: string
  status: string
}

const WorkflowPage: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false)
  const [currentWorkflow, setCurrentWorkflow] = useState<Workflow | null>(null)
  const [loading, setLoading] = useState(false)

  // 创建并启动工作流
  const handleStartWorkflow = async () => {
    if (isRunning || loading) return

    setLoading(true)
    try {
      // 创建简单工作流
      const workflowData = {
        name: '条件分支工作流',
        description: 'LangGraph 条件分支工作流演示：数据处理→条件判断→分支路径',
        workflow_type: 'conditional'
      }

      const workflow = await workflowService.createWorkflow(workflowData)
      message.success('工作流创建成功')
      
      // 启动工作流
      await workflowService.startWorkflow(workflow.id, {
        input_data: { message: '启动工作流' }
      })

      setCurrentWorkflow(workflow)
      setIsRunning(true)
      message.success('工作流启动成功')

    } catch (error) {
      logger.error('启动工作流失败:', error)
      message.error(`启动工作流失败: ${(error as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  // 停止工作流
  const handleStopWorkflow = async () => {
    if (!currentWorkflow || !isRunning) return

    setLoading(true)
    try {
      await workflowService.controlWorkflow(currentWorkflow.id, { action: 'cancel' })
      setIsRunning(false)
      message.success('工作流已停止')
    } catch (error) {
      logger.error('停止工作流失败:', error)
      message.error(`停止工作流失败: ${(error as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  // 重置状态
  const handleReset = () => {
    setCurrentWorkflow(null)
    setIsRunning(false)
    message.info('工作流状态已重置')
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>LangGraph 工作流可视化</Title>
      <Text type="secondary">学习 LangGraph 多代理工作流</Text>
      
      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card title="工作流控制">
            <Space>
              {!isRunning ? (
                <Button 
                  type="primary" 
                  icon={<PlayCircleOutlined />}
                  loading={loading}
                  onClick={handleStartWorkflow}
                >
                  启动工作流
                </Button>
              ) : (
                <Button 
                  type="default"
                  icon={<PauseCircleOutlined />}
                  loading={loading}
                  onClick={handleStopWorkflow}
                >
                  停止工作流
                </Button>
              )}
              <Button 
                icon={<ReloadOutlined />}
                onClick={handleReset}
                disabled={loading}
              >
                重置状态
              </Button>
            </Space>
            {currentWorkflow && (
              <div style={{ marginTop: '16px' }}>
                <Text type="secondary">
                  当前工作流: {currentWorkflow.name} (ID: {currentWorkflow.id})
                </Text>
                <br />
                <Text type={isRunning ? 'success' : 'secondary'}>
                  状态: {isRunning ? '运行中' : '已停止'}
                </Text>
              </div>
            )}
            {loading && (
              <div style={{ marginTop: '16px' }}>
                <Spin size="small" />
                <Text style={{ marginLeft: '8px' }}>正在处理...</Text>
              </div>
            )}
          </Card>
        </Col>
        
        {/* 工作流可视化部分 */}
        <Col span={24}>
          <Divider orientation="left">工作流图形化视图</Divider>
          {currentWorkflow ? (
            <WorkflowVisualization 
              workflowId={currentWorkflow.id}
              onNodeClick={(nodeId, nodeData) => {
                logger.log('节点点击:', nodeId, nodeData)
                message.info(`点击节点: ${nodeData?.name || nodeId}`)
              }}
            />
          ) : (
            <Card>
              <div className="flex items-center justify-center" style={{ padding: '20px' }}>
                <div className="text-center">
                  <Text type="secondary" className="text-lg">
                    🚀 启动工作流以查看实时可视化
                  </Text>
                  <br />
                  <Text type="secondary">
                    点击上方“启动工作流”按钮开始
                  </Text>
                </div>
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default WorkflowPage
