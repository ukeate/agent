import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Button, Space, Typography, message, Spin, Divider } from 'antd'
import { PlayCircleOutlined, ReloadOutlined, PauseCircleOutlined } from '@ant-design/icons'
import { apiClient } from '../services/apiClient'
import WorkflowVisualization from '../components/workflow/WorkflowVisualization'

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

      const workflow = await apiClient.createWorkflow(workflowData)
      if (!workflow || !workflow.id || !workflow.name) {
        throw new Error('工作流数据无效')
      }
      message.success('工作流创建成功')
      
      // 启动工作流
      const result = await apiClient.startWorkflow(workflow.id, {
        message: '启动LangGraph工作流演示'
      })

      setCurrentWorkflow(workflow)
      setIsRunning(true)
      message.success('工作流启动成功')

    } catch (error) {
      console.error('启动工作流失败:', error)
      message.error(`启动工作流失败: ${(error as Error).message}`)
      
      // 即使API失败，也创建一个演示工作流对象以便用户体验
      const demoWorkflow = {
        id: `demo-workflow-${Date.now()}`,
        name: '演示工作流 (离线模式)',
        status: 'running'
      }
      setCurrentWorkflow(demoWorkflow)
      setIsRunning(true)
      message.info('已切换到演示模式，您可以体验工作流可视化功能')
    } finally {
      setLoading(false)
    }
  }

  // 停止工作流
  const handleStopWorkflow = async () => {
    if (!currentWorkflow || !isRunning) return

    setLoading(true)
    try {
      await apiClient.controlWorkflow(currentWorkflow.id, 'cancel')
      setIsRunning(false)
      message.success('工作流已停止')
    } catch (error) {
      console.error('停止工作流失败:', error)
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
              demoMode={(currentWorkflow.name || '').includes('离线模式') || (currentWorkflow.id || '').includes('demo-workflow')}
              onNodeClick={(nodeId, nodeData) => {
                console.log('节点点击:', nodeId, nodeData)
                message.info(`点击节点: ${nodeData?.name || nodeId}`)
              }}
            />
          ) : (
            <div>
              <Card style={{ marginBottom: '16px' }}>
                <div className="flex items-center justify-center" style={{ padding: '20px' }}>
                  <div className="text-center">
                    <Text type="secondary" className="text-lg">
                      🚀 启动工作流以查看实时可视化
                    </Text>
                    <br />
                    <Text type="secondary">
                      点击上方"启动工作流"按钮开始，或查看下方演示
                    </Text>
                  </div>
                </div>
              </Card>
              
              {/* 演示模式的工作流可视化 */}
              <Card title="🎯 工作流可视化演示 (示例)" 
                    extra={<Text type="secondary">演示模式 - 展示界面功能</Text>}>
                <WorkflowVisualization 
                  workflowId="demo-workflow-preview"
                  demoMode={true}
                  onNodeClick={(nodeId, nodeData) => {
                    console.log('演示节点点击:', nodeId, nodeData)
                    message.info(`演示模式 - 点击节点: ${nodeData?.name || nodeId}`)
                  }}
                />
              </Card>
            </div>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default WorkflowPage
