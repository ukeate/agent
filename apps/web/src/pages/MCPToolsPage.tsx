import React, { useState, useEffect } from 'react'
import {
  Card,
  Tabs,
  Button,
  Input,
  Select,
  Space,
  Row,
  Col,
  Badge,
  Tag,
  Alert,
  Form,
  message,
  Spin,
  Statistic,
  Progress,
  List,
  Typography,
  Divider,
  Modal,
  Collapse
} from 'antd'
import {
  ApiOutlined,
  FileOutlined,
  DatabaseOutlined,
  DesktopOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  CodeOutlined,
  FolderOpenOutlined
} from '@ant-design/icons'
// import MonacoEditor from '@monaco-editor/react' // 暂时移除
import mcpService, {
  type ToolInfo,
  type HealthCheckResponse,
  type MetricsResponse,
  type ToolCallResponse
} from '../services/mcpService'

const { TabPane } = Tabs
const { TextArea } = Input
const { Text, Title, Paragraph } = Typography
const { Panel } = Collapse

const MCPToolsPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('filesystem')
  
  // 工具列表和状态
  const [availableTools, setAvailableTools] = useState<Record<string, ToolInfo[]>>({})
  const [healthStatus, setHealthStatus] = useState<HealthCheckResponse | null>(null)
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  
  // 执行历史
  const [executionHistory, setExecutionHistory] = useState<Array<{
    id: string
    timestamp: Date
    server_type: string
    tool_name: string
    arguments: any
    result?: any
    error?: string
    duration: number
  }>>([])
  
  // 表单状态
  const [fileSystemForm] = Form.useForm()
  const [databaseForm] = Form.useForm()
  const [systemForm] = Form.useForm()

  // 加载初始数据
  useEffect(() => {
    loadTools()
    checkHealth()
    loadMetrics()
    
    // 定期刷新
    const interval = setInterval(() => {
      checkHealth()
      loadMetrics()
    }, 30000)
    
    return () => clearInterval(interval)
  }, [])

  const loadTools = async () => {
    try {
      const response = await mcpService.listAvailableTools()
      setAvailableTools(response.tools)
    } catch (error) {
      message.error('加载工具列表失败')
    }
  }

  const checkHealth = async () => {
    try {
      const health = await mcpService.healthCheck()
      setHealthStatus(health)
    } catch (error) {
      console.error('健康检查失败:', error)
    }
  }

  const loadMetrics = async () => {
    try {
      const metricsData = await mcpService.getMetrics()
      setMetrics(metricsData)
    } catch (error) {
      console.error('加载指标失败:', error)
    }
  }

  // 执行工具调用
  const executeTool = async (serverType: string, toolName: string, args: any) => {
    setLoading(true)
    const startTime = Date.now()
    
    try {
      const response = await mcpService.callTool({
        server_type: serverType,
        tool_name: toolName,
        arguments: args
      })
      
      const duration = Date.now() - startTime
      
      // 添加到历史记录
      setExecutionHistory(prev => [{
        id: `exec-${Date.now()}`,
        timestamp: new Date(),
        server_type: serverType,
        tool_name: toolName,
        arguments: args,
        result: response.result,
        error: response.error,
        duration
      }, ...prev.slice(0, 49)])
      
      if (response.success) {
        message.success('工具执行成功')
        return response.result
      } else {
        message.error(response.error || '工具执行失败')
      }
    } catch (error) {
      message.error('工具调用失败')
    } finally {
      setLoading(false)
    }
  }

  // 文件系统工具面板
  const renderFileSystemPanel = () => (
    <Card>
      <Form form={fileSystemForm} layout="vertical">
        <Tabs>
          <TabPane tab="读取文件" key="read">
            <Form.Item name="read_path" label="文件路径" rules={[{ required: true }]}>
              <Input placeholder="/path/to/file" />
            </Form.Item>
            <Form.Item name="read_encoding" label="编码" initialValue="utf-8">
              <Select>
                <Select.Option value="utf-8">UTF-8</Select.Option>
                <Select.Option value="ascii">ASCII</Select.Option>
                <Select.Option value="latin-1">Latin-1</Select.Option>
              </Select>
            </Form.Item>
            <Button
              type="primary"
              icon={<FileOutlined />}
              loading={loading}
              onClick={async () => {
                const values = await fileSystemForm.validateFields(['read_path', 'read_encoding'])
                await executeTool('filesystem', 'read_file', {
                  path: values.read_path,
                  encoding: values.read_encoding
                })
              }}
            >
              读取文件
            </Button>
          </TabPane>
          
          <TabPane tab="写入文件" key="write">
            <Form.Item name="write_path" label="文件路径" rules={[{ required: true }]}>
              <Input placeholder="/path/to/file" />
            </Form.Item>
            <Form.Item name="write_content" label="文件内容" rules={[{ required: true }]}>
              <TextArea rows={10} placeholder="输入文件内容..." />
            </Form.Item>
            <Form.Item name="write_encoding" label="编码" initialValue="utf-8">
              <Select>
                <Select.Option value="utf-8">UTF-8</Select.Option>
                <Select.Option value="ascii">ASCII</Select.Option>
                <Select.Option value="latin-1">Latin-1</Select.Option>
              </Select>
            </Form.Item>
            <Button
              type="primary"
              icon={<FileOutlined />}
              loading={loading}
              onClick={async () => {
                const values = await fileSystemForm.validateFields(['write_path', 'write_content', 'write_encoding'])
                await executeTool('filesystem', 'write_file', {
                  path: values.write_path,
                  content: values.write_content,
                  encoding: values.write_encoding
                })
              }}
            >
              写入文件
            </Button>
          </TabPane>
          
          <TabPane tab="列出目录" key="list">
            <Form.Item name="list_path" label="目录路径" rules={[{ required: true }]}>
              <Input placeholder="/path/to/directory" />
            </Form.Item>
            <Form.Item name="include_hidden" label="显示隐藏文件" valuePropName="checked">
              <Select defaultValue={false}>
                <Select.Option value={true}>是</Select.Option>
                <Select.Option value={false}>否</Select.Option>
              </Select>
            </Form.Item>
            <Button
              type="primary"
              icon={<FolderOpenOutlined />}
              loading={loading}
              onClick={async () => {
                const values = await fileSystemForm.validateFields(['list_path', 'include_hidden'])
                await executeTool('filesystem', 'list_directory', {
                  path: values.list_path,
                  include_hidden: values.include_hidden
                })
              }}
            >
              列出目录
            </Button>
          </TabPane>
        </Tabs>
      </Form>
    </Card>
  )

  // 数据库工具面板
  const renderDatabasePanel = () => (
    <Card>
      <Form form={databaseForm} layout="vertical">
        <Form.Item name="query" label="SQL查询" rules={[{ required: true }]}>
          <TextArea 
            rows={8} 
            placeholder="SELECT * FROM users LIMIT 10;"
            defaultValue="SELECT * FROM users LIMIT 10;"
          />
        </Form.Item>
        <Form.Item name="parameters" label="参数 (JSON)">
          <TextArea rows={3} placeholder='{"user_id": 123}' />
        </Form.Item>
        <Space>
          <Button
            type="primary"
            icon={<DatabaseOutlined />}
            loading={loading}
            onClick={async () => {
              const values = await databaseForm.validateFields()
              let parameters = {}
              if (values.parameters) {
                try {
                  parameters = JSON.parse(values.parameters)
                } catch {
                  message.error('参数必须是有效的JSON')
                  return
                }
              }
              await executeTool('database', 'execute_query', {
                query: values.query,
                parameters
              })
            }}
          >
            执行查询
          </Button>
          <Button
            icon={<InfoCircleOutlined />}
            onClick={() => executeTool('database', 'describe_tables', {})}
          >
            查看表结构
          </Button>
        </Space>
      </Form>
    </Card>
  )

  // 系统工具面板
  const renderSystemPanel = () => (
    <Card>
      <Form form={systemForm} layout="vertical">
        <Form.Item name="command" label="系统命令" rules={[{ required: true }]}>
          <Input placeholder="ls -la" />
        </Form.Item>
        <Form.Item name="timeout" label="超时时间（秒）" initialValue={30}>
          <Input type="number" />
        </Form.Item>
        <Space>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            loading={loading}
            onClick={async () => {
              const values = await systemForm.validateFields()
              await executeTool('system', 'run_command', {
                command: values.command,
                timeout: values.timeout
              })
            }}
          >
            运行命令
          </Button>
          <Button
            icon={<DesktopOutlined />}
            onClick={() => executeTool('system', 'get_system_info', {})}
          >
            系统信息
          </Button>
          <Button
            icon={<CodeOutlined />}
            onClick={() => executeTool('system', 'get_env', {})}
          >
            环境变量
          </Button>
        </Space>
      </Form>
    </Card>
  )

  // 执行历史面板
  const renderExecutionHistory = () => (
    <Card title="执行历史" size="small">
      <List
        size="small"
        dataSource={executionHistory}
        renderItem={item => (
          <List.Item>
            <List.Item.Meta
              avatar={
                item.error ? 
                <CloseCircleOutlined style={{ color: '#ff4d4f' }} /> : 
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              }
              title={
                <Space>
                  <Tag color={
                    item.server_type === 'filesystem' ? 'blue' :
                    item.server_type === 'database' ? 'green' :
                    'orange'
                  }>
                    {item.server_type}
                  </Tag>
                  <Text strong>{item.tool_name}</Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </Text>
                </Space>
              }
              description={
                <div>
                  <div className="text-xs text-gray-500">
                    参数: {JSON.stringify(item.arguments, null, 2).substring(0, 100)}
                  </div>
                  {item.error && (
                    <Alert variant="destructive" message={item.error} className="mt-1" />
                  )}
                  {item.result && (
                    <Collapse ghost className="mt-1">
                      <Panel header="查看结果" key="1">
                        <pre className="text-xs overflow-auto max-h-40">
                          {typeof item.result === 'string' ? 
                            item.result : 
                            JSON.stringify(item.result, null, 2)
                          }
                        </pre>
                      </Panel>
                    </Collapse>
                  )}
                  <div className="text-xs text-gray-400 mt-1">
                    耗时: {item.duration}ms
                  </div>
                </div>
              }
            />
          </List.Item>
        )}
      />
    </Card>
  )

  return (
    <div className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">MCP 工具管理</h1>
            <Space>
              <Button 
                icon={<ReloadOutlined />}
                onClick={() => {
                  loadTools()
                  checkHealth()
                  loadMetrics()
                }}
              >
                刷新
              </Button>
            </Space>
          </div>

          {/* 健康状态和指标 */}
          <Row gutter={16} className="mb-4">
            <Col span={6}>
              <Card>
                <Statistic
                  title="系统状态"
                  value={healthStatus?.overall_healthy ? '健康' : '异常'}
                  valueStyle={{ 
                    color: healthStatus?.overall_healthy ? '#3f8600' : '#cf1322' 
                  }}
                  prefix={
                    healthStatus?.overall_healthy ? 
                    <CheckCircleOutlined /> : 
                    <CloseCircleOutlined />
                  }
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总调用次数"
                  value={metrics?.monitoring_stats.total_calls || 0}
                  prefix={<ApiOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="成功率"
                  value={
                    metrics?.monitoring_stats.total_calls ? 
                    ((metrics.monitoring_stats.successful_calls / metrics.monitoring_stats.total_calls) * 100).toFixed(1) : 
                    0
                  }
                  suffix="%"
                  valueStyle={{ color: '#3f8600' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均响应时间"
                  value={metrics?.monitoring_stats.average_response_time || 0}
                  suffix="ms"
                  precision={0}
                  prefix={<PlayCircleOutlined />}
                />
              </Card>
            </Col>
          </Row>

          {/* 服务器状态 */}
          {healthStatus && (
            <Row gutter={16} className="mb-4">
              {Object.entries(healthStatus.servers).map(([serverType, status]) => (
                <Col span={8} key={serverType}>
                  <Card size="small">
                    <div className="flex justify-between items-center">
                      <Space>
                        {serverType === 'filesystem' && <FileOutlined />}
                        {serverType === 'database' && <DatabaseOutlined />}
                        {serverType === 'system' && <DesktopOutlined />}
                        <Text strong>{serverType}</Text>
                      </Space>
                      <Badge 
                        status={status.healthy ? 'success' : 'error'} 
                        text={status.status}
                      />
                    </div>
                    <div className="mt-2">
                      <Text type="secondary">工具数: {status.tools_count}</Text>
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          )}

          {/* 工具操作区 */}
          <Row gutter={16}>
            <Col span={16}>
              <Tabs activeKey={activeTab} onChange={setActiveTab}>
                <TabPane 
                  tab={
                    <span>
                      <FileOutlined />
                      文件系统
                    </span>
                  } 
                  key="filesystem"
                >
                  {renderFileSystemPanel()}
                </TabPane>
                <TabPane 
                  tab={
                    <span>
                      <DatabaseOutlined />
                      数据库
                    </span>
                  }
                  key="database"
                >
                  {renderDatabasePanel()}
                </TabPane>
                <TabPane 
                  tab={
                    <span>
                      <DesktopOutlined />
                      系统
                    </span>
                  }
                  key="system"
                >
                  {renderSystemPanel()}
                </TabPane>
              </Tabs>
            </Col>
            
            <Col span={8}>
              {renderExecutionHistory()}
            </Col>
          </Row>
        </div>
    </div>
  )
}

export default MCPToolsPage