import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Select,
  Table,
  Tag,
  Space,
  Typography,
  Tabs,
  Alert,
  List,
  notification,
  Spin,
  Form,
  Modal
} from 'antd'
import {
  SearchOutlined,
  PlayCircleOutlined,
  SaveOutlined,
  HistoryOutlined,
  BookOutlined,
  CodeOutlined,
  TableOutlined,
  BarChartOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { TextArea } = Input

const GraphQueryEnginePage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('query')
  const [queryText, setQueryText] = useState('')
  const [queryResults, setQueryResults] = useState<any[]>([])

  const handleQuery = async () => {
    if (!queryText.trim()) {
      notification.warning({ message: '请输入查询语句' })
      return
    }

    setLoading(true)
    setTimeout(() => {
      setQueryResults([
        { id: 1, subject: '张三', predicate: 'works_for', object: '苹果公司' },
        { id: 2, subject: '苹果公司', predicate: 'located_in', object: '加州' }
      ])
      setLoading(false)
      notification.success({ message: '查询完成，找到 2 个结果' })
    }, 2000)
  }

  const renderQueryInterface = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="图查询引擎" extra={
          <Space>
            <Select defaultValue="cypher" style={{ width: 120 }}>
              <Select.Option value="cypher">Cypher</Select.Option>
              <Select.Option value="sparql">SPARQL</Select.Option>
              <Select.Option value="gremlin">Gremlin</Select.Option>
            </Select>
            <Button icon={<BookOutlined />}>语法帮助</Button>
          </Space>
        }>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <TextArea
              rows={8}
              placeholder="请输入图查询语句...&#10;例如: MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.name"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              style={{ fontFamily: 'monospace' }}
            />
            <Row justify="space-between">
              <Col>
                <Space>
                  <Button icon={<SaveOutlined />}>保存查询</Button>
                  <Button icon={<HistoryOutlined />}>查询历史</Button>
                </Space>
              </Col>
              <Col>
                <Button 
                  type="primary" 
                  icon={<PlayCircleOutlined />}
                  loading={loading}
                  onClick={handleQuery}
                >
                  执行查询
                </Button>
              </Col>
            </Row>
          </Space>
        </Card>
      </Col>
      <Col span={24}>
        <Card title="查询结果" extra={
          <Space>
            <Button icon={<TableOutlined />}>表格视图</Button>
            <Button icon={<BarChartOutlined />}>图表视图</Button>
          </Space>
        }>
          <Table
            dataSource={queryResults}
            columns={[
              { title: '主体', dataIndex: 'subject', key: 'subject' },
              { title: '关系', dataIndex: 'predicate', key: 'predicate' },
              { title: '客体', dataIndex: 'object', key: 'object' }
            ]}
            loading={loading}
          />
        </Card>
      </Col>
    </Row>
  )

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <SearchOutlined style={{ marginRight: 8 }} />
          图查询引擎
        </Title>
        <Paragraph type="secondary">
          使用Cypher、SPARQL等查询语言对知识图谱进行复杂查询和分析
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="查询界面" key="query">
          {renderQueryInterface()}
        </TabPane>
        <TabPane tab="查询模板" key="templates">
          <Card title="查询模板库">
            <List
              dataSource={[
                { name: '查找人员工作关系', query: 'MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p, c' },
                { name: '查找公司位置信息', query: 'MATCH (c:Company)-[:LOCATED_IN]->(l:Location) RETURN c, l' }
              ]}
              renderItem={(item) => (
                <List.Item
                  actions={[<Button type="link">使用模板</Button>]}
                >
                  <List.Item.Meta
                    title={item.name}
                    description={<Text code>{item.query}</Text>}
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default GraphQueryEnginePage