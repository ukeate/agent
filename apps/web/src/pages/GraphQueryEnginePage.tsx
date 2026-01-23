import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
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
  Form,
  Modal,
} from 'antd'
import {
  SearchOutlined,
  PlayCircleOutlined,
  SaveOutlined,
  HistoryOutlined,
  BookOutlined,
  TableOutlined,
  BarChartOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  knowledgeGraphService,
  type QueryTemplate,
} from '../services/knowledgeGraphService'
import { sparqlService } from '../services/sparqlService'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { TextArea } = Input

const GraphQueryEnginePage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('query')
  const [queryText, setQueryText] = useState('')
  const [queryResults, setQueryResults] = useState<any[]>([])
  const [resultColumns, setResultColumns] = useState<any[]>([])
  const [resultError, setResultError] = useState<string | null>(null)
  const [queryType, setQueryType] = useState<'cypher' | 'sparql' | 'gremlin'>(
    'cypher'
  )
  const [templates, setTemplates] = useState<QueryTemplate[]>([])
  const [templatesLoading, setTemplatesLoading] = useState(false)
  const [templateModalVisible, setTemplateModalVisible] = useState(false)
  const [templateForm] = Form.useForm()

  const loadTemplates = async () => {
    setTemplatesLoading(true)
    try {
      const list = await knowledgeGraphService.getQueryTemplates()
      setTemplates(list)
    } catch (error) {
      logger.error('加载模板失败:', error)
      setTemplates([])
    } finally {
      setTemplatesLoading(false)
    }
  }

  useEffect(() => {
    loadTemplates()
  }, [])

  const normalizeRows = (rows: any[]) => {
    if (!rows.length) return { data: [], columns: [] }
    const first = rows[0]
    if (first && typeof first === 'object' && !Array.isArray(first)) {
      const columns = Object.keys(first).map(key => ({
        title: key,
        dataIndex: key,
        key,
      }))
      return {
        data: rows.map((row, index) => ({
          key: row.id || row.key || index,
          ...row,
        })),
        columns,
      }
    }
    return {
      data: rows.map((value, index) => ({ key: index, value })),
      columns: [{ title: 'value', dataIndex: 'value', key: 'value' }],
    }
  }

  const normalizeSparqlResult = (result: any) => {
    const bindings = result?.results?.results?.bindings
    if (Array.isArray(bindings)) {
      const vars = result?.results?.head?.vars || Object.keys(bindings[0] || {})
      const rows = bindings.map((item: any) => {
        const row: Record<string, any> = {}
        vars.forEach((v: string) => {
          row[v] = item?.[v]?.value ?? ''
        })
        return row
      })
      return normalizeRows(rows)
    }
    const columns = result?.results?.columns
    const data = result?.results?.data
    if (Array.isArray(columns) && Array.isArray(data)) {
      const rows = data.map((row: any[]) => {
        const item: Record<string, any> = {}
        columns.forEach((col: string, idx: number) => {
          item[col] = row[idx]
        })
        return item
      })
      return normalizeRows(rows)
    }
    return normalizeRows([])
  }

  const handleQuery = async () => {
    if (!queryText.trim()) {
      notification.warning({ message: '请输入查询语句' })
      return
    }

    setResultError(null)
    setLoading(true)
    try {
      if (queryType === 'sparql') {
        const result = await sparqlService.executeQuery({ query: queryText })
        const normalized = normalizeSparqlResult(result)
        setQueryResults(normalized.data)
        setResultColumns(normalized.columns)
        notification.success({
          message: `查询完成，返回 ${normalized.data.length} 条结果`,
        })
      } else if (queryType === 'cypher') {
        const result = await knowledgeGraphService.executeQuery({
          query: queryText,
          parameters: {},
          read_only: true,
        })
        const normalized = normalizeRows(
          Array.isArray(result?.data) ? result.data : []
        )
        setQueryResults(normalized.data)
        setResultColumns(normalized.columns)
        notification.success({
          message: `查询完成，返回 ${normalized.data.length} 条结果`,
        })
      } else {
        throw new Error('当前查询类型暂不支持')
      }
    } catch (error) {
      logger.error('查询失败:', error)
      const messageText = (error as Error).message || '查询失败'
      setResultError(messageText)
      setQueryResults([])
      setResultColumns([])
      notification.error({ message: '查询失败', description: messageText })
    } finally {
      setLoading(false)
    }
  }

  const handleOpenTemplateModal = () => {
    if (!queryText.trim()) {
      notification.warning({ message: '请输入查询语句' })
      return
    }
    templateForm.setFieldsValue({ name: '', description: '' })
    setTemplateModalVisible(true)
  }

  const handleSaveTemplate = async () => {
    try {
      const values = await templateForm.validateFields()
      await knowledgeGraphService.saveQueryTemplate({
        name: values.name,
        description: values.description || '',
        query: queryText.trim(),
        category: queryType,
        parameters: [],
      })
      setTemplateModalVisible(false)
      templateForm.resetFields()
      await loadTemplates()
      notification.success({ message: '模板已保存' })
    } catch (error) {
      if ((error as any)?.errorFields) return
      const messageText = (error as Error).message || '保存失败'
      notification.error({ message: '保存失败', description: messageText })
    }
  }

  const renderQueryInterface = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card
          title="图查询引擎"
          extra={
            <Space>
              <Select
                value={queryType}
                onChange={setQueryType}
                name="graph-query-type"
                style={{ width: 120 }}
              >
                <Select.Option value="cypher">Cypher</Select.Option>
                <Select.Option value="sparql">SPARQL</Select.Option>
                <Select.Option value="gremlin">Gremlin</Select.Option>
              </Select>
              <Button icon={<BookOutlined />}>语法帮助</Button>
            </Space>
          }
        >
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <TextArea
              rows={8}
              placeholder="请输入图查询语句...&#10;例如: MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.name"
              value={queryText}
              onChange={e => setQueryText(e.target.value)}
              name="graph-query-text"
              style={{ fontFamily: 'monospace' }}
            />
            <Row justify="space-between">
              <Col>
                <Space>
                  <Button
                    icon={<SaveOutlined />}
                    onClick={handleOpenTemplateModal}
                  >
                    保存查询
                  </Button>
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
        <Card
          title="查询结果"
          extra={
            <Space>
              <Button icon={<TableOutlined />}>表格视图</Button>
              <Button icon={<BarChartOutlined />}>图表视图</Button>
            </Space>
          }
        >
          {resultError && (
            <Alert
              type="error"
              message="查询失败"
              description={resultError}
              style={{ marginBottom: 12 }}
            />
          )}
          <Table
            dataSource={queryResults}
            columns={resultColumns}
            loading={loading}
            rowKey={(record, index) => record.key || record.id || index}
            locale={{ emptyText: '暂无结果' }}
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
          <Card
            title="查询模板库"
            extra={
              <Button icon={<ReloadOutlined />} onClick={loadTemplates}>
                刷新
              </Button>
            }
          >
            <List
              loading={templatesLoading}
              dataSource={templates}
              locale={{ emptyText: '暂无模板，请保存查询后使用' }}
              renderItem={item => (
                <List.Item
                  actions={[
                    <Button
                      type="link"
                      onClick={() => {
                        setQueryText(item.query)
                        if (item.category) {
                          setQueryType(
                            item.category as 'cypher' | 'sparql' | 'gremlin'
                          )
                        }
                        setActiveTab('query')
                      }}
                    >
                      使用模板
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    title={
                      <Space>
                        <Text strong>{item.name}</Text>
                        <Tag>{item.category}</Tag>
                      </Space>
                    }
                    description={
                      <div>
                        {item.description && (
                          <Text type="secondary">{item.description}</Text>
                        )}
                        <div>
                          <Text code>{item.query}</Text>
                        </div>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      <Modal
        title="保存查询模板"
        visible={templateModalVisible}
        onCancel={() => setTemplateModalVisible(false)}
        onOk={handleSaveTemplate}
        okText="保存"
        cancelText="取消"
      >
        <Form form={templateForm} layout="vertical">
          <Form.Item
            name="name"
            label="模板名称"
            rules={[{ required: true, message: '请输入模板名称' }]}
          >
            <Input />
          </Form.Item>
          <Form.Item name="description" label="模板描述">
            <TextArea rows={3} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default GraphQueryEnginePage
