/**
 * 多模态结果展示组件
 * 展示检索结果和最终答案，突出技术细节
 */

import React, { useState } from 'react'
import {
  Card,
  Tabs,
  List,
  Tag,
  Space,
  Typography,
  Empty,
  Collapse,
  Badge,
  Row,
  Col,
  Statistic,
} from 'antd'
import {
  FileTextOutlined,
  FileImageOutlined,
  TableOutlined,
  CheckCircleOutlined,
  CodeOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
} from '@ant-design/icons'

const { Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Panel } = Collapse

interface RetrievalItem {
  content: string
  score: number
  source: string
  metadata: {
    chunk_id?: string
    page?: number
    type?: string
  }
}

interface RetrievalResultsData {
  texts: RetrievalItem[]
  images: RetrievalItem[]
  tables: RetrievalItem[]
  total_results: number
  retrieval_time_ms: number
  sources: string[]
}

interface QAResponseData {
  answer: string
  confidence: number
  processing_time_ms: number
  context_used: Record<string, number>
}

interface MultimodalResultsProps {
  retrievalResults: RetrievalResultsData | null
  qaResponse: QAResponseData | null
}

const MultimodalResults: React.FC<MultimodalResultsProps> = ({
  retrievalResults,
  qaResponse,
}) => {
  const [activeTab, setActiveTab] = useState('answer')

  if (!retrievalResults && !qaResponse) {
    return (
      <Card title="查询结果" size="small">
        <Empty description="等待查询执行..." />
      </Card>
    )
  }

  const renderRetrievalItem = (item: RetrievalItem, type: string) => {
    const getTypeIcon = () => {
      switch (type) {
        case 'text':
          return <FileTextOutlined style={{ color: '#1890ff' }} />
        case 'image':
          return <FileImageOutlined style={{ color: '#722ed1' }} />
        case 'table':
          return <TableOutlined style={{ color: '#52c41a' }} />
        default:
          return <FileTextOutlined />
      }
    }

    return (
      <List.Item>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Space>
            {getTypeIcon()}
            <Text strong>{item.source}</Text>
            <Badge
              count={`相似度: ${(item.score * 100).toFixed(1)}%`}
              style={{
                backgroundColor: item.score > 0.8 ? '#52c41a' : '#faad14',
              }}
            />
          </Space>

          <Paragraph
            ellipsis={{ rows: 3, expandable: true }}
            style={{ marginBottom: 0 }}
          >
            {item.content}
          </Paragraph>

          {item.metadata && Object.keys(item.metadata).length > 0 && (
            <Space wrap>
              {item.metadata.chunk_id && (
                <Tag color="blue">
                  <CodeOutlined /> {item.metadata.chunk_id}
                </Tag>
              )}
              {item.metadata.page && (
                <Tag color="cyan">第 {item.metadata.page} 页</Tag>
              )}
              {item.metadata.type && (
                <Tag color="purple">{item.metadata.type}</Tag>
              )}
            </Space>
          )}
        </Space>
      </List.Item>
    )
  }

  return (
    <Card
      title={
        <span>
          <ThunderboltOutlined className="mr-2" />
          多模态RAG结果
        </span>
      }
      size="small"
    >
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane
          tab={
            <span>
              <CheckCircleOutlined />
              最终答案
            </span>
          }
          key="answer"
        >
          {qaResponse ? (
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              {/* 答案内容 */}
              <Card size="small">
                <Paragraph>{qaResponse.answer}</Paragraph>
              </Card>

              {/* 生成统计 */}
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="置信度"
                    value={qaResponse.confidence * 100}
                    suffix="%"
                    precision={1}
                    valueStyle={{
                      color:
                        qaResponse.confidence > 0.8 ? '#52c41a' : '#faad14',
                    }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="处理时间"
                    value={qaResponse.processing_time_ms}
                    suffix="ms"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="文本块"
                    value={qaResponse.context_used?.text_chunks || 0}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="结果数"
                    value={retrievalResults?.total_results || 0}
                  />
                </Col>
              </Row>
            </Space>
          ) : (
            <Empty description="等待答案生成..." />
          )}
        </TabPane>

        <TabPane
          tab={
            <span>
              <DatabaseOutlined />
              检索结果
            </span>
          }
          key="retrieval"
        >
          {retrievalResults ? (
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              {/* 检索统计 */}
              <Card size="small">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="总结果数"
                      value={retrievalResults.total_results}
                      prefix={<DatabaseOutlined />}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="检索耗时"
                      value={retrievalResults.retrieval_time_ms}
                      suffix="ms"
                      prefix={<ThunderboltOutlined />}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="数据源"
                      value={retrievalResults.sources.length}
                      prefix={<FileTextOutlined />}
                    />
                  </Col>
                </Row>
              </Card>

              {/* 分类结果 */}
              <Collapse defaultActiveKey={['texts']}>
                {retrievalResults.texts.length > 0 && (
                  <Panel
                    header={
                      <span>
                        <FileTextOutlined /> 文本结果
                        <Badge
                          count={retrievalResults.texts.length}
                          className="ml-2"
                        />
                      </span>
                    }
                    key="texts"
                  >
                    <List
                      dataSource={retrievalResults.texts}
                      renderItem={item => renderRetrievalItem(item, 'text')}
                    />
                  </Panel>
                )}

                {retrievalResults.images.length > 0 && (
                  <Panel
                    header={
                      <span>
                        <FileImageOutlined /> 图像结果
                        <Badge
                          count={retrievalResults.images.length}
                          className="ml-2"
                        />
                      </span>
                    }
                    key="images"
                  >
                    <List
                      dataSource={retrievalResults.images}
                      renderItem={item => renderRetrievalItem(item, 'image')}
                    />
                  </Panel>
                )}

                {retrievalResults.tables.length > 0 && (
                  <Panel
                    header={
                      <span>
                        <TableOutlined /> 表格结果
                        <Badge
                          count={retrievalResults.tables.length}
                          className="ml-2"
                        />
                      </span>
                    }
                    key="tables"
                  >
                    <List
                      dataSource={retrievalResults.tables}
                      renderItem={item => renderRetrievalItem(item, 'table')}
                    />
                  </Panel>
                )}
              </Collapse>

              {/* 数据源列表 */}
              <Card size="small" title="数据源">
                <Space wrap>
                  {retrievalResults.sources.map((source, idx) => (
                    <Tag key={idx} icon={<FileTextOutlined />}>
                      {source}
                    </Tag>
                  ))}
                </Space>
              </Card>
            </Space>
          ) : (
            <Empty description="等待检索结果..." />
          )}
        </TabPane>
      </Tabs>
    </Card>
  )
}

export default MultimodalResults
