/**
 * 多模态结果展示组件
 * 展示检索结果和最终答案，突出技术细节
 */

import React, { useState } from 'react';
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
  Progress,
  Alert,
  Divider,
  Row,
  Col,
  Statistic
} from 'antd';
import {
  FileTextOutlined,
  FileImageOutlined,
  TableOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  CodeOutlined,
  ThunderboltOutlined,
  DatabaseOutlined
} from '@ant-design/icons';

const { Text, Title, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Panel } = Collapse;

interface RetrievalItem {
  content: string;
  score: number;
  source: string;
  metadata: {
    chunk_id?: string;
    page?: number;
    type?: string;
  };
}

interface RetrievalResultsData {
  texts: RetrievalItem[];
  images: RetrievalItem[];
  tables: RetrievalItem[];
  totalResults: number;
  retrievalTimeMs: number;
  sources: string[];
}

interface QAResponseData {
  answer: string;
  confidence: number;
  processingTime: number;
  tokensUsed: number;
  modelUsed: string;
  contextLength: number;
}

interface MultimodalResultsProps {
  retrievalResults: RetrievalResultsData | null;
  qaResponse: QAResponseData | null;
}

const MultimodalResults: React.FC<MultimodalResultsProps> = ({ 
  retrievalResults, 
  qaResponse 
}) => {
  const [activeTab, setActiveTab] = useState('answer');

  if (!retrievalResults && !qaResponse) {
    return (
      <Card title="查询结果" size="small">
        <Empty description="等待查询执行..." />
      </Card>
    );
  }

  const renderRetrievalItem = (item: RetrievalItem, type: string) => {
    const getTypeIcon = () => {
      switch (type) {
        case 'text':
          return <FileTextOutlined style={{ color: '#1890ff' }} />;
        case 'image':
          return <FileImageOutlined style={{ color: '#722ed1' }} />;
        case 'table':
          return <TableOutlined style={{ color: '#52c41a' }} />;
        default:
          return <FileTextOutlined />;
      }
    };

    return (
      <List.Item>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Space>
            {getTypeIcon()}
            <Text strong>{item.source}</Text>
            <Badge 
              count={`相似度: ${(item.score * 100).toFixed(1)}%`}
              style={{ backgroundColor: item.score > 0.8 ? '#52c41a' : '#faad14' }}
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
    );
  };

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
                      color: qaResponse.confidence > 0.8 ? '#52c41a' : '#faad14' 
                    }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="处理时间"
                    value={qaResponse.processingTime}
                    suffix="ms"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Token使用"
                    value={qaResponse.tokensUsed}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="上下文长度"
                    value={qaResponse.contextLength}
                  />
                </Col>
              </Row>

              {/* 技术细节 */}
              <Alert
                message="生成技术细节"
                description={
                  <Space direction="vertical" size="small">
                    <Text>• 模型: {qaResponse.modelUsed || 'GPT-4o'}</Text>
                    <Text>• 温度参数: 0.7</Text>
                    <Text>• 上下文窗口: 128K tokens</Text>
                    <Text>• 流式响应: 启用</Text>
                  </Space>
                }
                variant="default"
                icon={<InfoCircleOutlined />}
              />
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
                      value={retrievalResults.totalResults}
                      prefix={<DatabaseOutlined />}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="检索耗时"
                      value={retrievalResults.retrievalTimeMs}
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
                        <Badge count={retrievalResults.texts.length} className="ml-2" />
                      </span>
                    } 
                    key="texts"
                  >
                    <List
                      dataSource={retrievalResults.texts}
                      renderItem={(item) => renderRetrievalItem(item, 'text')}
                    />
                  </Panel>
                )}

                {retrievalResults.images.length > 0 && (
                  <Panel 
                    header={
                      <span>
                        <FileImageOutlined /> 图像结果
                        <Badge count={retrievalResults.images.length} className="ml-2" />
                      </span>
                    } 
                    key="images"
                  >
                    <List
                      dataSource={retrievalResults.images}
                      renderItem={(item) => renderRetrievalItem(item, 'image')}
                    />
                  </Panel>
                )}

                {retrievalResults.tables.length > 0 && (
                  <Panel 
                    header={
                      <span>
                        <TableOutlined /> 表格结果
                        <Badge count={retrievalResults.tables.length} className="ml-2" />
                      </span>
                    } 
                    key="tables"
                  >
                    <List
                      dataSource={retrievalResults.tables}
                      renderItem={(item) => renderRetrievalItem(item, 'table')}
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

        <TabPane 
          tab={
            <span>
              <CodeOutlined />
              技术追踪
            </span>
          } 
          key="technical"
        >
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Alert
              message="RAG Pipeline执行路径"
              variant="default"
              description={
                <ol style={{ paddingLeft: 20 }}>
                  <li>查询分析 (Query Analyzer) - 识别查询类型</li>
                  <li>检索策略选择 (Retrieval Strategy) - 动态权重分配</li>
                  <li>向量检索 (Vector Search) - Chroma多路召回</li>
                  <li>结果重排序 (Reranking) - MMR多样性优化</li>
                  <li>上下文组装 (Context Assembly) - 多模态内容整合</li>
                  <li>LLM生成 (GPT-4o Generation) - 流式响应输出</li>
                </ol>
              }
            />

            <Card size="small" title="性能分析">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text>查询分析</Text>
                  <Progress percent={100} size="small" steps={6} />
                </div>
                <div>
                  <Text>向量检索</Text>
                  <Progress percent={100} size="small" steps={6} strokeColor="#52c41a" />
                </div>
                <div>
                  <Text>上下文组装</Text>
                  <Progress percent={100} size="small" steps={6} strokeColor="#1890ff" />
                </div>
                <div>
                  <Text>LLM生成</Text>
                  <Progress percent={100} size="small" steps={6} strokeColor="#722ed1" />
                </div>
              </Space>
            </Card>

            <Alert
              message="优化建议"
              type="success"
              description={
                <ul style={{ paddingLeft: 20, marginBottom: 0 }}>
                  <li>启用查询缓存以减少重复计算</li>
                  <li>使用批量嵌入处理提高吞吐量</li>
                  <li>实施向量索引预热策略</li>
                  <li>优化chunk size和overlap参数</li>
                </ul>
              }
            />
          </Space>
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default MultimodalResults;