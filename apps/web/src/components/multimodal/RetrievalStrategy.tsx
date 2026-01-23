/**
 * 检索策略可视化组件
 * 展示智能检索策略的决策过程和权重分配
 */

import React from 'react'
import {
  Card,
  Progress,
  Tag,
  Space,
  Typography,
  Descriptions,
  Empty,
  Row,
  Col,
  Badge,
} from 'antd'
import {
  SearchOutlined,
  FileTextOutlined,
  FileImageOutlined,
  TableOutlined,
  MergeCellsOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'

const { Text } = Typography

interface RetrievalWeights {
  text: number
  image: number
  table: number
}

interface StrategyData {
  strategy: 'text' | 'visual' | 'document' | 'hybrid'
  weights: RetrievalWeights
  reranking: boolean
  top_k: number
  similarity_threshold: number
}

interface RetrievalStrategyProps {
  strategy: StrategyData | null
}

const RetrievalStrategy: React.FC<RetrievalStrategyProps> = ({ strategy }) => {
  if (!strategy) {
    return (
      <Card title="检索策略决策" size="small">
        <Empty description="等待查询分析..." />
      </Card>
    )
  }

  const getStrategyIcon = () => {
    switch (strategy.strategy) {
      case 'text':
        return <FileTextOutlined />
      case 'visual':
        return <FileImageOutlined />
      case 'document':
        return <TableOutlined />
      case 'hybrid':
        return <MergeCellsOutlined />
      default:
        return <SearchOutlined />
    }
  }

  const getStrategyColor = () => {
    switch (strategy.strategy) {
      case 'text':
        return 'blue'
      case 'visual':
        return 'purple'
      case 'document':
        return 'green'
      case 'hybrid':
        return 'orange'
      default:
        return 'default'
    }
  }

  const getStrategyDescription = () => {
    switch (strategy.strategy) {
      case 'text':
        return '纯文本向量相似度检索'
      case 'visual':
        return '多模态图像嵌入检索'
      case 'document':
        return '结构化文档和表格检索'
      case 'hybrid':
        return '多路召回融合检索'
      default:
        return '未知策略'
    }
  }

  return (
    <Card
      title={
        <span>
          <ThunderboltOutlined className="mr-2" />
          智能检索策略
        </span>
      }
      size="small"
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* 策略类型 */}
        <div>
          <Text type="secondary">选定策略:</Text>
          <div className="mt-2">
            <Tag
              icon={getStrategyIcon()}
              color={getStrategyColor()}
              style={{ fontSize: 14, padding: '4px 12px' }}
            >
              {strategy.strategy.toUpperCase()}
            </Tag>
            <Text className="ml-2">{getStrategyDescription()}</Text>
          </div>
        </div>

        {/* 权重分配 */}
        <div>
          <Text type="secondary">检索权重分配:</Text>
          <Row gutter={8} className="mt-2">
            <Col span={8}>
              <div>
                <FileTextOutlined className="mr-1" />
                <Text style={{ fontSize: 12 }}>文本</Text>
              </div>
              <Progress
                percent={Math.round(strategy.weights.text * 100)}
                size="small"
                strokeColor="#1890ff"
              />
            </Col>
            <Col span={8}>
              <div>
                <FileImageOutlined className="mr-1" />
                <Text style={{ fontSize: 12 }}>图像</Text>
              </div>
              <Progress
                percent={Math.round(strategy.weights.image * 100)}
                size="small"
                strokeColor="#722ed1"
              />
            </Col>
            <Col span={8}>
              <div>
                <TableOutlined className="mr-1" />
                <Text style={{ fontSize: 12 }}>表格</Text>
              </div>
              <Progress
                percent={Math.round(strategy.weights.table * 100)}
                size="small"
                strokeColor="#52c41a"
              />
            </Col>
          </Row>
        </div>

        {/* 决策理由 */}
        {/* 技术参数 */}
        <Descriptions size="small" column={2}>
          <Descriptions.Item label="Top-K">
            <Badge
              count={strategy.top_k}
              style={{ backgroundColor: '#52c41a' }}
            />
          </Descriptions.Item>
          <Descriptions.Item label="相似度阈值">
            <Text code>{strategy.similarity_threshold}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="重排序">
            {strategy.reranking ? (
              <Tag color="success">启用</Tag>
            ) : (
              <Tag color="default">禁用</Tag>
            )}
          </Descriptions.Item>
        </Descriptions>
      </Space>
    </Card>
  )
}

export default RetrievalStrategy
