/**
 * 索引管理面板
 *
 * 展示多种索引类型的管理和优化功能：
 * - HNSW索引参数调优
 * - IVF索引配置
 * - LSH哈希索引设置
 * - 自适应索引选择
 */

import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Form,
  Select,
  Slider,
  Button,
  Table,
  Tag,
  Space,
  Progress,
  Alert,
  Statistic,
  message,
  Tooltip,
  Typography,
} from 'antd'
import {
  ThunderboltOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons'
import { pgvectorApi } from '../../services/pgvectorApi'

const { Option } = Select
const { Text, Title } = Typography

interface IndexConfig {
  type: 'hnsw' | 'ivf' | 'lsh' | 'flat'
  name: string
  table: string
  column: string
  parameters: Record<string, any>
  status: 'active' | 'building' | 'error' | 'optimizing'
  performance: {
    latency_p50: number
    latency_p95: number
    recall: number
    memory_usage: number
  }
}

const IndexManagerPanel: React.FC = () => {
  const [indexes, setIndexes] = useState<IndexConfig[]>([])
  const [selectedIndex, setSelectedIndex] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    loadIndexes()
  }, [])

  const loadIndexes = async () => {
    setLoading(true)
    try {
      const list = await pgvectorApi.listIndexes()
      const mapped = list.map((item: any, idx: number): IndexConfig => {
        const def: string = item.indexdef || ''
        const type = def.toLowerCase().includes('hnsw')
          ? 'hnsw'
          : def.toLowerCase().includes('ivfflat')
            ? 'ivf'
            : def.toLowerCase().includes('ls_hnsw')
              ? 'lsh'
              : 'flat'
        return {
          type,
          name: item.indexname || `idx_${idx}`,
          table: item.tablename || '',
          column:
            def.split('(').pop()?.split(')')[0]?.split(',')[0]?.trim() || '',
          parameters: {},
          status: 'active',
          performance: {
            latency_p50: NaN,
            latency_p95: NaN,
            recall: NaN,
            memory_usage: NaN,
          },
        }
      })
      setIndexes(mapped)
    } catch (error) {
      message.error('加载索引信息失败')
      setIndexes([])
    } finally {
      setLoading(false)
    }
  }

  const handleCreateIndex = async (values: any) => {
    setLoading(true)
    try {
      const { type, name, table, column, parameters } = values
      await pgvectorApi.createOptimizedIndex({
        table_name: table,
        vector_column: column,
        index_type: type,
        config: parameters,
      })
      message.success('索引创建成功')
      loadIndexes()
    } catch (error) {
      message.error('创建索引失败')
    } finally {
      setLoading(false)
    }
  }

  const handleOptimizeIndex = async (indexName: string) => {
    setLoading(true)
    try {
      // 后端暂未提供单独优化接口，复用创建逻辑为占位
      message.info('索引优化需后端支持，目前未实现')
    } catch (error) {
      message.error('优化索引失败')
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success'
      case 'building':
        return 'processing'
      case 'error':
        return 'error'
      case 'optimizing':
        return 'warning'
      default:
        return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircleOutlined />
      case 'building':
        return <PlayCircleOutlined />
      case 'error':
        return <ExclamationCircleOutlined />
      case 'optimizing':
        return <SettingOutlined spin />
      default:
        return <InfoCircleOutlined />
    }
  }

  const renderHNSWParameters = () => (
    <Card size="small" title="HNSW 参数配置">
      <Form.Item
        label="M (连接数)"
        name={['parameters', 'm']}
        htmlFor={undefined}
      >
        <Slider
          min={4}
          max={64}
          step={4}
          marks={{ 16: '16', 32: '32', 48: '48' }}
        />
      </Form.Item>
      <Form.Item
        label="ef_construction"
        name={['parameters', 'ef_construction']}
        htmlFor={undefined}
      >
        <Slider
          min={100}
          max={800}
          step={50}
          marks={{ 200: '200', 400: '400', 600: '600' }}
        />
      </Form.Item>
      <Form.Item
        label="ef_search"
        name={['parameters', 'ef_search']}
        htmlFor={undefined}
      >
        <Slider
          min={50}
          max={400}
          step={25}
          marks={{ 100: '100', 200: '200', 300: '300' }}
        />
      </Form.Item>
    </Card>
  )

  const renderIVFParameters = () => (
    <Card size="small" title="IVF 参数配置">
      <Form.Item
        label="Lists (聚类数)"
        name={['parameters', 'lists']}
        htmlFor={undefined}
      >
        <Slider
          min={50}
          max={500}
          step={25}
          marks={{ 100: '100', 200: '200', 300: '300' }}
        />
      </Form.Item>
      <Form.Item
        label="Probes (探测数)"
        name={['parameters', 'probes']}
        htmlFor={undefined}
      >
        <Slider
          min={5}
          max={50}
          step={5}
          marks={{ 10: '10', 20: '20', 30: '30' }}
        />
      </Form.Item>
    </Card>
  )

  const renderLSHParameters = () => (
    <Card size="small" title="LSH 参数配置">
      <Form.Item
        label="Hash Tables"
        name={['parameters', 'n_tables']}
        htmlFor={undefined}
      >
        <Slider
          min={4}
          max={16}
          step={2}
          marks={{ 8: '8', 12: '12', 16: '16' }}
        />
      </Form.Item>
      <Form.Item
        label="Hash Bits"
        name={['parameters', 'n_bits']}
        htmlFor={undefined}
      >
        <Slider
          min={8}
          max={20}
          step={2}
          marks={{ 12: '12', 16: '16', 20: '20' }}
        />
      </Form.Item>
    </Card>
  )

  const columns = [
    {
      title: '索引名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: IndexConfig) => (
        <Space>
          <Text strong>{name}</Text>
          <Tag color={getStatusColor(record.status)}>
            {getStatusIcon(record.status)}
            {record.status.toUpperCase()}
          </Tag>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag
          color={type === 'hnsw' ? 'blue' : type === 'ivf' ? 'green' : 'orange'}
        >
          {type.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: '表/列',
      key: 'table_column',
      render: (record: IndexConfig) => (
        <Text>
          {record.table}.{record.column}
        </Text>
      ),
    },
    {
      title: '性能指标',
      key: 'performance',
      render: (record: IndexConfig) => (
        <Space direction="vertical" size="small">
          <Text type="secondary">
            P50:{' '}
            {Number.isFinite(record.performance.latency_p50)
              ? `${record.performance.latency_p50}ms`
              : '—'}
          </Text>
          <Text type="secondary">
            召回:{' '}
            {Number.isFinite(record.performance.recall)
              ? `${(record.performance.recall * 100).toFixed(1)}%`
              : '—'}
          </Text>
        </Space>
      ),
    },
    {
      title: '内存使用',
      key: 'memory',
      render: (record: IndexConfig) =>
        Number.isFinite(record.performance.memory_usage) ? (
          <Statistic
            value={record.performance.memory_usage}
            suffix="MB"
            valueStyle={{ fontSize: 14 }}
          />
        ) : (
          <Text type="secondary">—</Text>
        ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: IndexConfig) => (
        <Space>
          <Button
            size="small"
            onClick={() => handleOptimizeIndex(record.name)}
            disabled={record.status !== 'active'}
            loading={loading}
          >
            优化
          </Button>
          <Button size="small" type="link">
            详情
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div>
      {/* 功能说明 */}
      <Alert
        message="索引管理功能"
        description="管理和优化多种向量索引类型，包括HNSW图索引、IVF倒排索引和LSH哈希索引。可以实时调整参数并监控性能指标。"
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        {/* 左侧：创建索引 */}
        <Col span={8}>
          <Card title="创建新索引" size="small">
            <Form
              form={form}
              layout="vertical"
              onFinish={handleCreateIndex}
              initialValues={{
                type: 'hnsw',
                parameters: { m: 16, ef_construction: 200, ef_search: 100 },
              }}
            >
              <Form.Item label="索引类型" name="type">
                <Select>
                  <Option value="hnsw">
                    <Space>
                      <ThunderboltOutlined />
                      HNSW (推荐)
                    </Space>
                  </Option>
                  <Option value="ivf">IVF (倒排)</Option>
                  <Option value="lsh">LSH (哈希)</Option>
                  <Option value="flat">FLAT (暴力)</Option>
                </Select>
              </Form.Item>

              <Form.Item label="表名" name="table">
                <Select placeholder="选择表">
                  <Option value="documents">documents</Option>
                  <Option value="images">images</Option>
                  <Option value="audio">audio</Option>
                </Select>
              </Form.Item>

              <Form.Item label="向量列" name="column">
                <Select placeholder="选择向量列">
                  <Option value="embedding">embedding</Option>
                  <Option value="vector">vector</Option>
                </Select>
              </Form.Item>

              {/* 动态参数配置 */}
              <Form.Item shouldUpdate={(prev, curr) => prev.type !== curr.type}>
                {({ getFieldValue }) => {
                  const indexType = getFieldValue('type')
                  switch (indexType) {
                    case 'hnsw':
                      return renderHNSWParameters()
                    case 'ivf':
                      return renderIVFParameters()
                    case 'lsh':
                      return renderLSHParameters()
                    default:
                      return null
                  }
                }}
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  block
                >
                  创建索引
                </Button>
              </Form.Item>
            </Form>
          </Card>

          <Card title="索引概览" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={12}>
                <Statistic title="索引数量" value={indexes.length} />
              </Col>
              <Col span={12}>
                <Statistic
                  title="最新刷新"
                  valueStyle={{ fontSize: 12 }}
                  value={new Date().toLocaleTimeString()}
                />
              </Col>
            </Row>
            <Text type="secondary">
              数据直接来源于 /pgvector/indexes/list，无模拟值。
            </Text>
          </Card>
        </Col>

        {/* 右侧：索引列表 */}
        <Col span={16}>
          <Card
            title="现有索引"
            size="small"
            extra={
              <Button onClick={loadIndexes} loading={loading}>
                刷新
              </Button>
            }
          >
            <Table
              columns={columns}
              dataSource={indexes}
              rowKey="name"
              size="small"
              pagination={false}
              loading={loading}
            />
          </Card>

          {/* 自适应索引选择建议 */}
          <Card title="智能索引建议" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#f6ffed' }}>
                  <Space direction="vertical" size="small">
                    <Text strong>高精度场景</Text>
                    <Text type="secondary">推荐 HNSW</Text>
                    <Text type="secondary">召回率 &gt; 95%</Text>
                  </Space>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#f0f9ff' }}>
                  <Space direction="vertical" size="small">
                    <Text strong>大数据场景</Text>
                    <Text type="secondary">推荐 IVF</Text>
                    <Text type="secondary">平衡性能与内存</Text>
                  </Space>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#fff7e6' }}>
                  <Space direction="vertical" size="small">
                    <Text strong>低延迟场景</Text>
                    <Text type="secondary">推荐 LSH</Text>
                    <Text type="secondary">亚毫秒响应</Text>
                  </Space>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default IndexManagerPanel
