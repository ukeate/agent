import React, { useState } from 'react';
import {
  Card,
  Input,
  Button,
  Row,
  Col,
  Slider,
  Form,
  Table,
  Tag,
  Typography,
  Space,
  Alert,
  Statistic,
  Progress,
  Divider,
  Radio,
  Switch,
  Tabs
} from 'antd';
import {
  SearchOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  BulbOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import { pgvectorApi } from '../../services/pgvectorApi';

const { TextArea } = Input;
const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface SearchResult {
  id: string;
  content: string;
  metadata: any;
  pg_distance?: number;
  qdrant_score?: number;
  fused_score: number;
  sources: string[];
  pg_rank?: number;
  qdrant_rank?: number;
}

interface SearchMetrics {
  total_time_ms: number;
  pg_time_ms: number;
  qdrant_time_ms: number;
  fusion_time_ms: number;
  cache_hit: boolean;
  results_count: number;
}

interface BenchmarkResult {
  method: string;
  avg_latency_ms: number;
  results_per_query: number;
  success_rate: number;
  accuracy_score: number;
}

const HybridRetrievalPanel: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchMetrics, setSearchMetrics] = useState<SearchMetrics | null>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [benchmarking, setBenchmarking] = useState(false);
  
  // 搜索配置
  const [searchConfig, setSearchConfig] = useState({
    top_k: 10,
    pg_weight: 0.7,
    qdrant_weight: 0.3,
    use_cache: true,
    quantize: true,
    search_mode: 'hybrid' as 'hybrid' | 'pg_only' | 'qdrant_only'
  });

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setSearching(true);
      const { results, metrics } = await pgvectorApi.hybridSearch({
        query: searchQuery,
        ...searchConfig
      });
      setSearchResults(results);
      setSearchMetrics(metrics);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setSearching(false);
    }
  };

  const handleBenchmark = async () => {
    try {
      setBenchmarking(true);
      const results = await pgvectorApi.benchmarkRetrievalMethods({
        test_queries: [
          searchQuery || '人工智能的发展历史',
          'machine learning algorithms',
          '向量数据库的优化方法'
        ],
        top_k: searchConfig.top_k
      });
      setBenchmarkResults(results);
    } catch (error) {
      console.error('Benchmark failed:', error);
    } finally {
      setBenchmarking(false);
    }
  };

  const getSourceColor = (sources: string[]) => {
    if (sources.includes('pgvector') && sources.includes('qdrant')) {
      return 'purple';
    } else if (sources.includes('pgvector')) {
      return 'blue';
    } else {
      return 'green';
    }
  };

  const getSourceText = (sources: string[]) => {
    if (sources.includes('pgvector') && sources.includes('qdrant')) {
      return '混合';
    } else if (sources.includes('pgvector')) {
      return 'PG';
    } else {
      return 'Qdrant';
    }
  };

  const resultColumns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 100,
      render: (id: string) => (
        <Text code>{id.substring(0, 8)}...</Text>
      )
    },
    {
      title: '内容',
      dataIndex: 'content',
      key: 'content',
      ellipsis: true,
      width: 300
    },
    {
      title: 'PG距离',
      dataIndex: 'pg_distance',
      key: 'pg_distance',
      width: 100,
      render: (distance: number | undefined) => 
        distance !== undefined ? distance.toFixed(4) : '-'
    },
    {
      title: 'Qdrant分数',
      dataIndex: 'qdrant_score',
      key: 'qdrant_score',
      width: 100,
      render: (score: number | undefined) => 
        score !== undefined ? score.toFixed(4) : '-'
    },
    {
      title: '融合分数',
      dataIndex: 'fused_score',
      key: 'fused_score',
      width: 100,
      render: (score: number) => (
        <Text strong style={{ color: '#1890ff' }}>
          {score.toFixed(4)}
        </Text>
      ),
      sorter: (a: SearchResult, b: SearchResult) => b.fused_score - a.fused_score
    },
    {
      title: '来源',
      dataIndex: 'sources',
      key: 'sources',
      width: 80,
      render: (sources: string[]) => (
        <Tag color={getSourceColor(sources)}>
          {getSourceText(sources)}
        </Tag>
      )
    }
  ];

  const benchmarkColumns = [
    {
      title: '检索方法',
      dataIndex: 'method',
      key: 'method',
      render: (method: string) => {
        const colorMap: { [key: string]: string } = {
          'hybrid': 'purple',
          'pg_only': 'blue',
          'qdrant_only': 'green'
        };
        return <Tag color={colorMap[method] || 'default'}>{method.toUpperCase()}</Tag>;
      }
    },
    {
      title: '平均延迟 (ms)',
      dataIndex: 'avg_latency_ms',
      key: 'avg_latency_ms',
      render: (latency: number) => latency.toFixed(1),
      sorter: (a: BenchmarkResult, b: BenchmarkResult) => a.avg_latency_ms - b.avg_latency_ms
    },
    {
      title: '平均结果数',
      dataIndex: 'results_per_query',
      key: 'results_per_query',
      render: (count: number) => count.toFixed(1)
    },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      render: (rate: number) => (
        <Progress 
          percent={rate * 100} 
          size="small" 
          format={(percent) => `${percent?.toFixed(1)}%`}
          strokeColor={rate > 0.9 ? '#52c41a' : rate > 0.7 ? '#faad14' : '#f5222d'}
        />
      )
    },
    {
      title: '准确性评分',
      dataIndex: 'accuracy_score',
      key: 'accuracy_score',
      render: (score: number) => (
        <span style={{ color: score > 0.8 ? '#52c41a' : score > 0.6 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(3)}
        </span>
      )
    }
  ];

  return (
    <div>
      <Tabs defaultActiveKey="search">
        <TabPane tab={<span><SearchOutlined />实时搜索</span>} key="search">
          {/* 搜索配置 */}
          <Row gutter={16}>
            <Col span={16}>
              <Card title="搜索查询" extra={<SearchOutlined />}>
                <Space.Compact style={{ width: '100%', marginBottom: 16 }}>
                  <TextArea
                    placeholder="输入搜索查询..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    rows={2}
                    style={{ flex: 1 }}
                  />
                  <Button
                    type="primary"
                    icon={<SearchOutlined />}
                    onClick={handleSearch}
                    loading={searching}
                    style={{ height: 'auto' }}
                  >
                    搜索
                  </Button>
                </Space.Compact>

                <Row gutter={16}>
                  <Col span={8}>
                    <Form.Item label="搜索模式">
                      <Radio.Group
                        value={searchConfig.search_mode}
                        onChange={(e) => setSearchConfig({ ...searchConfig, search_mode: e.target.value })}
                        size="small"
                      >
                        <Radio.Button value="hybrid">混合</Radio.Button>
                        <Radio.Button value="pg_only">PG</Radio.Button>
                        <Radio.Button value="qdrant_only">Qdrant</Radio.Button>
                      </Radio.Group>
                    </Form.Item>
                  </Col>
                  
                  <Col span={8}>
                    <Form.Item label={`返回结果: ${searchConfig.top_k}`}>
                      <Slider
                        min={1}
                        max={20}
                        value={searchConfig.top_k}
                        onChange={(value) => setSearchConfig({ ...searchConfig, top_k: value })}
                      />
                    </Form.Item>
                  </Col>
                  
                  <Col span={8}>
                    <Form.Item label="启用选项">
                      <Space direction="vertical">
                        <Switch
                          checked={searchConfig.use_cache}
                          onChange={(checked) => setSearchConfig({ ...searchConfig, use_cache: checked })}
                          checkedChildren="缓存"
                          unCheckedChildren="缓存"
                        />
                        <Switch
                          checked={searchConfig.quantize}
                          onChange={(checked) => setSearchConfig({ ...searchConfig, quantize: checked })}
                          checkedChildren="量化"
                          unCheckedChildren="量化"
                        />
                      </Space>
                    </Form.Item>
                  </Col>
                </Row>

                {/* 权重配置（仅混合模式） */}
                {searchConfig.search_mode === 'hybrid' && (
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label={`pgvector权重: ${searchConfig.pg_weight}`}>
                        <Slider
                          min={0}
                          max={1}
                          step={0.1}
                          value={searchConfig.pg_weight}
                          onChange={(value) => setSearchConfig({ 
                            ...searchConfig, 
                            pg_weight: value,
                            qdrant_weight: 1 - value
                          })}
                        />
                      </Form.Item>
                    </Col>
                    
                    <Col span={12}>
                      <Form.Item label={`Qdrant权重: ${searchConfig.qdrant_weight}`}>
                        <Slider
                          min={0}
                          max={1}
                          step={0.1}
                          value={searchConfig.qdrant_weight}
                          onChange={(value) => setSearchConfig({ 
                            ...searchConfig, 
                            qdrant_weight: value,
                            pg_weight: 1 - value
                          })}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                )}
              </Card>
            </Col>

            <Col span={8}>
              <Card title="搜索指标" extra={<BarChartOutlined />}>
                {searchMetrics ? (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Statistic
                      title="总耗时"
                      value={searchMetrics.total_time_ms}
                      precision={1}
                      suffix="ms"
                      prefix={<ThunderboltOutlined />}
                    />
                    
                    <Divider style={{ margin: '8px 0' }} />
                    
                    <div>
                      <Text type="secondary">pgvector: {searchMetrics.pg_time_ms?.toFixed(1) || 0}ms</Text>
                    </div>
                    <div>
                      <Text type="secondary">Qdrant: {searchMetrics.qdrant_time_ms?.toFixed(1) || 0}ms</Text>
                    </div>
                    <div>
                      <Text type="secondary">融合: {searchMetrics.fusion_time_ms?.toFixed(1) || 0}ms</Text>
                    </div>
                    
                    <Divider style={{ margin: '8px 0' }} />
                    
                    <div>
                      <Tag color={searchMetrics.cache_hit ? 'green' : 'red'}>
                        {searchMetrics.cache_hit ? '缓存命中' : '缓存未命中'}
                      </Tag>
                    </div>
                    
                    <div>
                      <Text>结果数量: {searchMetrics.results_count}</Text>
                    </div>
                  </Space>
                ) : (
                  <div style={{ textAlign: 'center', padding: '40px' }}>
                    <Text type="secondary">执行搜索查看指标</Text>
                  </div>
                )}
              </Card>
            </Col>
          </Row>

          {/* 搜索结果 */}
          {searchResults.length > 0 && (
            <Card 
              title={`搜索结果 (${searchResults.length}条)`}
              style={{ marginTop: 16 }}
              extra={<BulbOutlined />}
            >
              <Table
                dataSource={searchResults.map((result, index) => ({ ...result, key: index }))}
                columns={resultColumns}
                size="small"
                pagination={{ pageSize: 10, showSizeChanger: false }}
                scroll={{ y: 400 }}
              />
            </Card>
          )}
        </TabPane>

        <TabPane tab={<span><BarChartOutlined />性能基准</span>} key="benchmark">
          <Card title="检索方法性能对比" extra={<LineChartOutlined />}>
            <div style={{ marginBottom: 16 }}>
              <Button
                type="primary"
                icon={<BarChartOutlined />}
                onClick={handleBenchmark}
                loading={benchmarking}
              >
                运行性能基准测试
              </Button>
              <Text type="secondary" style={{ marginLeft: 16 }}>
                对比不同检索方法的性能表现
              </Text>
            </div>

            {benchmarkResults.length > 0 && (
              <>
                <Table
                  dataSource={benchmarkResults.map((result, index) => ({ ...result, key: index }))}
                  columns={benchmarkColumns}
                  size="small"
                  pagination={false}
                />

                <Alert
                  style={{ marginTop: 16 }}
                  message="基准测试结果分析"
                  description={
                    <div>
                      <p><strong>性能对比：</strong></p>
                      <ul>
                        {benchmarkResults.map(result => (
                          <li key={result.method}>
                            <strong>{result.method.toUpperCase()}</strong>: 
                            平均延迟 {result.avg_latency_ms.toFixed(1)}ms, 
                            成功率 {(result.success_rate * 100).toFixed(1)}%, 
                            准确性 {result.accuracy_score.toFixed(3)}
                          </li>
                        ))}
                      </ul>
                      <p><strong>建议：</strong>根据延迟、准确性和稳定性需求选择合适的检索策略。</p>
                    </div>
                  }
                  variant="default"
                  showIcon
                />
              </>
            )}
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default HybridRetrievalPanel;