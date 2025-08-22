/**
 * 混合搜索面板
 * 
 * 展示语义搜索与关键词搜索的融合功能：
 * - 纯语义向量搜索
 * - BM25关键词搜索
 * - RRF融合策略
 * - 查询扩展和重排序
 */

import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Input,
  Button,
  Radio,
  Slider,
  Switch,
  Space,
  Table,
  Tag,
  Progress,
  Alert,
  Typography,
  message,
  Statistic,
  Divider
} from 'antd';
import {
  SearchOutlined,
  ThunderboltOutlined,
  FileTextOutlined,
  MergeOutlined,
  RobotOutlined,
  StarOutlined
} from '@ant-design/icons';

const { TextArea } = Input;
const { Text, Title } = Typography;

interface SearchResult {
  id: string;
  title: string;
  content: string;
  semantic_score: number;
  keyword_score: number;
  final_score: number;
  rank: number;
  source: 'semantic' | 'keyword' | 'hybrid';
}

interface SearchStats {
  total_time_ms: number;
  semantic_time_ms: number;
  keyword_time_ms: number;
  fusion_time_ms: number;
  total_candidates: number;
  semantic_candidates: number;
  keyword_candidates: number;
}

const HybridSearchPanel: React.FC = () => {
  const [searchMode, setSearchMode] = useState<'semantic' | 'keyword' | 'hybrid'>('hybrid');
  const [query, setQuery] = useState('');
  const [semanticWeight, setSemanticWeight] = useState(0.7);
  const [enableExpansion, setEnableExpansion] = useState(true);
  const [enableReranking, setEnableReranking] = useState(true);
  const [fusionMethod, setFusionMethod] = useState<'rrf' | 'linear'>('rrf');
  
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchStats, setSearchStats] = useState<SearchStats | null>(null);
  const [searching, setSearching] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) {
      message.warning('请输入搜索查询');
      return;
    }

    setSearching(true);
    try {
      // Mock search results
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const mockResults: SearchResult[] = [
        {
          id: '1',
          title: '深度学习在自然语言处理中的应用',
          content: '深度学习技术在NLP领域取得了重大突破，特别是在文本分类、语义理解等方面...',
          semantic_score: 0.92,
          keyword_score: 0.78,
          final_score: 0.87,
          rank: 1,
          source: 'hybrid'
        },
        {
          id: '2',
          title: 'Transformer架构详解',
          content: 'Transformer模型革命性地改变了自然语言处理领域，其自注意力机制...',
          semantic_score: 0.89,
          keyword_score: 0.65,
          final_score: 0.82,
          rank: 2,
          source: 'semantic'
        },
        {
          id: '3',
          title: '向量数据库在AI系统中的作用',
          content: '向量数据库为AI应用提供了高效的相似度搜索能力，支持大规模向量检索...',
          semantic_score: 0.85,
          keyword_score: 0.88,
          final_score: 0.86,
          rank: 3,
          source: 'keyword'
        }
      ];

      const mockStats: SearchStats = {
        total_time_ms: 127,
        semantic_time_ms: 45,
        keyword_time_ms: 32,
        fusion_time_ms: 12,
        total_candidates: 156,
        semantic_candidates: 89,
        keyword_candidates: 67
      };

      setResults(mockResults);
      setSearchStats(mockStats);
      message.success('搜索完成');
    } catch (error) {
      message.error('搜索失败');
    } finally {
      setSearching(false);
    }
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'semantic': return 'blue';
      case 'keyword': return 'green';
      case 'hybrid': return 'purple';
      default: return 'default';
    }
  };

  const renderSearchConfig = () => (
    <Card title="搜索配置" size="small">
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* 搜索模式 */}
        <div>
          <Text strong>搜索模式</Text>
          <Radio.Group 
            value={searchMode} 
            onChange={(e) => setSearchMode(e.target.value)}
            style={{ marginTop: 8 }}
          >
            <Radio.Button value="semantic">
              <ThunderboltOutlined /> 语义搜索
            </Radio.Button>
            <Radio.Button value="keyword">
              <FileTextOutlined /> 关键词
            </Radio.Button>
            <Radio.Button value="hybrid">
              <MergeOutlined /> 混合搜索
            </Radio.Button>
          </Radio.Group>
        </div>

        {/* 混合搜索参数 */}
        {searchMode === 'hybrid' && (
          <>
            <Divider />
            <div>
              <Text strong>语义权重: {semanticWeight}</Text>
              <Slider
                min={0.1}
                max={0.9}
                step={0.1}
                value={semanticWeight}
                onChange={setSemanticWeight}
                marks={{ 0.3: '0.3', 0.5: '0.5', 0.7: '0.7' }}
              />
            </div>

            <div>
              <Text strong>融合方法</Text>
              <Radio.Group 
                value={fusionMethod} 
                onChange={(e) => setFusionMethod(e.target.value)}
                style={{ marginTop: 8 }}
              >
                <Radio value="rrf">RRF (推荐)</Radio>
                <Radio value="linear">线性加权</Radio>
              </Radio.Group>
            </div>
          </>
        )}

        <Divider />

        {/* 高级选项 */}
        <div>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text>查询扩展</Text>
              <Switch 
                checked={enableExpansion} 
                onChange={setEnableExpansion}
                size="small"
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text>结果重排序</Text>
              <Switch 
                checked={enableReranking} 
                onChange={setEnableReranking}
                size="small"
              />
            </div>
          </Space>
        </div>
      </Space>
    </Card>
  );

  const renderSearchStats = () => {
    if (!searchStats) return null;

    return (
      <Card title="搜索统计" size="small">
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Statistic
              title="总耗时"
              value={searchStats.total_time_ms}
              suffix="ms"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="候选数量"
              value={searchStats.total_candidates}
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col span={12}>
            <Text type="secondary">语义搜索: {searchStats.semantic_time_ms}ms</Text>
            <Progress 
              percent={(searchStats.semantic_time_ms / searchStats.total_time_ms) * 100} 
              size="small" 
              strokeColor="#1890ff"
              showInfo={false}
            />
          </Col>
          <Col span={12}>
            <Text type="secondary">关键词搜索: {searchStats.keyword_time_ms}ms</Text>
            <Progress 
              percent={(searchStats.keyword_time_ms / searchStats.total_time_ms) * 100} 
              size="small" 
              strokeColor="#52c41a"
              showInfo={false}
            />
          </Col>
        </Row>
      </Card>
    );
  };

  const columns = [
    {
      title: '排名',
      dataIndex: 'rank',
      key: 'rank',
      width: 60,
      render: (rank: number) => (
        <Text strong style={{ fontSize: 16 }}>{rank}</Text>
      ),
    },
    {
      title: '标题',
      dataIndex: 'title',
      key: 'title',
      render: (title: string, record: SearchResult) => (
        <Space direction="vertical" size="small">
          <Text strong>{title}</Text>
          <Tag color={getSourceColor(record.source)}>
            {record.source === 'semantic' ? '语义' : 
             record.source === 'keyword' ? '关键词' : '混合'}
          </Tag>
        </Space>
      ),
    },
    {
      title: '内容预览',
      dataIndex: 'content',
      key: 'content',
      render: (content: string) => (
        <Text ellipsis style={{ maxWidth: 300 }}>
          {content}
        </Text>
      ),
    },
    {
      title: '分数详情',
      key: 'scores',
      render: (record: SearchResult) => (
        <Space direction="vertical" size="small">
          <div>
            <Text type="secondary">语义: </Text>
            <Text>{(record.semantic_score * 100).toFixed(1)}%</Text>
          </div>
          <div>
            <Text type="secondary">关键词: </Text>
            <Text>{(record.keyword_score * 100).toFixed(1)}%</Text>
          </div>
          <div>
            <Text strong>最终: {(record.final_score * 100).toFixed(1)}%</Text>
          </div>
        </Space>
      ),
    },
    {
      title: '相关性',
      dataIndex: 'final_score',
      key: 'final_score',
      render: (score: number) => (
        <Progress
          type="circle"
          percent={score * 100}
          width={50}
          strokeColor={score > 0.8 ? '#52c41a' : score > 0.6 ? '#1890ff' : '#faad14'}
        />
      ),
    },
  ];

  return (
    <div>
      {/* 功能说明 */}
      <Alert
        message="混合搜索引擎"
        description="结合语义向量搜索和BM25关键词搜索的优势，使用RRF融合算法和智能重排序，提供更准确的搜索结果。"
        variant="default"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        {/* 左侧：搜索界面 */}
        <Col span={8}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {/* 搜索输入 */}
            <Card title="搜索查询" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <TextArea
                  placeholder="输入搜索查询，支持自然语言和关键词"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={3}
                />
                <Button 
                  type="primary" 
                  icon={<SearchOutlined />} 
                  onClick={handleSearch}
                  loading={searching}
                  block
                >
                  搜索
                </Button>
              </Space>
            </Card>

            {/* 搜索配置 */}
            {renderSearchConfig()}

            {/* 搜索统计 */}
            {renderSearchStats()}
          </Space>
        </Col>

        {/* 右侧：搜索结果 */}
        <Col span={16}>
          <Card 
            title={
              <Space>
                <SearchOutlined />
                搜索结果
                {results.length > 0 && (
                  <Tag color="blue">{results.length} 条结果</Tag>
                )}
              </Space>
            }
            size="small"
          >
            {results.length > 0 ? (
              <Table
                columns={columns}
                dataSource={results}
                rowKey="id"
                size="small"
                pagination={{ pageSize: 10, showSizeChanger: false }}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <SearchOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">输入查询开始搜索</Text>
                </div>
              </div>
            )}
          </Card>

          {/* 搜索算法说明 */}
          <Card title="算法说明" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#f0f9ff' }}>
                  <Space direction="vertical" align="center">
                    <ThunderboltOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                    <Text strong>语义搜索</Text>
                    <Text type="secondary">基于向量相似度</Text>
                    <Text type="secondary">理解查询意图</Text>
                  </Space>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#f6ffed' }}>
                  <Space direction="vertical" align="center">
                    <FileTextOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                    <Text strong>关键词搜索</Text>
                    <Text type="secondary">BM25算法</Text>
                    <Text type="secondary">精确匹配</Text>
                  </Space>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#fff7e6' }}>
                  <Space direction="vertical" align="center">
                    <MergeOutlined style={{ fontSize: 24, color: '#fa8c16' }} />
                    <Text strong>RRF融合</Text>
                    <Text type="secondary">排名融合</Text>
                    <Text type="secondary">最优结果</Text>
                  </Space>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default HybridSearchPanel;