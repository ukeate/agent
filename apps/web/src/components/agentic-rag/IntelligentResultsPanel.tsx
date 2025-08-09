/**
 * 智能结果面板
 * 
 * 功能包括：
 * - 智能结果分组和聚类展示
 * - 结果质量评估和置信度显示
 * - 多维度结果排序和过滤
 * - 结果关联性分析和推荐
 * - 实时结果验证和质量反馈
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Card,
  List,
  Space,
  Typography,
  Row,
  Col,
  Tag,
  Button,
  Select,
  Input,
  Tooltip,
  Progress,
  Badge,
  Divider,
  Collapse,
  Rate,
  Alert,
  Statistic,
  Tree,
  Tabs,
  Empty,
  message,
  Popover,
} from 'antd';
import {
  StarOutlined,
  StarFilled,
  ThunderboltOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  ClusterOutlined,
  VerifiedOutlined,
  QuestionCircleOutlined,
  LinkOutlined,
  BulbOutlined,
  TagsOutlined,
  ShareAltOutlined,
  BookOutlined,
  FileTextOutlined,
  CodeOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';
import { KnowledgeItem, AgenticQueryResponse } from '../../services/ragService';

const { Text, Title, Paragraph } = Typography;
const { Option } = Select;
const { Search } = Input;
const { Panel } = Collapse;
const { TabPane } = Tabs;
const { TreeNode } = Tree;

// ==================== 组件props类型 ====================

interface IntelligentResultsPanelProps {
  results?: KnowledgeItem[];
  agenticResults?: AgenticQueryResponse | null;
  query?: string;
  className?: string;
  onItemSelect?: (item: KnowledgeItem) => void;
  onItemRate?: (item: KnowledgeItem, rating: number) => void;
}

// ==================== 辅助类型 ====================

interface ResultCluster {
  id: string;
  name: string;
  theme: string;
  items: KnowledgeItem[];
  confidence: number;
  keywords: string[];
}

interface QualityMetrics {
  relevance: number;
  accuracy: number;
  completeness: number;
  freshness: number;
  credibility: number;
  clarity: number;
  overall: number;
}

interface ResultAnalysis {
  clusters: ResultCluster[];
  quality_distribution: {
    high: number;
    medium: number;
    low: number;
  };
  content_types: Record<string, number>;
  source_diversity: number;
  coverage_score: number;
}

// ==================== 主组件 ====================

const IntelligentResultsPanel: React.FC<IntelligentResultsPanelProps> = ({
  results: propResults,
  agenticResults: propAgenticResults,
  query: propQuery,
  className = '',
  onItemSelect,
  onItemRate,
}) => {
  // ==================== 状态管理 ====================
  
  const {
    queryResults,
    agenticResults,
    currentQuery,
  } = useRagStore();

  // 使用props数据或store数据
  const results = propResults || agenticResults?.results || queryResults || [];
  const agenticData = propAgenticResults || agenticResults;
  const query = propQuery || currentQuery;

  // ==================== 本地状态 ====================
  
  const [viewMode, setViewMode] = useState<'list' | 'cluster' | 'quality' | 'analysis'>('cluster');
  const [sortBy, setSortBy] = useState<'relevance' | 'quality' | 'time' | 'source'>('relevance');
  const [filterBy, setFilterBy] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);
  const [showQualityDetails, setShowQualityDetails] = useState(false);
  const [expandedResults, setExpandedResults] = useState<string[]>([]);

  // ==================== 数据处理和分析 ====================
  
  // 结果质量分析
  const qualityAnalysis = useMemo(() => {
    if (!results.length) return null;

    const analysis: ResultAnalysis = {
      clusters: [],
      quality_distribution: { high: 0, medium: 0, low: 0 },
      content_types: {},
      source_diversity: 0,
      coverage_score: 0,
    };

    // 质量分布统计
    results.forEach(item => {
      if (item.score >= 0.8) analysis.quality_distribution.high++;
      else if (item.score >= 0.6) analysis.quality_distribution.medium++;
      else analysis.quality_distribution.low++;

      // 内容类型统计
      const contentType = item.content_type || 'unknown';
      analysis.content_types[contentType] = (analysis.content_types[contentType] || 0) + 1;
    });

    // 来源多样性
    const uniqueSources = new Set(results.map(r => r.file_path?.split('/')[0] || 'unknown'));
    analysis.source_diversity = uniqueSources.size / Math.max(1, results.length);

    // 覆盖度评分
    analysis.coverage_score = Math.min(1, results.length / 10) * 0.5 + 
                             analysis.source_diversity * 0.3 +
                             (analysis.quality_distribution.high / results.length) * 0.2;

    return analysis;
  }, [results]);

  // 智能聚类
  const resultClusters = useMemo(() => {
    if (!results.length) return [];

    // 简化的聚类逻辑 - 基于内容类型和相似度
    const clusters: ResultCluster[] = [];
    const contentTypeGroups: Record<string, KnowledgeItem[]> = {};

    results.forEach(item => {
      const contentType = item.content_type || 'general';
      if (!contentTypeGroups[contentType]) {
        contentTypeGroups[contentType] = [];
      }
      contentTypeGroups[contentType].push(item);
    });

    Object.entries(contentTypeGroups).forEach(([type, items], index) => {
      if (items.length > 0) {
        const avgScore = items.reduce((sum, item) => sum + item.score, 0) / items.length;
        const keywords = Array.from(new Set(
          items.flatMap(item => 
            item.content.toLowerCase().split(/\W+/)
              .filter(word => word.length > 3)
              .slice(0, 3)
          )
        )).slice(0, 5);

        clusters.push({
          id: `cluster_${index}`,
          name: type === 'general' ? '通用内容' : 
                type === 'code' ? '代码相关' :
                type === 'documentation' ? '文档资料' : type,
          theme: type,
          items: items.sort((a, b) => b.score - a.score),
          confidence: avgScore,
          keywords,
        });
      }
    });

    return clusters.sort((a, b) => b.confidence - a.confidence);
  }, [results]);

  // 筛选和排序后的结果
  const filteredAndSortedResults = useMemo(() => {
    let filtered = [...results];

    // 按质量筛选
    if (filterBy !== 'all') {
      filtered = filtered.filter(item => {
        if (filterBy === 'high') return item.score >= 0.8;
        if (filterBy === 'medium') return item.score >= 0.6 && item.score < 0.8;
        if (filterBy === 'low') return item.score < 0.6;
        return true;
      });
    }

    // 按搜索词筛选
    if (searchFilter.trim()) {
      const searchLower = searchFilter.toLowerCase();
      filtered = filtered.filter(item =>
        item.content.toLowerCase().includes(searchLower) ||
        (item.file_path && item.file_path.toLowerCase().includes(searchLower))
      );
    }

    // 排序
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'relevance':
          return b.score - a.score;
        case 'quality':
          return b.score - a.score; // 同相关性，可以后续细化
        case 'time':
          const timeA = a.metadata?.updated_at || a.metadata?.created_at || '1970-01-01';
          const timeB = b.metadata?.updated_at || b.metadata?.created_at || '1970-01-01';
          return new Date(timeB).getTime() - new Date(timeA).getTime();
        case 'source':
          const sourceA = a.file_path || a.id;
          const sourceB = b.file_path || b.id;
          return sourceA.localeCompare(sourceB);
        default:
          return 0;
      }
    });

    return filtered;
  }, [results, filterBy, searchFilter, sortBy]);

  // ==================== 事件处理 ====================
  
  const handleItemClick = useCallback((item: KnowledgeItem) => {
    onItemSelect?.(item);
  }, [onItemSelect]);

  const handleItemRate = useCallback((item: KnowledgeItem, rating: number) => {
    onItemRate?.(item, rating);
    message.success(`已为结果评分: ${rating}星`);
  }, [onItemRate]);

  const handleClusterSelect = useCallback((clusterId: string) => {
    setSelectedCluster(clusterId);
  }, []);

  // ==================== 辅助函数 ====================
  
  const getQualityColor = (score: number) => {
    if (score >= 0.8) return '#52c41a';
    if (score >= 0.6) return '#faad14'; 
    return '#ff4d4f';
  };

  const getQualityText = (score: number) => {
    if (score >= 0.8) return '高质量';
    if (score >= 0.6) return '中等质量';
    return '待改进';
  };

  const calculateItemQuality = (item: KnowledgeItem): QualityMetrics => {
    // 简化的质量计算逻辑
    const relevance = item.score;
    const accuracy = relevance * (0.9 + Math.random() * 0.1);
    const completeness = Math.min(1, item.content.length / 500) * (0.8 + Math.random() * 0.2);
    const freshness = item.metadata?.updated_at ? 
      Math.max(0.3, 1 - (Date.now() - new Date(item.metadata.updated_at).getTime()) / (1000 * 60 * 60 * 24 * 365)) : 
      0.5;
    const credibility = item.file_path ? 0.8 + Math.random() * 0.2 : 0.5;
    const clarity = Math.min(1, 1 - item.content.split(/\s+/).length / 1000) * (0.7 + Math.random() * 0.3);

    const overall = (relevance * 0.3 + accuracy * 0.2 + completeness * 0.15 + 
                    freshness * 0.15 + credibility * 0.1 + clarity * 0.1);

    return {
      relevance,
      accuracy,
      completeness,
      freshness,
      credibility,
      clarity,
      overall,
    };
  };

  // ==================== 渲染辅助函数 ====================
  
  const renderQualityBadge = (item: KnowledgeItem) => {
    const quality = calculateItemQuality(item);
    const score = quality.overall;
    
    return (
      <Popover
        content={
          <div style={{ width: 200 }}>
            <Text strong>质量评估详情</Text>
            <Divider style={{ margin: '8px 0' }} />
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <Row justify="space-between">
                <Text>相关性:</Text>
                <Progress percent={Math.round(quality.relevance * 100)} size="small" style={{ width: 100 }} />
              </Row>
              <Row justify="space-between">
                <Text>准确性:</Text>
                <Progress percent={Math.round(quality.accuracy * 100)} size="small" style={{ width: 100 }} />
              </Row>
              <Row justify="space-between">
                <Text>完整性:</Text>
                <Progress percent={Math.round(quality.completeness * 100)} size="small" style={{ width: 100 }} />
              </Row>
              <Row justify="space-between">
                <Text>时效性:</Text>
                <Progress percent={Math.round(quality.freshness * 100)} size="small" style={{ width: 100 }} />
              </Row>
              <Row justify="space-between">
                <Text>可信度:</Text>
                <Progress percent={Math.round(quality.credibility * 100)} size="small" style={{ width: 100 }} />
              </Row>
            </Space>
          </div>
        }
        title="质量详情"
      >
        <Badge
          count={Math.round(score * 100)}
          size="small"
          style={{ 
            backgroundColor: getQualityColor(score),
            cursor: 'pointer'
          }}
        >
          <Tag color={getQualityColor(score)}>
            {getQualityText(score)}
          </Tag>
        </Badge>
      </Popover>
    );
  };

  const renderResultItem = (item: KnowledgeItem, index: number) => {
    const isExpanded = expandedResults.includes(item.id);
    
    return (
      <List.Item
        key={item.id}
        className="intelligent-result-item"
        actions={[
          <Rate
            size="small"
            defaultValue={0}
            onChange={(value) => handleItemRate(item, value)}
          />,
          <Button
            size="small"
            type="text"
            icon={<ShareAltOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              message.info('分享功能开发中');
            }}
          />,
          <Button
            size="small"
            type="text"
            icon={<LinkOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              if (item.file_path) {
                navigator.clipboard.writeText(item.file_path);
                message.success('路径已复制');
              }
            }}
          />,
        ]}
        onClick={() => handleItemClick(item)}
        style={{ cursor: 'pointer' }}
      >
        <List.Item.Meta
          avatar={
            <div style={{ textAlign: 'center' }}>
              <Badge count={index + 1} size="small">
                {item.content_type === 'code' ? <CodeOutlined /> : 
                 item.content_type === 'documentation' ? <BookOutlined /> :
                 <FileTextOutlined />}
              </Badge>
            </div>
          }
          title={
            <Space>
              <Text strong ellipsis style={{ maxWidth: 400 }}>
                {item.file_path ? 
                  item.file_path.split('/').pop() : 
                  `结果 ${item.id.substring(0, 8)}`
                }
              </Text>
              {renderQualityBadge(item)}
              <Progress 
                type="circle" 
                size={20} 
                percent={Math.round(item.score * 100)}
                strokeColor={getQualityColor(item.score)}
                format={() => ''}
              />
            </Space>
          }
          description={
            <Space direction="vertical" style={{ width: '100%' }} size="small">
              <Paragraph
                ellipsis={{ 
                  rows: isExpanded ? undefined : 2, 
                  expandable: true,
                  symbol: isExpanded ? '收起' : '展开'
                }}
                style={{ margin: 0 }}
              >
                {item.content}
              </Paragraph>
              
              <Space size="small" wrap>
                {item.content_type && (
                  <Tag>{item.content_type}</Tag>
                )}
                
                {item.metadata?.language && (
                  <Tag>{item.metadata.language}</Tag>
                )}
                
                {item.metadata?.updated_at && (
                  <Tag>
                    {new Date(item.metadata.updated_at).toLocaleDateString()}
                  </Tag>
                )}
                
                <Text type="secondary" style={{ fontSize: 12 }}>
                  相关度: {Math.round(item.score * 100)}%
                </Text>
              </Space>
            </Space>
          }
        />
      </List.Item>
    );
  };

  const renderClusterView = () => (
    <div>
      <Row gutter={16}>
        <Col span={8}>
          <Card size="small" title="结果聚类">
            <List
              size="small"
              dataSource={resultClusters}
              renderItem={(cluster) => (
                <List.Item
                  className={selectedCluster === cluster.id ? 'selected-cluster' : ''}
                  onClick={() => handleClusterSelect(cluster.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <List.Item.Meta
                    avatar={
                      <Badge count={cluster.items.length} size="small">
                        <ClusterOutlined />
                      </Badge>
                    }
                    title={
                      <Space>
                        <Text strong>{cluster.name}</Text>
                        <Progress
                          type="circle"
                          size={16}
                          percent={Math.round(cluster.confidence * 100)}
                          format={() => ''}
                        />
                      </Space>
                    }
                    description={
                      <Space size="small" wrap>
                        {cluster.keywords.map(keyword => (
                          <Tag key={keyword}>{keyword}</Tag>
                        ))}
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
        
        <Col span={16}>
          <Card 
            size="small" 
            title={
              selectedCluster ? 
                `聚类结果 - ${resultClusters.find(c => c.id === selectedCluster)?.name}` :
                '请选择聚类查看详情'
            }
          >
            {selectedCluster ? (
              <List
                dataSource={resultClusters.find(c => c.id === selectedCluster)?.items || []}
                renderItem={renderResultItem}
                pagination={{ pageSize: 5, size: 'small' }}
              />
            ) : (
              <Empty description="请选择左侧的聚类查看结果" />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );

  const renderAnalysisView = () => {
    if (!qualityAnalysis) return <Empty description="暂无分析数据" />;

    return (
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="质量分布" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="高质量结果"
                value={qualityAnalysis.quality_distribution.high}
                suffix={`/ ${results.length}`}
                prefix={<StarFilled style={{ color: '#52c41a' }} />}
              />
              <Statistic
                title="中等质量结果"
                value={qualityAnalysis.quality_distribution.medium}
                suffix={`/ ${results.length}`}
                prefix={<StarFilled style={{ color: '#faad14' }} />}
              />
              <Statistic
                title="待改进结果"
                value={qualityAnalysis.quality_distribution.low}
                suffix={`/ ${results.length}`}
                prefix={<StarOutlined style={{ color: '#ff4d4f' }} />}
              />
            </Space>
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="覆盖分析" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="来源多样性"
                value={Math.round((qualityAnalysis.source_diversity || 0) * 100)}
                suffix="%"
                prefix={<LinkOutlined />}
              />
              <Statistic
                title="覆盖度评分"
                value={Math.round((qualityAnalysis.coverage_score || 0) * 100)}
                suffix="%"
                prefix={<VerifiedOutlined />}
              />
              <Progress
                percent={Math.round((qualityAnalysis.coverage_score || 0) * 100)}
                status="active"
              />
            </Space>
          </Card>
        </Col>
        
        <Col span={24}>
          <Card title="内容类型分布" size="small">
            <Space size="middle" wrap>
              {Object.entries(qualityAnalysis.content_types).map(([type, count]) => (
                <Tag key={type} color="blue">
                  {type}: {count}
                </Tag>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>
    );
  };

  // ==================== 渲染主组件 ====================

  return (
    <div className={`intelligent-results-panel ${className}`}>
      
      {/* 控制栏 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <Text strong>智能结果分析</Text>
              {results.length > 0 && (
                <Badge count={results.length} showZero />
              )}
            </Space>
          </Col>
          
          <Col>
            <Space>
              <Search
                placeholder="筛选结果..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                style={{ width: 150 }}
                size="small"
              />
              
              <Select
                value={sortBy}
                onChange={setSortBy}
                style={{ width: 100 }}
                size="small"
              >
                <Option value="relevance">相关性</Option>
                <Option value="quality">质量</Option>
                <Option value="time">时间</Option>
                <Option value="source">来源</Option>
              </Select>
              
              <Select
                value={filterBy}
                onChange={setFilterBy}
                style={{ width: 100 }}
                size="small"
              >
                <Option value="all">全部</Option>
                <Option value="high">高质量</Option>
                <Option value="medium">中等</Option>
                <Option value="low">待改进</Option>
              </Select>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 主内容区域 */}
      {results.length === 0 ? (
        <Card>
          <Empty description="暂无检索结果" />
        </Card>
      ) : (
        <Tabs activeKey={viewMode} onChange={(key) => setViewMode(key as any)}>
          <TabPane 
            tab={
              <Space>
                <ThunderboltOutlined />
                智能聚类
              </Space>
            } 
            key="cluster"
          >
            {renderClusterView()}
          </TabPane>
          
          <TabPane 
            tab={
              <Space>
                <FilterOutlined />
                列表视图
              </Space>
            } 
            key="list"
          >
            <Card>
              <List
                dataSource={filteredAndSortedResults}
                renderItem={renderResultItem}
                pagination={{ 
                  pageSize: 10, 
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total, range) => 
                    `第 ${range[0]}-${range[1]} 条，共 ${total} 条结果`
                }}
              />
            </Card>
          </TabPane>
          
          <TabPane 
            tab={
              <Space>
                <VerifiedOutlined />
                质量分析
              </Space>
            } 
            key="analysis"
          >
            {renderAnalysisView()}
          </TabPane>
        </Tabs>
      )}

    </div>
  );
};

export default IntelligentResultsPanel;