/**
 * RAG检索结果列表展示
 * 
 * 功能包括：
 * - 结果卡片布局，显示标题、内容片段、相关性分数
 * - 来源信息展示（文件路径、内容类型、修改时间）
 * - 结果高亮显示匹配的查询词和上下文
 * - 结果排序（相关性/时间）和分页功能
 * - 结果导出和分享功能
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Card,
  List,
  Typography,
  Space,
  Tag,
  Progress,
  Button,
  Select,
  Pagination,
  Empty,
  Tooltip,
  Drawer,
  Modal,
  Input,
  message,
  Divider,
  Row,
  Col,
} from 'antd';
import {
  FileTextOutlined,
  CodeOutlined,
  LinkOutlined,
  ShareAltOutlined,
  DownloadOutlined,
  CopyOutlined,
  ExpandOutlined,
  ClockCircleOutlined,
  StarOutlined,
  FilterOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';
import { KnowledgeItem } from '../../services/ragService';

const { Text, Title, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

// ==================== 组件props类型 ====================

interface RagResultsListProps {
  results?: KnowledgeItem[];
  query?: string;
  loading?: boolean;
  onItemClick?: (item: KnowledgeItem) => void;
  className?: string;
  pageSize?: number;
}

// ==================== 辅助类型 ====================

type SortType = 'relevance' | 'time' | 'filename';
type ViewMode = 'card' | 'list';

// ==================== 主组件 ====================

const RagResultsList: React.FC<RagResultsListProps> = ({
  results: propResults,
  query: propQuery,
  loading = false,
  onItemClick,
  className = '',
  pageSize = 10,
}) => {
  // ==================== 状态管理 ====================
  
  const { queryResults, currentQuery } = useRagStore();
  
  // 使用props数据或store数据
  const results = propResults || queryResults;
  const query = propQuery || currentQuery;

  // ==================== 本地状态 ====================
  
  const [currentPage, setCurrentPage] = useState(1);
  const [sortType, setSortType] = useState<SortType>('relevance');
  const [viewMode, setViewMode] = useState<ViewMode>('card');
  const [selectedItem, setSelectedItem] = useState<KnowledgeItem | null>(null);
  const [showDetailDrawer, setShowDetailDrawer] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareUrl, setShareUrl] = useState('');

  // ==================== 数据处理逻辑 ====================
  
  // 排序逻辑
  const sortedResults = useMemo(() => {
    if (!results || results.length === 0) return [];
    
    const sorted = [...results];
    
    switch (sortType) {
      case 'relevance':
        return sorted.sort((a, b) => b.score - a.score);
      case 'time':
        return sorted.sort((a, b) => {
          const timeA = a.metadata?.updated_at || a.metadata?.created_at || '1970-01-01';
          const timeB = b.metadata?.updated_at || b.metadata?.created_at || '1970-01-01';
          return new Date(timeB).getTime() - new Date(timeA).getTime();
        });
      case 'filename':
        return sorted.sort((a, b) => {
          const nameA = a.file_path || a.id;
          const nameB = b.file_path || b.id;
          return nameA.localeCompare(nameB);
        });
      default:
        return sorted;
    }
  }, [results, sortType]);

  // 分页数据
  const paginatedResults = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    return sortedResults.slice(startIndex, startIndex + pageSize);
  }, [sortedResults, currentPage, pageSize]);

  // ==================== 内容高亮逻辑 ====================
  
  const highlightContent = useCallback((content: string, searchQuery: string) => {
    if (!searchQuery.trim()) return content;
    
    // 提取查询词（去除逻辑操作符）
    const keywords = searchQuery
      .replace(/\b(AND|OR|NOT)\b/gi, '')
      .replace(/[\"()]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 1);
    
    let highlightedContent = content;
    
    keywords.forEach(keyword => {
      const regex = new RegExp(`(${keyword})`, 'gi');
      highlightedContent = highlightedContent.replace(
        regex, 
        '<mark style="background-color: #fff566; padding: 1px 2px;">$1</mark>'
      );
    });
    
    return highlightedContent;
  }, []);

  // ==================== 事件处理 ====================
  
  const handleItemClick = useCallback((item: KnowledgeItem) => {
    setSelectedItem(item);
    setShowDetailDrawer(true);
    onItemClick?.(item);
  }, [onItemClick]);

  const handleSortChange = useCallback((value: SortType) => {
    setSortType(value);
    setCurrentPage(1);
  }, []);

  const handlePageChange = useCallback((page: number) => {
    setCurrentPage(page);
  }, []);

  const handleCopyContent = useCallback((content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      message.success('内容已复制到剪贴板');
    }).catch(() => {
      message.error('复制失败');
    });
  }, []);

  const handleExportResults = useCallback(() => {
    const exportData = {
      query,
      timestamp: new Date().toISOString(),
      results: results.map(item => ({
        content: item.content,
        file_path: item.file_path,
        score: item.score,
        metadata: item.metadata,
      })),
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `rag_results_${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    message.success('结果已导出');
  }, [results, query]);

  const handleShareResults = useCallback(() => {
    const shareData = {
      query,
      results_count: results.length,
      timestamp: new Date().toISOString(),
    };
    
    // 生成分享链接（实际应用中需要后端支持）
    const shareParams = new URLSearchParams({
      q: query,
      r: results.length.toString(),
      t: Date.now().toString(),
    });
    
    const url = `${window.location.origin}/rag/share?${shareParams.toString()}`;
    setShareUrl(url);
    setShowShareModal(true);
  }, [query, results]);

  // ==================== 辅助函数 ====================
  
  const getFileIcon = useCallback((contentType?: string, filePath?: string) => {
    if (!contentType && !filePath) return <FileTextOutlined />;
    
    const type = contentType || '';
    const path = filePath || '';
    
    if (type.includes('code') || path.match(/\.(js|ts|py|java|cpp|c|go|rs)$/)) {
      return <CodeOutlined />;
    }
    
    return <FileTextOutlined />;
  }, []);

  const formatFileSize = useCallback((size?: number) => {
    if (!size) return '';
    
    if (size < 1024) return `${size}B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)}KB`;
    return `${(size / 1024 / 1024).toFixed(1)}MB`;
  }, []);

  const truncateContent = useCallback((content: string, maxLength: number = 200) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  }, []);

  // ==================== 渲染结果项 ====================
  
  const renderResultItem = useCallback((item: KnowledgeItem) => {
    const scorePercentage = Math.round(item.score * 100);
    const scoreColor = item.score >= 0.8 ? '#52c41a' : item.score >= 0.6 ? '#faad14' : '#ff4d4f';
    
    return (
      <Card
        hoverable
        size="small"
        className="result-item-card"
        onClick={() => handleItemClick(item)}
        style={{ marginBottom: 12 }}
        actions={[
          <Tooltip title="复制内容">
            <Button 
              type="text" 
              size="small" 
              icon={<CopyOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                handleCopyContent(item.content);
              }}
            />
          </Tooltip>,
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<ExpandOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                handleItemClick(item);
              }}
            />
          </Tooltip>,
        ]}
      >
        {/* 头部信息 */}
        <Row align="middle" style={{ marginBottom: 8 }}>
          <Col flex="auto">
            <Space size="small">
              {getFileIcon(item.content_type, item.file_path)}
              <Text strong ellipsis style={{ maxWidth: 300 }}>
                {item.file_path ? 
                  item.file_path.split('/').pop() : 
                  `文档 ${item.id.substring(0, 8)}`
                }
              </Text>
            </Space>
          </Col>
          <Col>
            <Space size="small">
              <Progress
                type="circle"
                size={24}
                percent={scorePercentage}
                strokeColor={scoreColor}
                format={() => ''}
              />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {scorePercentage}%
              </Text>
            </Space>
          </Col>
        </Row>

        {/* 内容片段 */}
        <Paragraph
          style={{ marginBottom: 8 }}
          ellipsis={{ rows: 3, expandable: false }}
        >
          <div 
            dangerouslySetInnerHTML={{ 
              __html: highlightContent(truncateContent(item.content), query) 
            }}
          />
        </Paragraph>

        {/* 元数据标签 */}
        <Space size="small" wrap>
          {item.content_type && (
            <Tag color="blue">
              {item.content_type}
            </Tag>
          )}
          
          {item.metadata?.language && (
            <Tag>
              {item.metadata.language}
            </Tag>
          )}
          
          {item.metadata?.size && (
            <Tag>
              {formatFileSize(item.metadata.size)}
            </Tag>
          )}
          
          {item.metadata?.updated_at && (
            <Tag icon={<ClockCircleOutlined />}>
              {new Date(item.metadata.updated_at).toLocaleDateString()}
            </Tag>
          )}
          
          {item.file_path && (
            <Tooltip title={item.file_path}>
              <Tag icon={<LinkOutlined />}>
                路径
              </Tag>
            </Tooltip>
          )}
        </Space>
      </Card>
    );
  }, [handleItemClick, handleCopyContent, getFileIcon, highlightContent, query, truncateContent, formatFileSize]);

  // ==================== 渲染组件 ====================

  return (
    <div className={`rag-results-list ${className}`}>
      
      {/* 结果头部控制栏 */}
      {results && results.length > 0 && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Row align="middle" justify="space-between">
            <Col>
              <Space>
                <Text strong>
                  找到 {results.length} 个结果
                </Text>
                {query && (
                  <Text type="secondary">
                    for "{query}"
                  </Text>
                )}
              </Space>
            </Col>
            
            <Col>
              <Space>
                {/* 排序选择 */}
                <Select
                  value={sortType}
                  onChange={handleSortChange}
                  style={{ width: 120 }}
                  size="small"
                  suffixIcon={<FilterOutlined />}
                >
                  <Option value="relevance">相关性</Option>
                  <Option value="time">时间</Option>
                  <Option value="filename">文件名</Option>
                </Select>

                {/* 导出按钮 */}
                <Tooltip title="导出结果">
                  <Button
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={handleExportResults}
                  />
                </Tooltip>

                {/* 分享按钮 */}
                <Tooltip title="分享结果">
                  <Button
                    size="small"
                    icon={<ShareAltOutlined />}
                    onClick={handleShareResults}
                  />
                </Tooltip>
              </Space>
            </Col>
          </Row>
        </Card>
      )}

      {/* 结果列表 */}
      {loading ? (
        <Card>
          <List
            loading
            dataSource={[1, 2, 3]}
            renderItem={() => <List.Item />}
          />
        </Card>
      ) : results && results.length > 0 ? (
        <div>
          {/* 结果项列表 */}
          <div style={{ minHeight: 400 }}>
            {paginatedResults.map(renderResultItem)}
          </div>

          {/* 分页控件 */}
          {sortedResults.length > pageSize && (
            <Row justify="center" style={{ marginTop: 24 }}>
              <Pagination
                current={currentPage}
                total={sortedResults.length}
                pageSize={pageSize}
                onChange={handlePageChange}
                showSizeChanger={false}
                showQuickJumper
                showTotal={(total, range) => 
                  `${range[0]}-${range[1]} of ${total} 个结果`
                }
              />
            </Row>
          )}
        </div>
      ) : (
        <Card>
          <Empty
            description={query ? `未找到相关结果` : '请输入查询词开始搜索'}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </Card>
      )}

      {/* 详情抽屉 */}
      <Drawer
        title="结果详情"
        placement="right"
        width={600}
        open={showDetailDrawer}
        onClose={() => setShowDetailDrawer(false)}
      >
        {selectedItem && (
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            
            {/* 基本信息 */}
            <Card size="small" title="基本信息">
              <Row gutter={[16, 8]}>
                <Col span={8}>
                  <Text type="secondary">相关性分数:</Text>
                </Col>
                <Col span={16}>
                  <Progress 
                    percent={Math.round(selectedItem.score * 100)}
                    size="small"
                    status={selectedItem.score >= 0.8 ? 'success' : 'normal'}
                  />
                </Col>
                
                {selectedItem.file_path && (
                  <>
                    <Col span={8}>
                      <Text type="secondary">文件路径:</Text>
                    </Col>
                    <Col span={16}>
                      <Text code copyable>
                        {selectedItem.file_path}
                      </Text>
                    </Col>
                  </>
                )}
                
                {selectedItem.content_type && (
                  <>
                    <Col span={8}>
                      <Text type="secondary">内容类型:</Text>
                    </Col>
                    <Col span={16}>
                      <Tag color="blue">
                        {selectedItem.content_type}
                      </Tag>
                    </Col>
                  </>
                )}
              </Row>
            </Card>

            {/* 内容展示 */}
            <Card size="small" title="内容">
              <div 
                style={{ 
                  maxHeight: 400, 
                  overflow: 'auto',
                  lineHeight: 1.6,
                  whiteSpace: 'pre-wrap'
                }}
                dangerouslySetInnerHTML={{ 
                  __html: highlightContent(selectedItem.content, query) 
                }}
              />
            </Card>

            {/* 元数据 */}
            {Object.keys(selectedItem.metadata || {}).length > 0 && (
              <Card size="small" title="元数据">
                <pre style={{ 
                  background: '#f5f5f5', 
                  padding: 12, 
                  borderRadius: 4,
                  fontSize: 12
                }}>
                  {JSON.stringify(selectedItem.metadata, null, 2)}
                </pre>
              </Card>
            )}

            {/* 操作按钮 */}
            <Space style={{ width: '100%', justifyContent: 'center' }}>
              <Button 
                icon={<CopyOutlined />}
                onClick={() => handleCopyContent(selectedItem.content)}
              >
                复制内容
              </Button>
              {selectedItem.file_path && (
                <Button 
                  icon={<LinkOutlined />}
                  onClick={() => handleCopyContent(selectedItem.file_path!)}
                >
                  复制路径
                </Button>
              )}
            </Space>
          </Space>
        )}
      </Drawer>

      {/* 分享模态框 */}
      <Modal
        title="分享搜索结果"
        open={showShareModal}
        onCancel={() => setShowShareModal(false)}
        footer={[
          <Button key="cancel" onClick={() => setShowShareModal(false)}>
            取消
          </Button>,
          <Button 
            key="copy" 
            type="primary"
            onClick={() => {
              handleCopyContent(shareUrl);
              setShowShareModal(false);
            }}
          >
            复制链接
          </Button>
        ]}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Text>分享此搜索结果给他人:</Text>
          <TextArea
            value={shareUrl}
            readOnly
            rows={3}
            style={{ resize: 'none' }}
          />
        </Space>
      </Modal>

    </div>
  );
};

export default RagResultsList;